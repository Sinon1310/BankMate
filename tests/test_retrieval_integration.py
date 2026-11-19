"""
Retrieval integration tests (Stage 5.3) with the real embedding model.

A fresh index is built in a temporary directory (the repository's
chroma_db/ is never touched), which also exercises first-launch
initialisation. Skipped if the embedding model cannot be loaded (e.g.
offline without a model cache). Searches use the production k and
threshold.
"""

import pytest

from app import vector_store as vs
from app.evaluate_rag import load_evaluation_dataset, run_evaluation
from app.retriever import POLICY_MAX_DISTANCE, POLICY_SEARCH_K


@pytest.fixture(scope="module")
def fresh_store(tmp_path_factory):
    try:
        from app.embeddings import get_embedding_model
        embeddings = get_embedding_model()
        embeddings.embed_query("warm-up")
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {error}")

    path = tmp_path_factory.mktemp("fresh_index")
    status = vs.ensure_policy_index(persist_dir=path)
    assert status.state == "valid" and status.built is True
    return vs.open_policy_store(path), path


def search(store, question):
    return [
        (doc.metadata, distance)
        for doc, distance in store.similarity_search_with_score(question, k=POLICY_SEARCH_K)
        if distance <= POLICY_MAX_DISTANCE
    ]


def test_fresh_index_has_expected_chunks_and_is_reused(fresh_store):
    _, path = fresh_store
    expected = len(vs.expected_chunks())

    assert vs.get_client(path).get_collection(vs.COLLECTION_NAME).count() == expected
    vs.reset_index_state()
    relaunch = vs.ensure_policy_index(persist_dir=path)
    assert relaunch.state == "valid" and relaunch.built is False


@pytest.mark.parametrize("question", [
    "What is the bank's home loan interest rate?",
    "Who is the chief executive of the bank?",
    "How do I open a new savings account?",
    "What is the daily ATM withdrawal limit?",
])
def test_out_of_scope_questions_retrieve_nothing(fresh_store, question):
    store, _ = fresh_store
    assert search(store, question) == []


@pytest.mark.parametrize("question, policy_id", [
    ("Someone used my card without my permission. What should I do?", "POL-001"),
    ("I was charged twice for the same purchase. Can I get my money back?", "POL-002"),
    ("How can I follow the progress of my refund?", "POL-002"),
])
def test_paraphrased_questions_find_the_right_policy(fresh_store, question, policy_id):
    store, _ = fresh_store
    results = search(store, question)
    assert results and results[0][0]["policy_id"] == policy_id


@pytest.mark.parametrize("question, policy_id, sections", [
    pytest.param(
        "What is checked during refund validation and how is the refund then processed?",
        "POL-002", {"VALIDATION", "PROCESSING"},
        marks=pytest.mark.xfail(
            strict=True,
            reason="Known limitation (Stage 5.3): the generic PURPOSE chunk takes a top-3 "
                   "slot, crowding out VALIDATION. Strict, so a future fix is noticed.",
        ),
    ),
    ("How is a card dispute raised and how is its status tracked?",
     "POL-001", {"DISPUTE INITIATION", "STATUS TRACKING"}),
])
def test_multi_section_questions_retrieve_both_sections(fresh_store, question, policy_id, sections):
    store, _ = fresh_store
    found = {meta["section"] for meta, _ in search(store, question) if meta["policy_id"] == policy_id}
    assert sections <= found


def test_no_header_chunk_takes_a_retrieval_slot(fresh_store):
    store, _ = fresh_store
    for row in load_evaluation_dataset():
        assert all(meta["section"] != "GENERAL" for meta, _ in search(store, row["question"]))


def test_fresh_index_evaluation_meets_current_baseline(fresh_store):
    """Regression guard at the measured Stage 5.3 baseline (not a tuning target)."""

    store, _ = fresh_store

    def store_search(question, k):
        return [(doc.metadata, d) for doc, d in store.similarity_search_with_score(question, k=k)]

    _, summary, _ = run_evaluation(store_search)

    assert summary["in_scope"]["hit_rate"] >= 0.93
    assert summary["in_scope"]["precision"] >= 0.85
    assert summary["no_answer"]["false_positive_rate"] <= 0.12
