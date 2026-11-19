"""
Policy index lifecycle tests (Stage 5.3).

Real ChromaDB in pytest temp directories with deterministic fake
embeddings: no model download, and the repository's chroma_db/ is never
touched.
"""

import hashlib
import json
import math
import threading

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import app.retriever as retriever
import app.tools as tools
from app import assistant, errors
from app import vector_store as vs
from app.document_loader import split_documents


def policy_doc(policy_id, name, body, source):
    text = f"DOCUMENT: {name}\nDOCUMENT ID: {policy_id}\nVERSION: 1.0\n\n{body}"
    return Document(page_content=text, metadata={"source": source})


DOCS_V1 = [
    policy_doc("POL-101", "TEST REFUND POLICY",
               "PURPOSE:\nDescribes refunds.\n\nPROCESSING:\nRefunds are processed in the workflow.\n",
               "/x/test_refund.txt"),
    policy_doc("POL-102", "TEST SECURITY POLICY",
               "CUSTOMER PROTECTION:\nNever share a PIN or OTP.\n",
               "/x/test_security.txt"),
]
DOCS_V2 = [DOCS_V1[0], policy_doc("POL-102", "TEST SECURITY POLICY",
                                  "CUSTOMER PROTECTION:\nNever share a PIN, OTP or password.\n",
                                  "/x/test_security.txt")]


class FakeEmbeddings:
    """Deterministic 16-dim unit vectors; can be made to fail."""

    def __init__(self, fail=False):
        self.fail = fail
        self.documents_embedded = 0

    def _vector(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        values = [b / 255.0 + 0.01 for b in digest[:16]]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("embedding service down")
        self.documents_embedded += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture(autouse=True)
def clean_state():
    vs.reset_index_state()
    yield
    vs.reset_index_state()


def ensure(tmp_path, docs=DOCS_V1, embeddings=None, **kwargs):
    return vs.ensure_policy_index(persist_dir=tmp_path, embeddings=embeddings or FakeEmbeddings(),
                                  documents=docs, embedding_id="fake", **kwargs)


def collections(tmp_path):
    return sorted(vs.get_client(tmp_path).list_collections())


def final_collection(tmp_path):
    return vs.get_client(tmp_path).get_collection(vs.COLLECTION_NAME)


def expected_count(docs=DOCS_V1):
    return len(split_documents(docs))


# ---------- Missing / empty / valid ----------

def test_missing_store_is_built_and_validated(tmp_path):
    embeddings = FakeEmbeddings()
    status = ensure(tmp_path / "fresh", embeddings=embeddings)

    assert status.state == "valid" and status.built is True
    assert status.count == expected_count() == embeddings.documents_embedded
    assert collections(tmp_path / "fresh") == [vs.COLLECTION_NAME]
    metadata = final_collection(tmp_path / "fresh").metadata
    assert metadata[vs.STATE_KEY] == vs.STATE_COMPLETE
    assert metadata["format_version"] == vs.INDEX_FORMAT_VERSION
    assert metadata["chunk_count"] == expected_count()


def test_empty_collection_is_rebuilt(tmp_path):
    vs.get_client(tmp_path).create_collection(vs.COLLECTION_NAME, metadata={vs.STATE_KEY: vs.STATE_COMPLETE})
    assert vs.inspect_index(tmp_path, split_documents(DOCS_V1), "fake").state == "empty"

    status = ensure(tmp_path)
    assert status.state == "valid" and status.built
    assert final_collection(tmp_path).count() == expected_count()


def test_existing_valid_index_is_reused_without_embedding(tmp_path):
    ensure(tmp_path)
    vs.reset_index_state()                  # simulate a new application launch

    embeddings = FakeEmbeddings()
    status = ensure(tmp_path, embeddings=embeddings)

    assert status.state == "valid" and status.built is False
    assert embeddings.documents_embedded == 0


def test_repeated_initialisation_in_one_process_is_cached(tmp_path):
    first = ensure(tmp_path)
    embeddings = FakeEmbeddings()
    second = ensure(tmp_path, embeddings=embeddings)

    assert second is first
    assert embeddings.documents_embedded == 0
    assert vs.is_index_ready(tmp_path)


def test_concurrent_initialisation_builds_once(tmp_path):
    embeddings = FakeEmbeddings()
    barrier = threading.Barrier(4, timeout=10)
    results = []

    def init():
        barrier.wait()
        results.append(ensure(tmp_path, embeddings=embeddings))

    threads = [threading.Thread(target=init) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(results) == 4 and all(r.state == "valid" for r in results)
    assert embeddings.documents_embedded == expected_count()     # one build only


# ---------- Failures and partial builds ----------

def test_failed_build_raises_and_leaves_no_valid_looking_index(tmp_path):
    with pytest.raises(vs.PolicyIndexError) as caught:
        ensure(tmp_path, embeddings=FakeEmbeddings(fail=True))

    assert "embedding service down" in str(caught.value)
    assert collections(tmp_path) == []                   # no final, no leftover temp
    assert not vs.is_index_ready(tmp_path)


def test_failed_rebuild_keeps_existing_index_untouched(tmp_path):
    ensure(tmp_path)
    old = final_collection(tmp_path)
    old_fingerprint, old_count = old.metadata["fingerprint"], old.count()
    vs.reset_index_state()

    with pytest.raises(vs.PolicyIndexError):
        ensure(tmp_path, docs=DOCS_V2, embeddings=FakeEmbeddings(fail=True))

    kept = final_collection(tmp_path)
    assert kept.metadata["fingerprint"] == old_fingerprint and kept.count() == old_count
    assert vs.BUILD_COLLECTION_NAME not in collections(tmp_path)


def test_interrupted_build_leftover_is_discarded_and_rebuilt(tmp_path):
    building = vs.get_client(tmp_path).create_collection(
        vs.BUILD_COLLECTION_NAME, metadata={vs.STATE_KEY: vs.STATE_BUILDING})
    building.add(ids=["partial"], embeddings=[[0.25] * 16], documents=["half"], metadatas=[{"policy_id": "POL-101"}])

    status = ensure(tmp_path)

    assert status.state == "valid" and status.built
    assert collections(tmp_path) == [vs.COLLECTION_NAME]


def test_collection_not_marked_complete_is_not_trusted(tmp_path):
    ensure(tmp_path)
    final_collection(tmp_path).modify(metadata={vs.STATE_KEY: vs.STATE_BUILDING})
    vs.reset_index_state()

    assert vs.inspect_index(tmp_path, split_documents(DOCS_V1), "fake").state == "incomplete"
    assert ensure(tmp_path).built


def test_invalid_metadata_is_detected_and_rebuilt(tmp_path):
    ensure(tmp_path)
    collection = final_collection(tmp_path)
    some_id = collection.get()["ids"][0]
    # Chroma merges metadata on update, so blank the field explicitly.
    collection.update(ids=[some_id], metadatas=[{"policy_id": ""}])
    vs.reset_index_state()

    status = vs.inspect_index(tmp_path, split_documents(DOCS_V1), "fake")
    assert status.state == "invalid" and "metadata" in status.reason
    assert ensure(tmp_path).built


def test_missing_chunk_is_detected_and_rebuilt(tmp_path):
    ensure(tmp_path)
    collection = final_collection(tmp_path)
    collection.delete(ids=[collection.get()["ids"][0]])
    vs.reset_index_state()

    assert vs.inspect_index(tmp_path, split_documents(DOCS_V1), "fake").state == "invalid"
    rebuilt = ensure(tmp_path)
    assert rebuilt.built and rebuilt.count == expected_count()


def test_changed_policy_files_trigger_rebuild(tmp_path):
    ensure(tmp_path)
    vs.reset_index_state()

    assert vs.inspect_index(tmp_path, split_documents(DOCS_V2), "fake").state == "outdated"
    status = ensure(tmp_path, docs=DOCS_V2)
    assert status.built


def test_changed_embedding_model_triggers_rebuild(tmp_path):
    ensure(tmp_path)
    vs.reset_index_state()
    assert vs.inspect_index(tmp_path, split_documents(DOCS_V1), "another-model").state == "outdated"


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c.metadata.pop("policy_id"), "missing metadata"),
    (lambda c: c.metadata.update(policy_id="BAD-1"), "Invalid policy ID"),
    (lambda c: c.metadata.update(chunk_id="dup"), "Duplicate chunk ID"),
])
def test_chunk_validation_rejects_bad_metadata(mutate, message):
    chunks = split_documents(DOCS_V1)
    if message == "Duplicate chunk ID":
        for chunk in chunks[:2]:
            mutate(chunk)
    else:
        mutate(chunks[0])

    with pytest.raises(vs.PolicyIndexError, match=message):
        vs.validate_chunks(chunks)


def test_no_policy_documents_is_a_clear_error(tmp_path):
    with pytest.raises(vs.PolicyIndexError, match="No policy chunks"):
        ensure(tmp_path, docs=[])


# ---------- Unavailable index is never "no relevant policy" ----------

def _unavailable(*args, **kwargs):
    raise vs.PolicyIndexError("index unavailable for test")


def test_search_raises_instead_of_returning_no_results(monkeypatch):
    monkeypatch.setattr(retriever, "ensure_policy_index", _unavailable)

    with pytest.raises(vs.PolicyIndexError):
        retriever.search_relevant_documents("refund?")

    with pytest.raises(vs.PolicyIndexError):
        tools.search_banking_policies.invoke({"question": "refund?"})


def test_assistant_reports_index_unavailable_not_no_policy(monkeypatch):
    monkeypatch.setattr(retriever, "ensure_policy_index", _unavailable)
    session_id = "idx-unavailable"

    class Model:
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[
                {"name": "search_banking_policies", "args": {"question": "refund"}, "id": "c1"}])

    try:
        with pytest.raises(errors.AssistantError) as caught:
            assistant.ask_bankmate_with_sources(
                "What happens when a refund is approved?", session_id=session_id,
                model=Model(), tools={"search_banking_policies": tools.search_banking_policies})

        assert caught.value.kind == errors.INDEX_UNAVAILABLE
        assert "policy index isn't available" in caught.value.user_message
        assert "NO_RELEVANT" not in caught.value.user_message
        assert session_id not in assistant.conversation_sessions          # nothing committed
    finally:
        for store in (assistant.conversation_sessions, assistant.grounding_context,
                      assistant.session_last_active, assistant.conversation_versions,
                      assistant.request_results, assistant.requests_in_flight):
            store.pop(session_id, None)


def test_tool_payload_shape_is_unchanged_when_index_is_available(monkeypatch):
    monkeypatch.setattr(tools, "search_relevant_documents", lambda *a, **k: [])
    payload = json.loads(tools.search_banking_policies.invoke({"question": "home loan rate"}))
    assert payload["status"] == "NO_RELEVANT_POLICY_FOUND"
