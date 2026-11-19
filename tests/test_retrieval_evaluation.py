"""
Retrieval evaluation tests (Stage 5.3): production/evaluation threshold
consistency, dataset integrity and metric arithmetic. No index needed.
"""

import csv
from collections import Counter

import pytest

import app.evaluate_rag as evaluation
import app.retriever as retriever
import app.tools as tools
from app.document_loader import load_policy_documents, split_documents


def chunk_sections():
    sections = {}
    for chunk in split_documents(load_policy_documents()):
        sections.setdefault(chunk.metadata["policy_id"], set()).add(chunk.metadata["section"])
    return sections


# ---------- Threshold consistency ----------

def test_evaluation_uses_production_retrieval_settings():
    assert evaluation.MAX_DISTANCE is retriever.POLICY_MAX_DISTANCE
    assert evaluation.TOP_K == retriever.POLICY_SEARCH_K
    assert evaluation.MAX_DISTANCE == 1.2 and evaluation.TOP_K == 3    # unchanged in 5.3


def test_policy_tool_searches_with_production_settings(monkeypatch):
    captured = {}

    def fake_search(question, k, max_distance):
        captured.update(k=k, max_distance=max_distance)
        return []

    monkeypatch.setattr(tools, "search_relevant_documents", fake_search)
    tools.search_banking_policies.invoke({"question": "anything"})

    assert captured == {"k": evaluation.TOP_K, "max_distance": evaluation.MAX_DISTANCE}


def test_search_default_threshold_is_the_production_one():
    import inspect

    defaults = inspect.signature(retriever.search_relevant_documents).parameters
    assert defaults["max_distance"].default == retriever.POLICY_MAX_DISTANCE
    assert defaults["k"].default == retriever.POLICY_SEARCH_K


# ---------- Dataset integrity ----------

def test_dataset_size_and_category_coverage():
    rows = evaluation.load_evaluation_dataset()
    counts = Counter(row["category"] for row in rows)

    assert 30 <= len(rows) <= 40
    assert set(counts) == set(evaluation.CATEGORIES)
    assert counts["out_of_scope"] + counts["near_miss"] >= 8
    assert len({row["question"] for row in rows}) == len(rows)          # no duplicates


def test_dataset_references_real_policies_and_sections():
    sections = chunk_sections()

    for row in evaluation.load_evaluation_dataset():
        if row["category"] in evaluation.NO_ANSWER:
            assert row["expected_policy_ids"] == [], row["question"]
            continue

        assert row["expected_policy_ids"], row["question"]
        for policy_id in row["expected_policy_ids"]:
            assert policy_id in sections, (row["question"], policy_id)
        for section in row["expected_sections"]:
            assert any(section in sections[p] for p in row["expected_policy_ids"]), (row["question"], section)

    multi = [r for r in evaluation.load_evaluation_dataset() if r["category"] == "multi_section"]
    assert multi and all(len(r["expected_sections"]) >= 2 for r in multi)


def test_dataset_rejects_unknown_category(tmp_path):
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "expected_policy_ids", "category", "expected_sections"])
        writer.writerow(["Q?", "POL-001", "guesswork", ""])

    with pytest.raises(ValueError, match="Unknown category"):
        evaluation.load_evaluation_dataset(path)


# ---------- Metric arithmetic ----------

def meta(policy_id, section="S"):
    return {"policy_id": policy_id, "section": section}


def row(expected, category="direct", sections=()):
    return {"question": "q-" + "-".join(expected) + category, "category": category,
            "expected_policy_ids": list(expected), "expected_sections": list(sections)}


def test_score_applies_threshold_and_counts_relevance():
    record = evaluation.score_question(
        row(["POL-002"], sections=["PROCESSING", "TRACKING"]),
        [(meta("POL-002", "PROCESSING"), 0.4), (meta("POL-001"), 0.9), (meta("POL-002", "TRACKING"), 1.25)],
    )

    assert record["retrieved"] == ["POL-002", "POL-001"]        # 1.25 is beyond 1.2
    assert record["hit"] is True
    assert (record["relevant"], record["irrelevant"]) == (1, 1)
    assert (record["sections_found"], record["sections_expected"]) == (1, 2)


def test_out_of_scope_false_positive_and_clean_result():
    flagged = evaluation.score_question(row([], "near_miss"), [(meta("POL-001"), 1.05)])
    clean = evaluation.score_question(row([], "out_of_scope"), [(meta("POL-001"), 1.4)])

    assert flagged["false_positive"] is True and flagged["hit"] is None
    assert clean["false_positive"] is False and clean["retrieved"] == []


def test_summary_reports_in_scope_and_out_of_scope_separately():
    records = [
        evaluation.score_question(row(["POL-001"]), [(meta("POL-001"), 0.5), (meta("POL-003"), 0.9)]),
        evaluation.score_question(row(["POL-002"], "paraphrase"), []),
        evaluation.score_question(row([], "out_of_scope"), [(meta("POL-001"), 1.0)]),
        evaluation.score_question(row([], "near_miss"), [(meta("POL-001"), 1.5)]),
    ]
    summary = evaluation.summarize(records)

    assert summary["in_scope"] == {"n": 2, "hit_rate": 0.5, "precision": 0.5, "irrelevant_rate": 0.5,
                                   "section_recall": None, "miss_rate": 0.5}
    assert summary["no_answer"] == {"n": 2, "false_positive_rate": 0.5}
    assert summary["by_category"]["paraphrase"]["hit_rate"] == 0.0


def test_sweep_reuses_the_same_raw_results():
    dataset = [row(["POL-001"]), row([], "near_miss")]
    calls = []

    def search(question, k):
        calls.append(question)
        return [(meta("POL-001"), 1.15)]

    records, summary, sweep = evaluation.run_evaluation(search, dataset)

    assert len(calls) == len(dataset)                   # one search per question
    assert sweep[1.1]["in_scope"]["hit_rate"] == 0.0 and sweep[1.2]["in_scope"]["hit_rate"] == 1.0
    assert sweep[1.1]["no_answer"]["false_positive_rate"] == 0.0
    assert summary == sweep[evaluation.MAX_DISTANCE]
