"""
Retrieval evaluation against data/rag_evaluation.csv.

Uses the production retrieval settings (k and distance threshold come
from app.retriever, not copies) and reads policy IDs and sections from
each retrieved chunk's metadata.

Dataset columns:
- question
- expected_policy_ids: policy IDs separated by ";" (any match counts), or NONE
- category: direct | paraphrase | section | multi_section | ambiguous
            (in scope) or out_of_scope | near_miss (should retrieve nothing)
- expected_sections: optional section names separated by ";"

Metrics (all on results within the production threshold, top-k):
- Hit@k: share of in-scope questions with at least one chunk from an
  expected policy.
- Precision@k: relevant chunks / all retrieved chunks, over all in-scope
  questions (micro-average). Irrelevant-retrieval rate = 1 - precision.
- Section recall: share of expected sections retrieved (questions that
  list sections).
- Miss rate: in-scope questions that retrieved nothing.
- False-positive rate: out-of-scope / near-miss questions that retrieved
  anything.

Run: python -m app.evaluate_rag
     python -m app.evaluate_rag --json evals/results/retrieval.json   (also save a JSON report)
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.retriever import POLICY_MAX_DISTANCE, POLICY_SEARCH_K


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = PROJECT_ROOT / "data" / "rag_evaluation.csv"

TOP_K = POLICY_SEARCH_K
MAX_DISTANCE = POLICY_MAX_DISTANCE

IN_SCOPE = ("direct", "paraphrase", "section", "multi_section", "ambiguous")
NO_ANSWER = ("out_of_scope", "near_miss")
CATEGORIES = IN_SCOPE + NO_ANSWER

SWEEP_THRESHOLDS = (1.0, 1.1, 1.2, 1.3, 1.4)


def _split(value):
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def load_evaluation_dataset(path=EVALUATION_FILE):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(line for line in file if line.strip())

        if not reader.fieldnames:
            raise ValueError("Evaluation CSV is empty or missing headers.")

        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        required = {"question", "expected_policy_ids", "category"}
        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(f"Missing CSV columns: {missing}. Found: {reader.fieldnames}")

        rows = []

        for row in reader:
            category = (row.get("category") or "").strip()
            expected = _split(row.get("expected_policy_ids"))

            if category not in CATEGORIES:
                raise ValueError(f"Unknown category {category!r} for {row.get('question')!r}")

            rows.append({
                "question": row["question"].strip(),
                "category": category,
                "expected_policy_ids": [] if expected == ["NONE"] else expected,
                "expected_sections": _split(row.get("expected_sections")),
            })

        return rows


def score_question(row, results, max_distance=MAX_DISTANCE, k=TOP_K):
    """
    results: [(metadata_dict, distance), ...] ordered by distance (raw
    top-k from the vector store). Returns a per-question record.
    """

    kept = [(meta, distance) for meta, distance in results[:k] if distance <= max_distance]
    retrieved = [meta.get("policy_id") for meta, _ in kept]
    sections = {(meta.get("policy_id"), meta.get("section")) for meta, _ in kept}
    expected = set(row["expected_policy_ids"])

    relevant = sum(1 for policy_id in retrieved if policy_id in expected)
    expected_sections = [
        section for section in row["expected_sections"]
        if any((policy_id, section) in sections for policy_id in expected)
    ]

    return {
        "question": row["question"],
        "category": row["category"],
        "expected": sorted(expected),
        "retrieved": retrieved,
        "distances": [round(distance, 3) for _, distance in kept],
        "top_distance": round(results[0][1], 3) if results else None,
        "hit": bool(relevant) if expected else None,
        "relevant": relevant,
        "irrelevant": len(retrieved) - relevant,
        "sections_expected": len(row["expected_sections"]),
        "sections_found": len(expected_sections),
        "false_positive": bool(kept) if not expected else None,
    }


def summarize(records):
    in_scope = [r for r in records if r["category"] in IN_SCOPE]
    no_answer = [r for r in records if r["category"] in NO_ANSWER]
    retrieved = sum(r["relevant"] + r["irrelevant"] for r in in_scope)
    relevant = sum(r["relevant"] for r in in_scope)
    with_sections = [r for r in in_scope if r["sections_expected"]]

    def rate(numerator, denominator):
        return round(numerator / denominator, 3) if denominator else None

    by_category = {}
    for category in CATEGORIES:
        rows = [r for r in records if r["category"] == category]
        if not rows:
            continue
        if category in IN_SCOPE:
            by_category[category] = {"n": len(rows), "hit_rate": rate(sum(r["hit"] for r in rows), len(rows))}
        else:
            by_category[category] = {"n": len(rows),
                                     "false_positive_rate": rate(sum(r["false_positive"] for r in rows), len(rows))}

    return {
        "in_scope": {
            "n": len(in_scope),
            "hit_rate": rate(sum(r["hit"] for r in in_scope), len(in_scope)),
            "precision": rate(relevant, retrieved),
            "irrelevant_rate": rate(retrieved - relevant, retrieved),
            "section_recall": rate(sum(r["sections_found"] for r in with_sections),
                                   sum(r["sections_expected"] for r in with_sections)),
            "miss_rate": rate(sum(1 for r in in_scope if not r["retrieved"]), len(in_scope)),
        },
        "no_answer": {
            "n": len(no_answer),
            "false_positive_rate": rate(sum(r["false_positive"] for r in no_answer), len(no_answer)),
        },
        "by_category": by_category,
    }


def threshold_sweep(dataset, raw_results, thresholds=SWEEP_THRESHOLDS):
    """Diagnostic only: metrics at other thresholds from the same results."""

    return {
        threshold: summarize([score_question(row, raw_results[row["question"]], threshold)
                              for row in dataset])
        for threshold in thresholds
    }


def run_evaluation(search, dataset=None):
    """
    search(question, k) -> [(metadata, distance), ...]. Returns
    (records, summary, sweep). Raw results are fetched once per question.
    """

    dataset = load_evaluation_dataset() if dataset is None else dataset
    raw = {row["question"]: search(row["question"], TOP_K) for row in dataset}
    records = [score_question(row, raw[row["question"]]) for row in dataset]
    return records, summarize(records), threshold_sweep(dataset, raw)


def _store_search(question, k):
    from app.retriever import search_with_scores

    return [(document.metadata, distance) for document, distance in search_with_scores(question, k)]


def build_report(records, summary, sweep, metadata=None):
    """JSON-serialisable report of an evaluation run (same figures as the printed output)."""

    return {
        "run": {
            "kind": "retrieval",
            "settings": {"k": TOP_K, "max_distance": MAX_DISTANCE, "sweep_thresholds": list(SWEEP_THRESHOLDS)},
            **(metadata or {}),
        },
        "summary": summary,
        "sweep": {str(threshold): result for threshold, result in sweep.items()},
        "records": records,
    }


def _report_metadata(path=EVALUATION_FILE):
    """Reproducibility details: dataset hash, embedding model and policy index fingerprint."""

    import contextlib
    import io

    from app import vector_store

    with contextlib.redirect_stdout(io.StringIO()):
        chunks = vector_store.expected_chunks()
    return {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {"file": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()},
        "embedding_id": vector_store.EMBEDDING_ID,
        "policy_index_fingerprint": vector_store.fingerprint(chunks),
    }


def write_json_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrieval evaluation (production k and threshold).")
    parser.add_argument("--json", metavar="PATH", help="Also save the results as a JSON report")
    args = parser.parse_args(argv)

    records, summary, sweep = run_evaluation(_store_search)

    print(f"\nBANKMATE RETRIEVAL EVALUATION  (k={TOP_K}, max distance={MAX_DISTANCE}, production settings)")
    print("=" * 96)

    for record in records:
        if record["category"] in IN_SCOPE:
            outcome = "HIT " if record["hit"] else "MISS"
        else:
            outcome = "FP  " if record["false_positive"] else "OK  "
        print(f"{outcome} [{record['category']:<13}] {record['question'][:58]:<58} "
              f"-> {record['retrieved'] or '-'} {record['distances'] or ''} (top {record['top_distance']})")

    in_scope, no_answer = summary["in_scope"], summary["no_answer"]
    print("\nPolicy questions (in scope)")
    print(f"  questions            {in_scope['n']}")
    print(f"  Hit@{TOP_K}                {in_scope['hit_rate']}")
    print(f"  Precision@{TOP_K}          {in_scope['precision']}  (irrelevant-retrieval rate {in_scope['irrelevant_rate']})")
    print(f"  Section recall       {in_scope['section_recall']}")
    print(f"  Miss rate            {in_scope['miss_rate']}")
    print("\nOut-of-scope and near-miss questions")
    print(f"  questions            {no_answer['n']}")
    print(f"  False-positive rate  {no_answer['false_positive_rate']}")
    print("\nBy category")
    for category, values in summary["by_category"].items():
        print(f"  {category:<14} {values}")

    print("\nThreshold sweep (diagnostic only; production threshold unchanged)")
    print(f"  {'threshold':<10} {'hit':>6} {'precision':>10} {'section':>8} {'miss':>6} {'false+':>7}")
    for threshold, result in sweep.items():
        i, n = result["in_scope"], result["no_answer"]
        marker = "  <- production" if threshold == MAX_DISTANCE else ""
        print(f"  {threshold:<10} {i['hit_rate']:>6} {i['precision']:>10} {i['section_recall']:>8} "
              f"{i['miss_rate']:>6} {n['false_positive_rate']:>7}{marker}")

    if args.json:
        path = write_json_report(build_report(records, summary, sweep, _report_metadata()), args.json)
        print(f"\nJSON report: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
