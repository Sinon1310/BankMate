"""
Compare similarity vs MMR ranking on the evaluation dataset.

Reuses app.evaluate_rag's dataset loader and reads policy IDs and
sections from chunk metadata. LangChain's MMR retriever returns no
distances, so this comparison covers ranking quality on in-scope
questions only (top-k, no threshold); out-of-scope behaviour is measured
by `python -m app.evaluate_rag` with the production threshold.

Run: python -m app.compare_retrieval
"""

from app.evaluate_rag import IN_SCOPE, TOP_K, load_evaluation_dataset
from app.retriever import get_retriever


def evaluate_strategy(name, retriever, dataset):
    hits = relevant = retrieved = 0
    sections_found = sections_expected = 0

    for row in dataset:
        documents = retriever.invoke(row["question"])[:TOP_K]
        found = [(d.metadata.get("policy_id"), d.metadata.get("section")) for d in documents]
        expected = set(row["expected_policy_ids"])

        matches = sum(1 for policy_id, _ in found if policy_id in expected)
        hits += bool(matches)
        relevant += matches
        retrieved += len(found)
        sections_expected += len(row["expected_sections"])
        sections_found += sum(
            1 for section in row["expected_sections"]
            if any((policy_id, section) in found for policy_id in expected)
        )

    return {
        "strategy": name,
        "questions": len(dataset),
        "hit_rate": round(hits / len(dataset), 3) if dataset else None,
        "precision": round(relevant / retrieved, 3) if retrieved else None,
        "section_recall": round(sections_found / sections_expected, 3) if sections_expected else None,
    }


def main():
    dataset = [row for row in load_evaluation_dataset() if row["category"] in IN_SCOPE]

    results = [
        evaluate_strategy("similarity", get_retriever("similarity"), dataset),
        evaluate_strategy("mmr", get_retriever("mmr"), dataset),
    ]

    print(f"\nRANKING COMPARISON (in-scope questions, top {TOP_K}, no threshold)")
    for result in results:
        print(f"  {result['strategy']:<11} hit {result['hit_rate']}  precision {result['precision']}  "
              f"section recall {result['section_recall']}  (n={result['questions']})")


if __name__ == "__main__":
    main()
