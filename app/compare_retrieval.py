
import csv
from pathlib import Path
from collections import defaultdict

from app.retriever import get_retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = PROJECT_ROOT / "data" / "rag_evaluation.csv"

TOP_K = 3


def load_evaluation_dataset():
    with open(EVALUATION_FILE, "r", encoding="utf-8-sig", newline="") as file:
        lines = (line for line in file if line.strip())
        reader = csv.DictReader(lines)

        if not reader.fieldnames:
            raise ValueError("Evaluation CSV is missing headers.")

        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        return [
            row for row in reader
            if row.get("question") and row.get("expected_policy_id")
        ]


def get_policy_id(document):
    source = document.metadata.get("source", "")
    filename = Path(source).name

    policy_mapping = {
        "card_dispute_policy.txt": "POL-001",
        "refund_policy.txt": "POL-002",
        "transaction_security_policy.txt": "POL-003"
    }

    return policy_mapping.get(filename, "UNKNOWN")


def evaluate_strategy(strategy_name, retriever, dataset):
    positive_hits = 0
    positive_precisions = []
    negative_false_positives = 0
    negative_queries = 0

    print(f"\n{'=' * 75}")
    print(f"EVALUATING: {strategy_name.upper()}")
    print("=" * 75)

    for row in dataset:
        question = row["question"].strip()
        expected = row["expected_policy_id"].strip()

        documents = retriever.invoke(question)[:TOP_K]
        retrieved_ids = [get_policy_id(doc) for doc in documents]

        if expected == "NONE":
            negative_queries += 1

            if documents:
                negative_false_positives += 1

            print(f"\nQuestion: {question}")
            print(f"Expected: NONE")
            print(f"Retrieved: {retrieved_ids}")
            continue

        relevant_count = retrieved_ids.count(expected)
        hit = relevant_count > 0

        if hit:
            positive_hits += 1

        precision = relevant_count / TOP_K
        positive_precisions.append(precision)

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Hit: {hit} | Precision@3: {precision:.3f}")

    positive_count = len(positive_precisions)

    hit_rate = positive_hits / positive_count if positive_count else 0
    avg_precision = (
        sum(positive_precisions) / positive_count
        if positive_count else 0
    )
    false_positive_rate = (
        negative_false_positives / negative_queries
        if negative_queries else 0
    )

    return {
        "strategy": strategy_name,
        "hit_rate": hit_rate,
        "precision": avg_precision,
        "false_positives": negative_false_positives,
        "false_positive_rate": false_positive_rate
    }


def main():
    dataset = load_evaluation_dataset()

    similarity_retriever = get_retriever("similarity")
    mmr_retriever = get_retriever("mmr")

    results = [
        evaluate_strategy("Similarity", similarity_retriever, dataset),
        evaluate_strategy("MMR", mmr_retriever, dataset)
    ]

    print("\n" + "=" * 75)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 75)

    for result in results:
        print(f"\nStrategy: {result['strategy']}")
        print(f"Hit Rate@3: {result['hit_rate']:.3f}")
        print(f"Average Precision@3: {result['precision']:.3f}")
        print(f"False Positives: {result['false_positives']}")
        print(
            f"False Positive Rate: "
            f"{result['false_positive_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
