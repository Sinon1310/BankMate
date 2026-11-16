
import csv
from pathlib import Path
from collections import defaultdict

from app.retriever import search_with_scores


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = PROJECT_ROOT / "data" / "rag_evaluation.csv"

TOP_K = 3
MAX_DISTANCE = 1.4

POLICY_MAPPING = {
    "card_dispute_policy.txt": "POL-001",
    "refund_policy.txt": "POL-002",
    "transaction_security_policy.txt": "POL-003"
}



def load_evaluation_dataset():
    with open(EVALUATION_FILE, "r", encoding="utf-8-sig", newline="") as file:
        lines = (line for line in file if line.strip())
        reader = csv.DictReader(lines)

        if not reader.fieldnames:
            raise ValueError("Evaluation CSV is empty or missing headers.")

        reader.fieldnames = [
            name.strip() for name in reader.fieldnames
        ]

        required_columns = {
            "question",
            "expected_policy_id"
        }

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}. "
                f"Found: {reader.fieldnames}"
            )

        return list(reader)


def evaluate_question(question, expected_policy_id):
    results = search_with_scores(question, k=TOP_K)

    # Keep only results within the current distance threshold
    accepted_results = [
        (document, distance)
        for document, distance in results
        if distance <= MAX_DISTANCE
    ]

    retrieved_policy_ids = []

    for document, distance in accepted_results:
        source = Path(
            document.metadata.get("source", "")
        ).name

        policy_id = POLICY_MAPPING.get(source)

        if policy_id:
            retrieved_policy_ids.append(policy_id)

    if expected_policy_id == "NONE":
        return {
            "question": question,
            "expected": expected_policy_id,
            "retrieved": retrieved_policy_ids,
            "hit": None,
            "precision": None,
            "false_positive": len(accepted_results) > 0
        }

    relevant_count = sum(
        1 for policy_id in retrieved_policy_ids
        if policy_id == expected_policy_id
    )

    hit = relevant_count > 0

    precision = relevant_count / TOP_K

    return {
        "question": question,
        "expected": expected_policy_id,
        "retrieved": retrieved_policy_ids,
        "hit": hit,
        "precision": precision,
        "false_positive": False
    }


def main():
    dataset = load_evaluation_dataset()

    positive_results = []
    negative_results = []

    print("\nBANKMATE - RAG EVALUATION")
    print("=" * 70)

    for row in dataset:
        result = evaluate_question(
            row["question"],
            row["expected_policy_id"]
        )

        if result["expected"] == "NONE":
            negative_results.append(result)
        else:
            positive_results.append(result)

        print("\nQuestion:", result["question"])
        print("Expected:", result["expected"])
        print("Retrieved:", result["retrieved"])

        if result["expected"] == "NONE":
            print(
                "False Positive:",
                result["false_positive"]
            )
        else:
            print("Hit:", result["hit"])
            print(
                "Precision@3:",
                round(result["precision"], 2)
            )

    total_positive = len(positive_results)

    recall = (
        sum(result["hit"] for result in positive_results)
        / total_positive
        if total_positive else 0.0
    )

    precision_values = [
        result["precision"]
        for result in positive_results
    ]

    average_precision = (
        sum(precision_values) / len(precision_values)
        if precision_values else 0.0
    )

    false_positives = sum(
        result["false_positive"]
        for result in negative_results
    )

    false_positive_rate = (
        false_positives / len(negative_results)
        if negative_results else 0.0
    )

    print("\n" + "=" * 70)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 70)

    print("Positive Questions:", total_positive)
    print("Negative Questions:", len(negative_results))
    print("Recall@3:", round(recall, 3))
    print("Average Precision@3:", round(average_precision, 3))
    print("False Positives:", false_positives)
    print(
        "False Positive Rate:",
        round(false_positive_rate, 3)
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
