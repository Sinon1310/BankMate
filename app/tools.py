
import json
from decimal import Decimal
from app.retriever import get_retriever
from langchain_core.tools import tool

from app.data_loader import (
    find_customer,
    find_transactions
)


@tool
def get_customer_profile(customer_id: str) -> str:
    """Retrieve customer name, city and account type using customer ID."""

    customer = find_customer(customer_id)

    if not customer:
        return json.dumps({
            "error": "Customer not found",
            "customer_id": customer_id
        })

    return json.dumps(customer)


@tool
def get_customer_transactions(customer_id: str) -> str:
    """Retrieve all transaction records for a customer ID."""

    customer = find_customer(customer_id)

    if not customer:
        return json.dumps({
            "error": "Customer not found",
            "customer_id": customer_id
        })

    records = find_transactions(customer_id)

    return json.dumps({
        "customer_id": customer_id,
        "transaction_count": len(records),
        "transactions": records
    })


@tool
def get_customer_metrics(customer_id: str) -> str:
    """Calculate debit, credit, net transaction flow and transaction count."""

    customer = find_customer(customer_id)

    if not customer:
        return json.dumps({
            "error": "Customer not found",
            "customer_id": customer_id
        })

    records = find_transactions(customer_id)

    debit = sum(
        (
            Decimal(t["amount"])
            for t in records
            if t["transaction_type"].lower() == "debit"
        ),
        Decimal("0")
    )

    credit = sum(
        (
            Decimal(t["amount"])
            for t in records
            if t["transaction_type"].lower() == "credit"
        ),
        Decimal("0")
    )

    return json.dumps({
        "customer_id": customer_id,
        "transaction_count": len(records),
        "total_debit_inr": str(debit),
        "total_credit_inr": str(credit),
        "net_transaction_flow_inr": str(credit - debit)
    })
from langchain_core.tools import tool



from app.retriever import search_relevant_documents


@tool
def search_banking_policies(question: str) -> str:
    """
    Search fictional banking policy documents for information
    about card disputes, refunds and transaction security.
    """

    results = search_relevant_documents(
    question,
    k=3,
    max_distance=1.3

    )

    if not results:
        return (
            "NO_RELEVANT_POLICY_FOUND: "
            "The available policy documents do not contain "
            "sufficiently relevant information to answer this question."
        )

    
    formatted_results = []

    for document, distance in results:

        content = document.page_content

        # Extract document metadata from the policy text
        lines = content.splitlines()

        policy_id = "Unknown"
        version = "Unknown"
        title = "Unknown Policy"

        for line in lines:

            if line.startswith("DOCUMENT ID:"):
                policy_id = line.split(":", 1)[1].strip()

            elif line.startswith("VERSION:"):
                version = line.split(":", 1)[1].strip()

            elif line.startswith("DOCUMENT:"):
                title = line.split(":", 1)[1].strip()

        # Some chunks don't contain the document header.
        # Read the original source file to obtain metadata.
        if "Unknown" in [policy_id, version, title]:

            source = document.metadata.get("source", "")

            try:
                with open(source, "r", encoding="utf-8") as file:
                    full_content = file.read()

                for line in full_content.splitlines():

                    if line.startswith("DOCUMENT ID:"):
                        policy_id = line.split(":", 1)[1].strip()

                    elif line.startswith("VERSION:"):
                        version = line.split(":", 1)[1].strip()

                    elif line.startswith("DOCUMENT:"):
                        title = line.split(":", 1)[1].strip()

            except OSError:
                pass

        formatted_results.append(
            f"Policy: {title}\n"
            f"Policy ID: {policy_id}\n"
            f"Version: {version}\n"
            f"Distance: {distance:.4f}\n"
            f"Evidence:\n{content}"
        )

    return "\n\n".join(formatted_results)
