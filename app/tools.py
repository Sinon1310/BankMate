
import json
from decimal import Decimal

from langchain_core.tools import tool

from app.data_loader import find_customer, find_transactions
from app.retriever import search_relevant_documents


@tool
def get_customer_profile(customer_id: str) -> str:
    """Retrieve a customer's profile using their customer ID."""

    customer = find_customer(customer_id)

    if not customer:
        return f"Customer {customer_id} was not found."

    return json.dumps(customer, ensure_ascii=False)


@tool
def get_customer_transactions(customer_id: str) -> str:
    """Retrieve all transactions belonging to a customer."""

    transactions = find_transactions(customer_id)

    if not transactions:
        return f"No transactions found for customer {customer_id}."

    return json.dumps(transactions, ensure_ascii=False)


@tool
def get_customer_metrics(customer_id: str) -> str:
    """Calculate debit, credit and net transaction flow for a customer."""

    transactions = find_transactions(customer_id)

    if not transactions:
        return f"No transactions found for customer {customer_id}."

    total_debit = Decimal("0")
    total_credit = Decimal("0")

    for transaction in transactions:
        amount = Decimal(transaction["amount"])

        if transaction["transaction_type"].lower() == "debit":
            total_debit += amount

        elif transaction["transaction_type"].lower() == "credit":
            total_credit += amount

    net_flow = total_credit - total_debit

    metrics = {
        "customer_id": customer_id,
        "total_debit": str(total_debit),
        "total_credit": str(total_credit),
        "net_transaction_flow": str(net_flow),
        "transaction_count": len(transactions),
        "currency": "INR"
    }

    return json.dumps(metrics, ensure_ascii=False)


@tool
def search_banking_policies(question: str) -> str:
    """
    Search fictional banking policy documents for information
    about card disputes, refunds and transaction security.
    Returns relevant policy evidence and source metadata.
    """

    results = search_relevant_documents(
        question,
        k=3,
        max_distance=1.2
    )

    if not results:
        return json.dumps({
            "status": "NO_RELEVANT_POLICY_FOUND",
            "message": (
                "The available policy documents do not contain "
                "sufficiently relevant information to answer this question."
            ),
            "sources": [],
            "evidence": []
        })

    evidence = []
    sources = []

    for document, distance in results:
        metadata = document.metadata

        source = {
            "policy_id": metadata.get("policy_id", "Unknown"),
            "policy_name": metadata.get("policy_name", "Unknown"),
            "version": metadata.get("version", "Unknown"),
            "section": metadata.get("section", "Unknown"),
            "source_file": metadata.get("source_file", "Unknown"),
            "distance": round(float(distance), 4)
        }

        evidence.append({
            **source,
            "content": document.page_content
        })

        if source not in sources:
            sources.append(source)

    return json.dumps({
        "status": "SUCCESS",
        "sources": sources,
        "evidence": evidence
    }, ensure_ascii=False)
