
import json
from decimal import Decimal

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
