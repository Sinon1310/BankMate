"""
Read-only data views and formatting (format_inr, format_day) shared by the pages.

Everything here reuses the backend's data access:
- records come from app.data_loader (customers, transactions,
  find_customer, find_transactions);
- per-customer metrics come from the existing get_customer_metrics tool,
  so the UI shows exactly the figures the assistant reports.

Aggregates over filtered transactions use the same rule as that tool
(Decimal amounts; type compared case-insensitively to "debit"/"credit"),
and tests check the two agree.

"Net transaction flow" = credits - debits for the transactions included.
It is not an account balance: the data has no opening balances.
"""

import json
from datetime import date
from decimal import Decimal, InvalidOperation

from app import data_loader
from app.data_loader import customers, find_customer, find_transactions, transactions  # noqa: F401
from app.tools import get_customer_metrics


DEBIT = "Debit"
CREDIT = "Credit"
MINUS = "\u2212"          # "−", never the hyphen-minus, for negative amounts


# ---------- Formatting ----------

def format_inr(value, signed=False):
    """
    Format an amount in rupees with Indian digit grouping
    (1,00,000), dropping paise when they are zero. Negative amounts use
    the Unicode minus sign (U+2212), the same as the transaction tables
    and chart axes.
    """

    amount = Decimal(str(value))
    sign = MINUS if amount < 0 else ("+" if signed and amount > 0 else "")
    amount = abs(amount)

    whole = int(amount)
    paise = amount - whole

    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])

    text = f"₹{digits}"

    if paise:
        text += f"{paise:.2f}"[1:]

    return sign + text


def format_day(value):
    """'2026-11-03' or date -> '3 Nov 2026' (portable, no %-d)."""

    day = value if isinstance(value, date) else date.fromisoformat(value)
    return f"{day.day} {day.strftime('%b %Y')}"


def initials(name):
    parts = [p for p in (name or "").split() if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


# ---------- Customers ----------

def all_customers():
    # Read from the loaded dataset each time (not a list bound at import),
    # so the list always matches what the tools read.
    return list(data_loader.get_dataset().customers)


def search_customers(query):
    """Case-insensitive match on ID, name, city or account type."""

    needle = (query or "").strip().lower()
    everyone = all_customers()

    if not needle:
        return everyone

    return [
        customer for customer in everyone
        if any(
            needle in (customer.get(field) or "").lower()
            for field in ("customer_id", "customer_name", "city", "account_type")
        )
    ]


def get_customer(customer_id):
    return find_customer(customer_id)


def customer_metrics(customer_id):
    """
    Metrics from the backend tool, as Decimals. Returns None when the
    customer has no transactions (the tool returns a plain message).
    """

    raw = get_customer_metrics.invoke({"customer_id": customer_id})

    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None

    try:
        return {
            "total_debit": Decimal(data["total_debit"]),
            "total_credit": Decimal(data["total_credit"]),
            "net_flow": Decimal(data["net_transaction_flow"]),
            "count": int(data["transaction_count"]),
        }
    except (KeyError, InvalidOperation, TypeError, ValueError):
        return None


def customer_transactions(customer_id):
    """A customer's transactions, newest first."""

    return sort_newest_first(find_transactions(customer_id))


# ---------- Transactions ----------

def _amount(row):
    return Decimal(row["amount"])


def _type(row):
    return (row.get("transaction_type") or "").strip().lower()


def _day(row):
    return date.fromisoformat(row["transaction_date"])


def sort_newest_first(rows):
    return sorted(rows, key=lambda row: (row["transaction_date"], row["transaction_id"]), reverse=True)


def date_bounds(rows=None):
    rows = transactions if rows is None else rows
    days = [_day(row) for row in rows]
    return (min(days), max(days)) if days else (None, None)


def filter_transactions(customer_ids=None, types=None, start=None, end=None, rows=None):
    """
    customer_ids / types: iterables to include (None = all).
    start / end: inclusive date bounds (None = open).
    """

    rows = transactions if rows is None else rows
    ids = set(customer_ids) if customer_ids is not None else None
    wanted_types = {t.lower() for t in types} if types is not None else None

    return [
        row for row in rows
        if (ids is None or row["customer_id"] in ids)
        and (wanted_types is None or _type(row) in wanted_types)
        and (start is None or _day(row) >= start)
        and (end is None or _day(row) <= end)
    ]


def flow_summary(rows):
    """Debit/credit totals with the same rule as get_customer_metrics."""

    total_debit = sum((_amount(r) for r in rows if _type(r) == "debit"), Decimal("0"))
    total_credit = sum((_amount(r) for r in rows if _type(r) == "credit"), Decimal("0"))

    return {
        "total_debit": total_debit,
        "total_credit": total_credit,
        "net_flow": total_credit - total_debit,
        "count": len(rows),
        "debit_count": sum(1 for r in rows if _type(r) == "debit"),
        "credit_count": sum(1 for r in rows if _type(r) == "credit"),
    }


def debit_by_merchant(rows):
    """[(merchant, Decimal total)] for debits, largest first."""

    totals = {}

    for row in rows:
        if _type(row) == "debit":
            merchant = row.get("merchant") or "Unknown"
            totals[merchant] = totals.get(merchant, Decimal("0")) + _amount(row)

    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def daily_flow(rows):
    """[(date, credit total, debit total)] in date order."""

    days = {}

    for row in rows:
        credit, debit = days.get(_day(row), (Decimal("0"), Decimal("0")))

        if _type(row) == "credit":
            credit += _amount(row)
        elif _type(row) == "debit":
            debit += _amount(row)

        days[_day(row)] = (credit, debit)

    return [(day, credit, debit) for day, (credit, debit) in sorted(days.items())]


def net_flow_by_customer(rows):
    """[(customer_id, customer_name, Decimal net)] for customers present, largest first."""

    names = {c["customer_id"]: c["customer_name"] for c in customers}
    totals = {}

    for row in rows:
        net = totals.get(row["customer_id"], Decimal("0"))

        if _type(row) == "credit":
            net += _amount(row)
        elif _type(row) == "debit":
            net -= _amount(row)

        totals[row["customer_id"]] = net

    return sorted(
        ((cid, names.get(cid, cid), net) for cid, net in totals.items()),
        key=lambda item: (-item[2], item[0]),
    )
