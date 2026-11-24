"""
Pure banking calculations over dataset rows (no I/O, Decimal only).

Rows are the CSV dicts (all values are strings). This module is the only
place financial figures are defined; tools and UI select rows and format
results but never re-implement these rules.

Stored vs derived
- Only `opening_balance` and transaction amounts are stored.

Dates
- Every period is inclusive at both ends: a transaction dated D is in
  start..end when start <= D <= end (`in_period`). Transaction listings
  and metrics use the same rule, and totals always cover every matching
  transaction, independent of how many are displayed.

Balances
- Account balance on day D (end of day) = opening_balance + credits -
  debits dated opening_balance_date..D; None before opening_balance_date
  or after the as-of date.
- Customer balance on D = sum over accounts opened on or before D; None
  outside window_start..as_of. "Available balance" is not modelled.

Credits and debits
- total_credits: every Credit, refunds included (once).
- income: Salary, Pension, Business Income, Interest.
- other_inflows: Transfer In (including initial deposits); never income.
- total_debits: every Debit; refunds never reduce it.
- transfer_out_total: Transfer Out debits (SIPs, EMIs, insurance, family,
  suppliers). Part of debits, not of spending.
- spending_gross: debits excluding Transfer Out (cash withdrawals count).
- spending_net: spending_gross - refunds.
- net_cash_flow: total_credits - total_debits. It is not a balance.

Refunds
- A refund is a separate Credit (category "Refund") linked to its
  original Debit; the original is never altered. On its posting date it
  raises the account balance and counts as a credit in net cash flow.
  In spending it reduces the original debit's category, in the period of
  the refund's own date (cash basis). It is never income.

Identities (tested): total_credits = income + other_inflows + refund_total;
total_debits = transfer_out_total + spending_gross; spending_net =
spending_gross - refund_total; net_cash_flow = income + other_inflows -
transfer_out_total - spending_net; closing = opening + net_cash_flow.

Own-account transfers
- Transfer In/Out are external by dataset rule (invariant I31). A Transfer
  Out and a Transfer In between two accounts of the same customer on the
  same date for the same amount would be an internal move, which this
  dataset does not allow (`own_account_transfer_pairs`).

Cards
- Card validity is decided by dates, not by the card's current status.
"""

import calendar
from collections import OrderedDict
from datetime import date, timedelta
from decimal import Decimal


ZERO = Decimal("0")

DEBIT = "Debit"
CREDIT = "Credit"
REFUND = "Refund"

TRANSFER_IN = "Transfer In"
TRANSFER_OUT = "Transfer Out"

INCOME_CATEGORIES = ("Salary", "Pension", "Business Income", "Interest")
OTHER_INFLOW_CATEGORIES = (TRANSFER_IN,)
CREDIT_CATEGORIES = INCOME_CATEGORIES + OTHER_INFLOW_CATEGORIES + (REFUND,)
DEBIT_CATEGORIES = (
    "Groceries", "Dining", "Shopping", "Travel", "Transport", "Utilities",
    "Rent", "Health", "Education", "Entertainment", "Cash Withdrawal",
    "Transfer Out",
)
# Debits that can sensibly be refunded by a merchant.
REFUNDABLE_CATEGORIES = tuple(
    c for c in DEBIT_CATEGORIES if c not in ("Cash Withdrawal", "Transfer Out")
)


def to_date(value):
    return value if isinstance(value, date) else date.fromisoformat(value)


def money(value):
    return value if isinstance(value, Decimal) else Decimal(value)


def month_of(value):
    return to_date(value).isoformat()[:7]


# ---------- Cards ----------

def expiry_end(expires_on):
    """'2026-08' -> 2026-08-31 (a card is valid through its expiry month)."""

    year, month = (int(part) for part in expires_on.split("-"))
    return date(year, month, calendar.monthrange(year, month)[1])


def card_valid_until(card):
    end = expiry_end(card["expires_on"])

    if card.get("blocked_on"):
        # A block takes effect from the start of blocked_on.
        end = min(end, to_date(card["blocked_on"]) - timedelta(days=1))

    return end


def card_valid_on(card, day):
    """True if the card was in service on `day` (not whether a use was legitimate)."""

    return to_date(card["issued_on"]) <= to_date(day) <= card_valid_until(card)


def card_status_on(card, day):
    day = to_date(day)

    if day < to_date(card["issued_on"]):
        return "Not issued"
    if card.get("blocked_on") and day >= to_date(card["blocked_on"]):
        return "Blocked"
    if day > expiry_end(card["expires_on"]):
        return "Expired"
    return "Active"


# ---------- Balances ----------

def _sorted(transactions):
    return sorted(transactions, key=lambda t: (t["transaction_date"], t["transaction_id"]))


def day_end_balances(account, transactions):
    """OrderedDict date -> end-of-day balance, for days with transactions."""

    balance = money(account["opening_balance"])
    balances = OrderedDict()

    for txn in _sorted(transactions):
        amount = money(txn["amount"])
        balance += amount if txn["transaction_type"] == CREDIT else -amount
        balances[to_date(txn["transaction_date"])] = balance

    return balances


def balance_on(account, transactions, day, as_of):
    """End-of-day balance on `day`, or None outside opening_balance_date..as_of."""

    day = to_date(day)

    if day < to_date(account["opening_balance_date"]) or day > to_date(as_of):
        return None

    balance = money(account["opening_balance"])

    for txn in transactions:
        if to_date(txn["transaction_date"]) <= day:
            amount = money(txn["amount"])
            balance += amount if txn["transaction_type"] == CREDIT else -amount

    return balance


# ---------- Periods ----------

class Period:
    """An inclusive date range after clamping to the dataset window."""

    __slots__ = ("start", "end", "clamped", "empty")

    def __init__(self, start, end, clamped, empty):
        self.start, self.end, self.clamped, self.empty = start, end, clamped, empty

    def __repr__(self):
        return f"Period({self.start}..{self.end}, clamped={self.clamped}, empty={self.empty})"


def clamp_period(start, end, window_start, as_of):
    """
    Resolve a requested start..end (inclusive; None = open) against the
    dataset window. Raises ValueError if start is after end. `clamped` is
    True when the request reached outside the window; `empty` when it does
    not overlap the window at all (no transaction can match).
    """

    window_start, as_of = to_date(window_start), to_date(as_of)
    start = to_date(start) if start else None
    end = to_date(end) if end else None

    if start and end and start > end:
        raise ValueError("start_date is after end_date")

    resolved_start = max(start, window_start) if start else window_start
    resolved_end = min(end, as_of) if end else as_of
    clamped = bool((start and start < window_start) or (end and end > as_of))
    return Period(resolved_start, resolved_end, clamped, resolved_start > resolved_end)


def in_period(day, start=None, end=None):
    """The one date rule for listings and metrics: start <= day <= end (None = open)."""

    day = to_date(day)
    return (start is None or day >= to_date(start)) and (end is None or day <= to_date(end))


def select_transactions(transactions, start=None, end=None, transaction_type=None,
                        category=None, account_id=None, min_amount=None, merchant=None):
    """
    Rows matching every given filter (dates inclusive, `in_period`).
    `merchant` matches a case-insensitive part of the merchant name.
    """

    minimum = money(min_amount) if min_amount is not None else None
    needle = merchant.casefold() if merchant else None
    return [
        t for t in transactions
        if in_period(t["transaction_date"], start, end)
        and (transaction_type is None or t["transaction_type"] == transaction_type)
        and (category is None or t["category"] == category)
        and (account_id is None or t["account_id"] == account_id)
        and (minimum is None or money(t["amount"]) >= minimum)
        and (needle is None or needle in t["merchant"].casefold())
    ]


# ---------- Period figures ----------

def period_summary(transactions, originals_by_id, start=None, end=None):
    """
    Figures for transactions dated start..end (inclusive; None = open).

    `originals_by_id` maps transaction_id -> row for every transaction, so
    a refund can be attributed to its original debit's category.
    """

    start = to_date(start) if start else None
    end = to_date(end) if end else None

    figures = {
        "count": 0,
        "debit_count": 0,
        "credit_count": 0,
        "total_debits": ZERO,
        "total_credits": ZERO,
        "refund_total": ZERO,
        "income": ZERO,
        "other_inflows": ZERO,
        "transfer_out_total": ZERO,
        "gross_spend": {},
        "refunds_by_category": {},
    }

    for txn in transactions:
        if not in_period(txn["transaction_date"], start, end):
            continue

        amount = money(txn["amount"])
        category = txn["category"]
        figures["count"] += 1

        if txn["transaction_type"] == DEBIT:
            figures["debit_count"] += 1
            figures["total_debits"] += amount
            figures["gross_spend"][category] = figures["gross_spend"].get(category, ZERO) + amount
            if category == TRANSFER_OUT:
                figures["transfer_out_total"] += amount
            continue

        figures["credit_count"] += 1
        figures["total_credits"] += amount

        if category == REFUND:
            figures["refund_total"] += amount
            original = originals_by_id[txn["related_transaction_id"]]
            target = original["category"]
            figures["refunds_by_category"][target] = figures["refunds_by_category"].get(target, ZERO) + amount
        elif category in INCOME_CATEGORIES:
            figures["income"] += amount
        elif category in OTHER_INFLOW_CATEGORIES:
            figures["other_inflows"] += amount

    categories = set(figures["gross_spend"]) | set(figures["refunds_by_category"])
    figures["net_spend"] = {
        c: figures["gross_spend"].get(c, ZERO) - figures["refunds_by_category"].get(c, ZERO)
        for c in sorted(categories)
    }
    figures["net_cash_flow"] = figures["total_credits"] - figures["total_debits"]
    # Spending excludes Transfer Out (never refundable, so refunds only
    # ever reduce spending categories).
    figures["spending_gross"] = figures["total_debits"] - figures["transfer_out_total"]
    figures["spending_net"] = figures["spending_gross"] - figures["refund_total"]

    return figures


def months_between(start, end):
    """['2026-05', ...] for every calendar month touching start..end."""

    start, end = to_date(start), to_date(end)
    year, month = start.year, start.month
    months = []

    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    return months


def monthly_statements(account, transactions, originals_by_id, as_of):
    """
    One statement per month from the account's opening_balance_date to
    as_of: opening, credits, debits, closing (closing = opening + credits
    - debits; each month opens at the previous month's closing).
    """

    opening = money(account["opening_balance"])
    statements = []

    for month in months_between(account["opening_balance_date"], as_of):
        year, mon = (int(p) for p in month.split("-"))
        first = date(year, mon, 1)
        last = date(year, mon, calendar.monthrange(year, mon)[1])
        figures = period_summary(transactions, originals_by_id, first, last)
        closing = opening + figures["total_credits"] - figures["total_debits"]
        statements.append({
            "month": month,
            "opening": opening,
            "credits": figures["total_credits"],
            "debits": figures["total_debits"],
            "closing": closing,
            "figures": figures,
        })
        opening = closing

    return statements


# ---------- Customer level ----------

def _by_account(transactions):
    grouped = {}
    for txn in transactions:
        grouped.setdefault(txn["account_id"], []).append(txn)
    return grouped


def customer_balance_on(accounts, transactions, day, window_start, as_of):
    """
    End-of-day balance on `day` across the customer's accounts that were
    open on that day; None outside window_start..as_of.
    """

    day = to_date(day)
    if day < to_date(window_start) or day > to_date(as_of):
        return None

    grouped = _by_account(transactions)
    total = ZERO
    for account in accounts:
        if to_date(account["opened_on"]) <= day:
            # opened <= day and day >= window_start, so day >= opening_balance_date.
            total += balance_on(account, grouped.get(account["account_id"], []), day, as_of)
    return total


def customer_opening_balance(accounts, transactions, day, window_start, as_of):
    """
    Balance at the start of `day` (before its transactions) across the
    accounts open on that day; None outside window_start..as_of.
    """

    day = to_date(day)
    if day < to_date(window_start) or day > to_date(as_of):
        return None

    grouped = _by_account(transactions)
    total = ZERO
    for account in accounts:
        if to_date(account["opened_on"]) > day:
            continue
        if day == to_date(account["opening_balance_date"]):
            total += money(account["opening_balance"])
        else:
            total += balance_on(account, grouped.get(account["account_id"], []),
                                day - timedelta(days=1), as_of)
    return total


def customer_monthly(accounts, transactions, originals_by_id, window_start, as_of, start=None, end=None):
    """
    One row per calendar month overlapping the (clamped, inclusive) period.
    Flows count only transactions inside the period; opening is the
    balance at the start of the month's first day in the period, closing
    the end-of-day balance on its last day (closing = opening + net cash
    flow). Raises ValueError if start is after end; [] if the period does
    not overlap the dataset window.
    """

    period = clamp_period(start, end, window_start, as_of)
    if period.empty:
        return []

    rows = []
    for month in months_between(period.start, period.end):
        first = max(date(int(month[:4]), int(month[5:]), 1), period.start)
        last = min(expiry_end(month), period.end)
        figures = period_summary(transactions, originals_by_id, first, last)
        rows.append({
            "month": month,
            "start": first,
            "end": last,
            "opening_balance": customer_opening_balance(accounts, transactions, first, window_start, as_of),
            "closing_balance": customer_balance_on(accounts, transactions, last, window_start, as_of),
            "figures": figures,
        })
    return rows


# ---------- Dataset rules ----------

def own_account_transfer_pairs(transactions):
    """
    (Transfer Out id, Transfer In id) pairs that look like a move between
    two accounts of the same customer: same customer, same date, same
    amount, different accounts. The dataset rule (I31) is that there are
    none, so Transfer In/Out can be treated as external.
    """

    incoming = {}
    for txn in transactions:
        if txn["category"] == TRANSFER_IN:
            key = (txn["customer_id"], txn["transaction_date"], money(txn["amount"]))
            incoming.setdefault(key, []).append(txn)

    pairs = []
    for txn in transactions:
        if txn["category"] != TRANSFER_OUT:
            continue
        key = (txn["customer_id"], txn["transaction_date"], money(txn["amount"]))
        for other in incoming.get(key, []):
            if other["account_id"] != txn["account_id"]:
                pairs.append((txn["transaction_id"], other["transaction_id"]))
    return sorted(pairs)
