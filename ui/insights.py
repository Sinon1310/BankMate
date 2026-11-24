"""
Dataset-wide figures for the Overview, Transactions, Analytics and Disputes
pages.

Every financial figure comes from app.ledger (period_summary, balance_on,
customer_monthly, select_transactions); this module only selects rows,
groups them and memoises results per loaded dataset, so nothing is
recomputed on every rerun and no formula is duplicated in the UI.
Per-customer views (Customer 360) use the assistant's own tools instead,
so the UI and the assistant always agree.
"""

from datetime import date
from decimal import Decimal

from app import data_loader, ledger

ZERO = Decimal("0")
OPEN_STATUSES = ("Open", "Under Investigation")
DISPUTE_STATUSES = ("Open", "Under Investigation", "Resolved", "Rejected")

_memo = {}


def dataset():
    return data_loader.get_dataset()


def available():
    """The redesigned pages need the v1 dataset (accounts, cards, disputes)."""

    return dataset().format == data_loader.FORMAT_V1


def _cached(key, compute):
    ds = dataset()
    if _memo.get("dataset") is not ds:
        _memo.clear()
        _memo["dataset"] = ds
    if key not in _memo:
        _memo[key] = compute()
    return _memo[key]


def _index():
    def build():
        tables = dataset().tables
        index = {
            "customers": {c["customer_id"]: c for c in tables["customers"]},
            "accounts": {a["account_id"]: a for a in tables["accounts"]},
            "by_id": {t["transaction_id"]: t for t in tables["transactions"]},
            "card_by_id": {c["card_id"]: c for c in tables["cards"]},
            "accounts_by_customer": {}, "tx_by_account": {}, "tx_by_customer": {}, "cards_by_account": {},
        }
        for a in tables["accounts"]:
            index["accounts_by_customer"].setdefault(a["customer_id"], []).append(a)
        for t in tables["transactions"]:
            index["tx_by_account"].setdefault(t["account_id"], []).append(t)
            index["tx_by_customer"].setdefault(t["customer_id"], []).append(t)
        for c in tables["cards"]:
            index["cards_by_account"].setdefault(c["account_id"], []).append(c)
        return index

    return _cached("index", build)


# ---------- Window ----------

def window():
    meta = dataset().meta
    return date.fromisoformat(meta["window_start"]), date.fromisoformat(meta["as_of_date"])


def as_of():
    return window()[1]


# ---------- Balances ----------

def account_balances():
    """{account_id: balance as of the dataset's as-of date} (app.ledger.balance_on)."""

    def compute():
        index, end = _index(), as_of()
        return {aid: ledger.balance_on(a, index["tx_by_account"].get(aid, []), end, end)
                for aid, a in index["accounts"].items()}

    return _cached("balances", compute)


def customer_balance(customer_id):
    balances = account_balances()
    return sum((balances[a["account_id"]] for a in _index()["accounts_by_customer"].get(customer_id, [])), ZERO)


# ---------- Overview ----------

def overview():
    def compute():
        tables, index = dataset().tables, _index()
        disputes = tables["disputes"]
        cards = tables["cards"]
        figures = ledger.period_summary(tables["transactions"], index["by_id"])
        return {
            "customers": len(tables["customers"]),
            "accounts": len(tables["accounts"]),
            "cards": len(cards),
            "active_cards": sum(1 for c in cards if c["status"] == "Active"),
            "transactions": len(tables["transactions"]),
            "disputes": len(disputes),
            "open_disputes": sum(1 for d in disputes if d["status"] in OPEN_STATUSES),
            "total_balance": sum(account_balances().values(), ZERO),
            "figures": figures,
        }

    return _cached("overview", compute)


def segment_summary():
    """[(segment, customers, accounts, balance as of)] sorted by balance."""

    def compute():
        index = _index()
        groups = {}
        for cid, customer in index["customers"].items():
            seg = groups.setdefault(customer["segment"], {"customers": 0, "accounts": 0, "balance": ZERO})
            seg["customers"] += 1
            seg["accounts"] += len(index["accounts_by_customer"].get(cid, []))
            seg["balance"] += customer_balance(cid)
        return sorted(((s, g["customers"], g["accounts"], g["balance"]) for s, g in groups.items()),
                      key=lambda row: -row[3])

    return _cached("segments", compute)


# ---------- Selections ----------

def _scope(customer_ids=None):
    index = _index()
    if not customer_ids:
        return dataset().tables["accounts"], dataset().tables["transactions"]
    accounts = [a for cid in customer_ids for a in index["accounts_by_customer"].get(cid, [])]
    rows = [t for cid in customer_ids for t in index["tx_by_customer"].get(cid, [])]
    return accounts, rows


def period(start=None, end=None):
    first, last = window()
    return ledger.clamp_period(start, end, first, last)


def figures(start=None, end=None, customer_ids=None):
    """app.ledger.period_summary for the selection (dates inclusive)."""

    key = ("figures", start, end, tuple(customer_ids or ()))

    def compute():
        span = period(start, end)
        _, rows = _scope(customer_ids)
        selected = [] if span.empty else ledger.select_transactions(rows, span.start, span.end)
        return ledger.period_summary(selected, _index()["by_id"])

    return _cached(key, compute)


def balances_for(start=None, end=None, customer_ids=None):
    """(opening, closing) for the selection's accounts over the period, or (None, None)."""

    span = period(start, end)
    if span.empty:
        return None, None
    accounts, rows = _scope(customer_ids)
    first, last = window()
    return (ledger.customer_opening_balance(accounts, rows, span.start, first, last),
            ledger.customer_balance_on(accounts, rows, span.end, first, last))


def monthly(start=None, end=None, customer_ids=None):
    """app.ledger.customer_monthly over the selection: one row per month."""

    key = ("monthly", start, end, tuple(customer_ids or ()))

    def compute():
        span = period(start, end)
        if span.empty:
            return []
        accounts, rows = _scope(customer_ids)
        first, last = window()
        return ledger.customer_monthly(accounts, rows, _index()["by_id"], first, last, span.start, span.end)

    return _cached(key, compute)


def categories(summary):
    """[(category, gross, refunds, net)] for spending categories (Transfer Out excluded), largest net first."""

    items = [(c, summary["gross_spend"].get(c, ZERO), summary["refunds_by_category"].get(c, ZERO), net)
             for c, net in summary["net_spend"].items() if c != ledger.TRANSFER_OUT]
    return sorted(items, key=lambda row: (-row[3], row[0]))


def account_summaries(start=None, end=None, customer_ids=None):
    """Per-account flows for the period plus opening and closing balances."""

    span = period(start, end)
    if span.empty:
        return []
    index = _index()
    first, last = window()
    accounts, _ = _scope(customer_ids)
    result = []
    for account in accounts:
        if account["opened_on"] > span.end.isoformat():
            continue
        rows = index["tx_by_account"].get(account["account_id"], [])
        f = ledger.period_summary(ledger.select_transactions(rows, span.start, span.end), index["by_id"])
        opening_day = max(span.start.isoformat(), account["opening_balance_date"])
        result.append({
            "account_id": account["account_id"],
            "customer_id": account["customer_id"],
            "customer_name": index["customers"][account["customer_id"]]["customer_name"],
            "account_type": account["account_type"],
            "credits": f["total_credits"], "debits": f["total_debits"], "net": f["net_cash_flow"],
            "count": f["count"],
            "opening": ledger.customer_opening_balance([account], rows, opening_day, first, last),
            "closing": ledger.customer_balance_on([account], rows, span.end, first, last),
        })
    return result


# ---------- Transactions ----------

def customer_options():
    """[(customer_id, name, segment)] in ID order."""

    return [(c["customer_id"], c["customer_name"], c["segment"]) for c in dataset().tables["customers"]]


def account_options(customer_id=None):
    accounts = _index()["accounts_by_customer"].get(customer_id, []) if customer_id else dataset().tables["accounts"]
    return [(a["account_id"], a["account_type"], a["customer_id"]) for a in accounts]


def card_last4(card_id):
    card = _index()["card_by_id"].get(card_id) if card_id else None
    return card["card_last4"] if card else ""


def transactions_page(customer_ids=None, account_id=None, start=None, end=None, transaction_type=None,
                      category=None, merchant=None, min_amount=None, limit=25, offset=0):
    """
    One page of matching transactions (newest first) plus the total count and
    app.ledger totals over every match. Filtering uses ledger.select_transactions,
    the same rule as the assistant's transaction tool.
    """

    span = period(start, end)
    _, rows = _scope(customer_ids)
    matches = [] if span.empty else ledger.select_transactions(
        rows, span.start, span.end, transaction_type=transaction_type, category=category,
        account_id=account_id, min_amount=min_amount, merchant=merchant)
    matches.sort(key=lambda t: (t["transaction_date"], t["transaction_id"]), reverse=True)
    totals = ledger.period_summary(matches, _index()["by_id"])
    return {"rows": matches[offset:offset + limit], "total": len(matches), "totals": totals, "period": span}


def customer_name(customer_id):
    customer = _index()["customers"].get(customer_id)
    return customer["customer_name"] if customer else customer_id


# ---------- Disputes ----------

def disputes(status=None):
    """Disputes joined with their transaction, customer and (only if it exists) refund."""

    index = _index()
    rows = []
    for d in dataset().tables["disputes"]:
        if status and d["status"] != status:
            continue
        txn = index["by_id"][d["transaction_id"]]
        refund = index["by_id"].get(d["refund_transaction_id"]) if d["refund_transaction_id"] else None
        rows.append({
            **d,
            "customer_id": txn["customer_id"],
            "customer_name": index["customers"][txn["customer_id"]]["customer_name"],
            "account_id": txn["account_id"],
            "transaction_date": txn["transaction_date"],
            "merchant": txn["merchant"],
            "amount": Decimal(txn["amount"]),
            "card_last4": card_last4(txn["card_id"]),
            "refund": ({"transaction_id": refund["transaction_id"], "date": refund["transaction_date"],
                        "amount": Decimal(refund["amount"])} if refund else None),
        })
    return sorted(rows, key=lambda r: (r["raised_on"], r["dispute_id"]), reverse=True)


def dispute_counts():
    counts = {s: 0 for s in DISPUTE_STATUSES}
    for d in dataset().tables["disputes"]:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    return counts
