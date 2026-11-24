"""
Expected figures, computed at evaluation time from the active dataset.

Nothing here is hardcoded: each case names a figure ("period total_debits
for C102 in October"), and the value comes from app.ledger over the
loaded dataset, independently of the tool that the assistant calls. If
the dataset is regenerated, expectations follow it. This module never
re-implements a financial rule; it only selects rows and calls ledger.
"""

from decimal import Decimal

from app import data_loader, ledger

PERIOD_FIELDS = ("total_debits", "total_credits", "net_cash_flow", "income", "other_inflows",
                 "transfer_out_total", "refund_total", "spending_gross", "spending_net")
FIGURE_KINDS = ("period", "balance", "account_balance", "selection_total", "dispute_amount", "refund_amount")


class FigureError(ValueError):
    """A case asked for a figure that the dataset cannot provide."""


def _tables():
    dataset = data_loader.get_dataset()
    if dataset.format == data_loader.FORMAT_LEGACY:
        raise FigureError("expected figures need the v1 dataset (accounts, cards, disputes)")
    return dataset


def _customer_rows(tables, customer_id):
    return [t for t in tables["transactions"] if t["customer_id"] == customer_id]


def figure(spec):
    """Decimal value for one figure spec (see FIGURE_KINDS)."""

    dataset = _tables()
    tables, meta = dataset.tables, dataset.meta
    window_start, as_of = meta["window_start"], meta["as_of_date"]
    by_id = {t["transaction_id"]: t for t in tables["transactions"]}
    kind = spec.get("figure")

    if kind == "period":
        if spec.get("field") not in PERIOD_FIELDS:
            raise FigureError(f"unknown period field {spec.get('field')!r}")
        rows = ledger.select_transactions(_customer_rows(tables, spec["customer_id"]),
                                          spec.get("start"), spec.get("end"))
        return ledger.period_summary(rows, by_id)[spec["field"]]

    if kind == "balance":
        accounts = [a for a in tables["accounts"] if a["customer_id"] == spec["customer_id"]]
        return ledger.customer_balance_on(accounts, _customer_rows(tables, spec["customer_id"]),
                                          spec.get("date", as_of), window_start, as_of)

    if kind == "account_balance":
        account = next((a for a in tables["accounts"] if a["account_id"] == spec["account_id"]), None)
        if account is None:
            raise FigureError(f"unknown account {spec['account_id']}")
        rows = [t for t in tables["transactions"] if t["account_id"] == account["account_id"]]
        return ledger.customer_balance_on([account], rows, spec.get("date", as_of), window_start, as_of)

    if kind == "selection_total":
        filters = dict(spec.get("filters", {}))
        rows = ledger.select_transactions(_customer_rows(tables, spec["customer_id"]), **filters)
        return sum((ledger.money(t["amount"]) for t in rows if t["transaction_type"] == spec["type"]),
                   Decimal("0"))

    if kind in ("dispute_amount", "refund_amount"):
        dispute = next((d for d in tables["disputes"] if d["dispute_id"] == spec["dispute_id"]), None)
        if dispute is None:
            raise FigureError(f"unknown dispute {spec['dispute_id']}")
        if kind == "dispute_amount":
            return ledger.money(by_id[dispute["transaction_id"]]["amount"])
        if not dispute["refund_transaction_id"]:
            raise FigureError(f"dispute {spec['dispute_id']} has no refund transaction")
        return ledger.money(by_id[dispute["refund_transaction_id"]]["amount"])

    raise FigureError(f"unknown figure kind {kind!r}")


def describe(spec):
    """Short label for reports, e.g. 'period total_debits C102 2026-10-01..2026-10-31'."""

    parts = [spec.get("figure", "?")]
    for key in ("field", "type", "customer_id", "account_id", "dispute_id"):
        if spec.get(key):
            parts.append(str(spec[key]))
    if spec.get("start") or spec.get("end"):
        parts.append(f"{spec.get('start', '…')}..{spec.get('end', '…')}")
    if spec.get("filters"):
        parts.append(",".join(f"{k}={v}" for k, v in sorted(spec["filters"].items())))
    return " ".join(parts)
