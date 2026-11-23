"""
Referential and financial integrity checks for a BankMate dataset folder
(customers, accounts, cards, transactions, disputes + dataset.json).

`validate_dataset(dataset)` returns a list of error strings, each prefixed
with its invariant code (I1-I29, Stage 1 design); `assert_valid` raises
DataValidationError. Nothing here repairs data: a dataset either passes
every check or is rejected.

I31 (no transfers between a customer's own accounts): the schema has no
counterparty-account field and related_transaction_id is reserved for
refunds (I15), so an own-account transfer cannot be recorded, and there
is no pairing to validate. Transfers are therefore external by rule, and
I31 rejects the two observable signs of an unrecorded internal move:
  - a Transfer In/Out whose counterparty names an account ID or the
    customer's own name (explicit self-reference);
  - a Transfer Out and a Transfer In on two different accounts of the same
    customer, on the same date, for the same amount. This is reported as
    ambiguous (it cannot be shown to be external), never classified.
Moves with different dates or amounts cannot be detected from the data;
the generator never creates internal transfers.

Invariant I8 (tool account filters must belong to the scoped customer) is
a tool-level rule, enforced where the tools are implemented, not here.
"""

import csv
import hashlib
import json
import re
from collections import OrderedDict, defaultdict
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app import ledger


TABLES = {
    "customers": ["customer_id", "customer_name", "city", "segment", "customer_since"],
    "accounts": ["account_id", "customer_id", "account_type", "opened_on",
                 "opening_balance_date", "opening_balance"],
    "cards": ["card_id", "account_id", "network", "card_last4", "issued_on", "expires_on",
              "blocked_on", "replaces_card_id", "status"],
    "transactions": ["transaction_id", "account_id", "customer_id", "card_id", "transaction_date",
                     "transaction_type", "amount", "merchant", "category", "channel",
                     "related_transaction_id"],
    "disputes": ["dispute_id", "transaction_id", "raised_on", "reason", "status", "resolved_on",
                 "resolution_outcome", "refund_transaction_id"],
}

PRIMARY_KEYS = {
    "customers": ("customer_id", re.compile(r"^C\d{3}$")),
    "accounts": ("account_id", re.compile(r"^ACC\d{4}$")),
    "cards": ("card_id", re.compile(r"^CRD\d{4}$")),
    "transactions": ("transaction_id", re.compile(r"^T\d{3,6}$")),
    "disputes": ("dispute_id", re.compile(r"^D\d{4}$")),
}

SEGMENTS = {"Retail", "Premier", "Student", "Senior", "Business"}
ACCOUNT_TYPES = {"Savings", "Current"}
NETWORKS = {"RuPay", "Visa", "Mastercard"}
CARD_STATUSES = {"Active", "Blocked", "Expired"}
TRANSACTION_TYPES = {ledger.DEBIT, ledger.CREDIT}
CHANNELS = {"Card", "UPI", "Bank Transfer", "ATM"}
CARD_CHANNELS = {"Card", "ATM"}
DISPUTE_REASONS = {"Unauthorized", "Duplicate charge", "Incorrect amount", "Goods not received"}
# Exactly the statuses defined in knowledge/policies/card_dispute_policy.txt.
DISPUTE_STATUSES = ("Open", "Under Investigation", "Resolved", "Rejected")
CLOSED_STATUSES = {"Resolved", "Rejected"}
# Dataset field (not policy): financial result of a Resolved dispute.
RESOLUTION_OUTCOMES = {"Full refund", "Partial refund", "No financial adjustment"}
REFUND_OUTCOMES = {"Full refund", "Partial refund"}

MAX_CARDS_IN_SERVICE = 2
BLOCK_REPLACEMENT_DAYS = 7
RENEWAL_LEAD_DAYS = 31

AMOUNT = re.compile(r"^\d+(\.\d{1,2})?$")
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
LAST4 = re.compile(r"^\d{4}$")


class DataValidationError(ValueError):
    """The dataset breaks one or more integrity rules."""

    def __init__(self, errors):
        self.errors = list(errors)
        preview = "\n".join(self.errors[:20])
        more = f"\n... and {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
        super().__init__(f"{len(self.errors)} dataset validation error(s):\n{preview}{more}")


# ---------- Reading ----------

def read_dataset(folder):
    """Read the five CSVs and dataset.json from `folder` (values kept as strings)."""

    folder = Path(folder)
    dataset = {"meta": json.loads((folder / "dataset.json").read_text(encoding="utf-8")), "headers": {}}

    for table in TABLES:
        with open(folder / f"{table}.csv", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            dataset[table] = [dict(row) for row in reader]
            dataset["headers"][table] = list(reader.fieldnames or [])

    return dataset


def file_hashes(folder):
    folder = Path(folder)
    return {f"{t}.csv": hashlib.sha256((folder / f"{t}.csv").read_bytes()).hexdigest() for t in TABLES}


# ---------- Validation ----------

def assert_valid(dataset, require_monthly_closings=True):
    errors = validate_dataset(dataset, require_monthly_closings)
    if errors:
        raise DataValidationError(errors)


def validate_dataset(dataset, require_monthly_closings=True):
    """
    All invariants. `require_monthly_closings=False` is only for datasets
    created before dataset.json recorded month-end balances (the legacy
    test fixture); every monthly check still runs against the ledger.
    """

    errors = []
    meta = _check_meta(dataset, errors)
    if meta is None:
        return errors

    _check_structure(dataset, meta, errors)
    if errors:
        # Later checks assume well-formed values and resolvable keys.
        return errors

    index = _index(dataset)
    _check_ownership(dataset, index, errors)
    _check_refund_links(dataset, index, errors)
    if any(e.startswith(("I15", "I16")) for e in errors):
        # Balance and spending figures need every refund linked to a debit.
        return errors

    _check_accounts_and_balances(dataset, index, meta, errors, require_monthly_closings)
    _check_identities(dataset, index, meta, errors)
    _check_transfers(dataset, index, errors)
    _check_cards(dataset, index, meta, errors)
    _check_disputes(dataset, index, meta, errors)
    return errors


def _parse_date(value):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _check_meta(dataset, errors):
    meta = dataset.get("meta") or {}
    start = _parse_date(meta.get("window_start"))
    as_of = _parse_date(meta.get("as_of_date"))

    if start is None or as_of is None or start > as_of:
        errors.append("I1 dataset.json: window_start/as_of_date missing, invalid or out of order")
        return None
    if meta.get("currency") != "INR":
        errors.append("I1 dataset.json: currency must be INR")

    return {"start": start, "as_of": as_of, "raw": meta}


def _check_structure(dataset, meta, errors):
    """I1 (format, enums, keys) and I2 (foreign keys resolve)."""

    for table, columns in TABLES.items():
        if dataset["headers"].get(table) != columns:
            errors.append(f"I1 {table}.csv: header must be {columns}, got {dataset['headers'].get(table)}")

    counts = meta["raw"].get("row_counts")
    if counts is not None:
        for table in TABLES:
            if counts.get(table) != len(dataset[table]):
                errors.append(f"I1 dataset.json: row_counts[{table}]={counts.get(table)} but file has {len(dataset[table])}")

    if errors:
        return

    ids = {}
    for table, (key, pattern) in PRIMARY_KEYS.items():
        seen = set()
        for row in dataset[table]:
            value = row[key]
            if not pattern.match(value or ""):
                errors.append(f"I1 {table} {value!r}: malformed {key}")
            if value in seen:
                errors.append(f"I1 {table} {value}: duplicate {key}")
            seen.add(value)
        ids[table] = seen

    def need_date(table, row_id, field, value, nullable=False):
        if nullable and value == "":
            return
        if _parse_date(value) is None:
            errors.append(f"I1 {table} {row_id}: {field}={value!r} is not an ISO date")

    def need_enum(table, row_id, field, value, allowed, nullable=False):
        if nullable and value == "":
            return
        if value not in allowed:
            errors.append(f"I1 {table} {row_id}: {field}={value!r} not in {sorted(allowed)}")

    def need_amount(table, row_id, field, value, allow_zero=False):
        if not AMOUNT.match(value or ""):
            errors.append(f"I1 {table} {row_id}: {field}={value!r} is not a decimal with at most 2 places")
            return
        try:
            amount = Decimal(value)
        except InvalidOperation:
            errors.append(f"I1 {table} {row_id}: {field}={value!r} is not a decimal")
            return
        if amount < 0 or (amount == 0 and not allow_zero):
            errors.append(f"I1 {table} {row_id}: {field}={value} must be {'>= 0' if allow_zero else '> 0'}")

    def need_fk(table, row_id, field, value, target, nullable=False):
        if nullable and value == "":
            return
        if value not in ids[target]:
            errors.append(f"I2 {table} {row_id}: {field}={value!r} does not exist in {target}")

    for row in dataset["customers"]:
        cid = row["customer_id"]
        if not row["customer_name"].strip() or not row["city"].strip():
            errors.append(f"I1 customers {cid}: name and city are required")
        need_enum("customers", cid, "segment", row["segment"], SEGMENTS)
        need_date("customers", cid, "customer_since", row["customer_since"])

    for row in dataset["accounts"]:
        aid = row["account_id"]
        need_fk("accounts", aid, "customer_id", row["customer_id"], "customers")
        need_enum("accounts", aid, "account_type", row["account_type"], ACCOUNT_TYPES)
        need_date("accounts", aid, "opened_on", row["opened_on"])
        need_date("accounts", aid, "opening_balance_date", row["opening_balance_date"])
        need_amount("accounts", aid, "opening_balance", row["opening_balance"], allow_zero=True)

    for row in dataset["cards"]:
        cid = row["card_id"]
        need_fk("cards", cid, "account_id", row["account_id"], "accounts")
        need_enum("cards", cid, "network", row["network"], NETWORKS)
        if not LAST4.match(row["card_last4"] or ""):
            errors.append(f"I1 cards {cid}: card_last4 must be exactly 4 digits")
        need_date("cards", cid, "issued_on", row["issued_on"])
        if not MONTH.match(row["expires_on"] or ""):
            errors.append(f"I1 cards {cid}: expires_on={row['expires_on']!r} must be YYYY-MM")
        need_date("cards", cid, "blocked_on", row["blocked_on"], nullable=True)
        need_fk("cards", cid, "replaces_card_id", row["replaces_card_id"], "cards", nullable=True)
        need_enum("cards", cid, "status", row["status"], CARD_STATUSES)

    for row in dataset["transactions"]:
        tid = row["transaction_id"]
        need_fk("transactions", tid, "account_id", row["account_id"], "accounts")
        need_fk("transactions", tid, "customer_id", row["customer_id"], "customers")
        need_fk("transactions", tid, "card_id", row["card_id"], "cards", nullable=True)
        need_date("transactions", tid, "transaction_date", row["transaction_date"])
        need_enum("transactions", tid, "transaction_type", row["transaction_type"], TRANSACTION_TYPES)
        need_amount("transactions", tid, "amount", row["amount"])
        if not row["merchant"].strip():
            errors.append(f"I1 transactions {tid}: merchant is required")
        need_enum("transactions", tid, "channel", row["channel"], CHANNELS)
        allowed = ledger.DEBIT_CATEGORIES if row["transaction_type"] == ledger.DEBIT else ledger.CREDIT_CATEGORIES
        need_enum("transactions", tid, "category", row["category"], set(allowed))
        need_fk("transactions", tid, "related_transaction_id", row["related_transaction_id"],
                "transactions", nullable=True)

    for row in dataset["disputes"]:
        did = row["dispute_id"]
        need_fk("disputes", did, "transaction_id", row["transaction_id"], "transactions")
        need_date("disputes", did, "raised_on", row["raised_on"])
        need_enum("disputes", did, "reason", row["reason"], DISPUTE_REASONS)
        need_enum("disputes", did, "status", row["status"], set(DISPUTE_STATUSES))
        need_date("disputes", did, "resolved_on", row["resolved_on"], nullable=True)
        need_enum("disputes", did, "resolution_outcome", row["resolution_outcome"], RESOLUTION_OUTCOMES,
                  nullable=True)
        need_fk("disputes", did, "refund_transaction_id", row["refund_transaction_id"], "transactions",
                nullable=True)


def _index(dataset):
    index = {
        "customers": {r["customer_id"]: r for r in dataset["customers"]},
        "accounts": {r["account_id"]: r for r in dataset["accounts"]},
        "cards": {r["card_id"]: r for r in dataset["cards"]},
        "transactions": {r["transaction_id"]: r for r in dataset["transactions"]},
        "by_account": defaultdict(list),
        "cards_by_account": defaultdict(list),
        "accounts_by_customer": defaultdict(list),
        "refunds_of": defaultdict(list),
    }
    for row in dataset["transactions"]:
        index["by_account"][row["account_id"]].append(row)
        if row["related_transaction_id"]:
            index["refunds_of"][row["related_transaction_id"]].append(row)
    for row in dataset["cards"]:
        index["cards_by_account"][row["account_id"]].append(row)
    for row in dataset["accounts"]:
        index["accounts_by_customer"][row["customer_id"]].append(row)
    return index


def _check_ownership(dataset, index, errors):
    """I3-I7: every link stays within one customer / one account."""

    accounts, cards, txns = index["accounts"], index["cards"], index["transactions"]

    for row in dataset["transactions"]:
        tid = row["transaction_id"]
        owner = accounts[row["account_id"]]["customer_id"]
        if row["customer_id"] != owner:
            errors.append(f"I3 transactions {tid}: customer_id {row['customer_id']} but account "
                          f"{row['account_id']} belongs to {owner}")
        if row["card_id"] and cards[row["card_id"]]["account_id"] != row["account_id"]:
            errors.append(f"I4 transactions {tid}: card {row['card_id']} is not on account {row['account_id']}")
        if row["related_transaction_id"]:
            original = txns[row["related_transaction_id"]]
            if original["account_id"] != row["account_id"]:
                errors.append(f"I5 transactions {tid}: refund is on {row['account_id']} but its original "
                              f"{original['transaction_id']} is on {original['account_id']}")

    for row in dataset["disputes"]:
        txn = txns[row["transaction_id"]]
        if txn["card_id"] and cards[txn["card_id"]]["account_id"] != txn["account_id"]:
            errors.append(f"I6 disputes {row['dispute_id']}: disputed card is not on the transaction's account")

    for row in dataset["cards"]:
        if row["replaces_card_id"]:
            old = cards[row["replaces_card_id"]]
            if old["account_id"] != row["account_id"]:
                errors.append(f"I7 cards {row['card_id']}: replaces {old['card_id']} on a different account")


def _check_accounts_and_balances(dataset, index, meta, errors, require_monthly_closings=True):
    """I9-I14."""

    start, as_of = meta["start"], meta["as_of"]
    customers, txns = index["customers"], index["transactions"]
    expected_closing = meta["raw"].get("closing_balances")

    if expected_closing is None:
        errors.append("I12 dataset.json: closing_balances missing")
        expected_closing = {}
    elif set(expected_closing) != set(index["accounts"]):
        errors.append("I12 dataset.json: closing_balances must list exactly the dataset's accounts")

    # Month-end balances from the generator's own ledger (independent of app.ledger).
    expected_monthly = meta["raw"].get("monthly_closing_balances")
    if expected_monthly is None:
        if require_monthly_closings:
            errors.append("I13 dataset.json: monthly_closing_balances missing")
        expected_monthly = None
    elif set(expected_monthly) != set(index["accounts"]):
        errors.append("I13 dataset.json: monthly_closing_balances must list exactly the dataset's accounts")
    month_closings = {}     # account -> {month: Decimal}, recomputed here

    for account in dataset["accounts"]:
        aid = account["account_id"]
        opened = ledger.to_date(account["opened_on"])
        obd = ledger.to_date(account["opening_balance_date"])
        opening = ledger.money(account["opening_balance"])
        since = ledger.to_date(customers[account["customer_id"]]["customer_since"])

        # I9
        if obd != max(start, opened):
            errors.append(f"I9 accounts {aid}: opening_balance_date {obd} must be max(window_start, opened_on) "
                          f"= {max(start, opened)}")
        if opened < since:
            errors.append(f"I9 accounts {aid}: opened_on {opened} is before customer_since {since}")
        if opened > as_of:
            errors.append(f"I9 accounts {aid}: opened_on {opened} is after as_of_date {as_of}")
        if opened >= start and opening != 0:
            errors.append(f"I9 accounts {aid}: opened inside the window, so opening_balance must be 0.00")

        rows = index["by_account"][aid]

        # I10
        for row in rows:
            day = ledger.to_date(row["transaction_date"])
            if day < obd or day > as_of:
                errors.append(f"I10 transactions {row['transaction_id']}: dated {day}, outside {aid}'s "
                              f"history {obd}..{as_of}")

        # I11
        balances = ledger.day_end_balances(account, rows)
        for day, balance in balances.items():
            if balance < 0:
                errors.append(f"I11 accounts {aid}: end-of-day balance {balance} on {day} is negative")

        # I12: recomputed closing vs the generator's independent ledger.
        closing = ledger.balance_on(account, rows, as_of, as_of)
        if aid in expected_closing and closing != ledger.money(expected_closing[aid]):
            errors.append(f"I12 accounts {aid}: recomputed closing {closing} != dataset.json {expected_closing[aid]}")

        # I13: month-by-month chain, checked against day-level balances.
        statements = ledger.monthly_statements(account, rows, txns, as_of)
        if statements and statements[0]["opening"] != opening:
            errors.append(f"I13 accounts {aid}: first month must open at opening_balance")
        for previous, current in zip(statements, statements[1:]):
            if current["opening"] != previous["closing"]:
                errors.append(f"I13 accounts {aid}: {current['month']} opens at {current['opening']} "
                              f"but {previous['month']} closed at {previous['closing']}")
        for statement in statements:
            # Last day of the month ("YYYY-MM" has the same form as a card expiry).
            month_end = min(ledger.expiry_end(statement["month"]), as_of)
            day_level = ledger.balance_on(account, rows, month_end, as_of)
            if statement["closing"] != statement["opening"] + statement["credits"] - statement["debits"]:
                errors.append(f"I13 accounts {aid}: {statement['month']} closing does not reconcile")
            if day_level is not None and statement["closing"] != day_level:
                errors.append(f"I13 accounts {aid}: {statement['month']} closing {statement['closing']} "
                              f"!= day-level balance {day_level}")
        total_flow = sum((s["credits"] - s["debits"] for s in statements), ledger.ZERO)
        if closing is not None and opening + total_flow != closing:
            errors.append(f"I13 accounts {aid}: monthly net flows do not sum to closing - opening")

        month_closings[aid] = OrderedDict((s["month"], s["closing"]) for s in statements)

        # I13 (independent): recomputed month-end balances vs dataset.json.
        if expected_monthly is not None and aid in expected_monthly:
            recorded = expected_monthly[aid]
            if list(recorded) != list(month_closings[aid]):
                errors.append(f"I13 accounts {aid}: dataset.json months {list(recorded)} != account history "
                              f"months {list(month_closings[aid])}")
            for month, value in month_closings[aid].items():
                if month in recorded and ledger.money(recorded[month]) != value:
                    errors.append(f"I13 accounts {aid}: {month} closing {value} != dataset.json {recorded[month]}")
            last = next(reversed(month_closings[aid]), None)
            if last in recorded:
                if closing is not None and ledger.money(recorded[last]) != closing:
                    errors.append(f"I13 accounts {aid}: final month closing {recorded[last]} != as-of balance {closing}")
                if aid in expected_closing and ledger.money(recorded[last]) != ledger.money(expected_closing[aid]):
                    errors.append(f"I12 accounts {aid}: final month closing {recorded[last]} != "
                                  f"closing_balances {expected_closing[aid]}")

    # I14: customer figures = sum of their accounts.
    months = ledger.months_between(start, as_of)
    for cid, accounts in index["accounts_by_customer"].items():
        rows = [t for a in accounts for t in index["by_account"][a["account_id"]]]
        whole = ledger.period_summary(rows, txns)
        parts = [ledger.period_summary(index["by_account"][a["account_id"]], txns) for a in accounts]
        for field in ("total_debits", "total_credits", "refund_total", "income", "other_inflows"):
            if whole[field] != sum((p[field] for p in parts), ledger.ZERO):
                errors.append(f"I14 customers {cid}: {field} differs from the sum of their accounts")

        # Customer-level monthly balance, built from the customer's own
        # transactions, must equal the sum of their accounts' month-end
        # balances (dataset.json when present). An account not yet opened
        # counts as 0 (accounts opened inside the window start at 0.00).
        balance = sum((ledger.money(a["opening_balance"]) for a in accounts), ledger.ZERO)
        for month in months:
            figures = ledger.period_summary(rows, txns, f"{month}-01", ledger.expiry_end(month))
            balance += figures["total_credits"] - figures["total_debits"]
            total = ledger.ZERO
            for a in accounts:
                source = (expected_monthly or {}).get(a["account_id"]) or month_closings.get(a["account_id"], {})
                if month in source:
                    total += ledger.money(source[month])
            if balance != total:
                errors.append(f"I14 customers {cid}: {month} customer balance {balance} != sum of account "
                              f"month-end balances {total}")


def _check_refund_links(dataset, index, errors):
    """I15-I17."""

    txns = index["transactions"]

    for row in dataset["transactions"]:
        tid = row["transaction_id"]
        is_refund = row["category"] == ledger.REFUND
        linked = bool(row["related_transaction_id"])

        # I15
        if is_refund != linked or (is_refund and row["transaction_type"] != ledger.CREDIT):
            errors.append(f"I15 transactions {tid}: category Refund <=> Credit with related_transaction_id")
            continue

        if not is_refund:
            continue

        # I16
        original = txns[row["related_transaction_id"]]
        if original["transaction_type"] != ledger.DEBIT or original["category"] == ledger.REFUND:
            errors.append(f"I16 transactions {tid}: original {original['transaction_id']} must be a non-refund Debit")
        elif original["category"] not in ledger.REFUNDABLE_CATEGORIES:
            errors.append(f"I16 transactions {tid}: original category {original['category']} is not refundable")
        if original["transaction_date"] > row["transaction_date"]:
            errors.append(f"I16 transactions {tid}: refund dated before its original {original['transaction_id']}")

    # I17
    for original_id, refunds in index["refunds_of"].items():
        total = sum((ledger.money(r["amount"]) for r in refunds), ledger.ZERO)
        if total > ledger.money(txns[original_id]["amount"]):
            errors.append(f"I17 transactions {original_id}: refunds {total} exceed the original amount")



def _check_identities(dataset, index, meta, errors):
    """I18: identities R1-R4 per account x month and per account overall."""

    txns = index["transactions"]
    as_of = meta["as_of"]
    for account in dataset["accounts"]:
        rows = index["by_account"][account["account_id"]]
        scopes = [("all", ledger.period_summary(rows, txns))]
        scopes += [(s["month"], s["figures"]) for s in ledger.monthly_statements(account, rows, txns, as_of)]
        for label, f in scopes:
            where = f"I18 accounts {account['account_id']} {label}"
            if f["total_credits"] != f["income"] + f["other_inflows"] + f["refund_total"]:
                errors.append(f"{where}: R1 credits != income + other_inflows + refunds")
            if sum(f["gross_spend"].values(), ledger.ZERO) != f["total_debits"]:
                errors.append(f"{where}: R2 sum of gross spend != total debits")
            if sum(f["refunds_by_category"].values(), ledger.ZERO) != f["refund_total"]:
                errors.append(f"{where}: R3 refunds by category != refund total")
            if sum(f["net_spend"].values(), ledger.ZERO) != f["total_debits"] - f["refund_total"]:
                errors.append(f"{where}: R4 net spend != debits - refunds")


_ACCOUNT_REFERENCE = re.compile(r"\bACC\d{4}\b", re.IGNORECASE)


def _check_transfers(dataset, index, errors):
    """I31: no transfers between a customer's own accounts (see module docstring)."""

    transfers = (ledger.TRANSFER_IN, ledger.TRANSFER_OUT)

    for row in dataset["transactions"]:
        if row["category"] not in transfers:
            continue
        name = index["customers"][row["customer_id"]]["customer_name"].strip().lower()
        merchant = row["merchant"].lower()
        if _ACCOUNT_REFERENCE.search(row["merchant"]) or (name and name in merchant):
            errors.append(f"I31 transactions {row['transaction_id']}: {row['category']} counterparty refers to "
                          f"an account or the customer's own name; transfers must be external")

    for out_id, in_id in ledger.own_account_transfer_pairs(dataset["transactions"]):
        out_row, in_row = index["transactions"][out_id], index["transactions"][in_id]
        errors.append(f"I31 transactions {out_id}/{in_id}: Transfer Out on {out_row['account_id']} and Transfer In "
                      f"on {in_row['account_id']} (customer {out_row['customer_id']}, {out_row['transaction_date']}, "
                      f"{out_row['amount']}) cannot be distinguished from an own-account transfer")


def _check_cards(dataset, index, meta, errors):
    """I19-I23."""

    cards, accounts = index["cards"], index["accounts"]
    start, as_of = meta["start"], meta["as_of"]

    for row in dataset["transactions"]:
        tid = row["transaction_id"]
        # I19
        if bool(row["card_id"]) != (row["channel"] in CARD_CHANNELS):
            errors.append(f"I19 transactions {tid}: card_id must be set exactly for Card/ATM channels")
        if (row["channel"] == "ATM") != (row["category"] == "Cash Withdrawal"):
            errors.append(f"I19 transactions {tid}: ATM channel <=> Cash Withdrawal category")
        # I20
        if row["card_id"] and not ledger.card_valid_on(cards[row["card_id"]], row["transaction_date"]):
            errors.append(f"I20 transactions {tid}: card {row['card_id']} was not in service on "
                          f"{row['transaction_date']}")

    replaced = defaultdict(list)

    for card in dataset["cards"]:
        cid = card["card_id"]
        issued = ledger.to_date(card["issued_on"])
        account = accounts[card["account_id"]]

        if issued < ledger.to_date(account["opened_on"]):
            errors.append(f"I22 cards {cid}: issued before its account was opened")
        if issued > as_of:
            errors.append(f"I22 cards {cid}: issued after as_of_date")
        if ledger.expiry_end(card["expires_on"]) <= issued:
            errors.append(f"I22 cards {cid}: expires_on must be after issued_on")
        if card["blocked_on"] and ledger.to_date(card["blocked_on"]) < issued:
            errors.append(f"I21 cards {cid}: blocked before it was issued")

        # I21
        derived = ledger.card_status_on(card, as_of)
        if bool(card["blocked_on"]) != (derived == "Blocked"):
            errors.append(f"I21 cards {cid}: blocked_on must be set exactly when the card is blocked")
        if card["status"] != derived:
            errors.append(f"I21 cards {cid}: stored status {card['status']} != derived status {derived} at {as_of}")

        # I22
        if card["replaces_card_id"]:
            old = cards[card["replaces_card_id"]]
            replaced[old["card_id"]].append(cid)
            if old["card_last4"] == card["card_last4"]:
                errors.append(f"I22 cards {cid}: replacement must have a different card_last4")
            if old["blocked_on"]:
                gap = (issued - ledger.to_date(old["blocked_on"])).days
                if not 0 <= gap <= BLOCK_REPLACEMENT_DAYS:
                    errors.append(f"I22 cards {cid}: replacement issued {gap} days after the block "
                                  f"(allowed 0-{BLOCK_REPLACEMENT_DAYS})")
            else:
                old_end = ledger.expiry_end(old["expires_on"])
                if not old_end - timedelta(days=RENEWAL_LEAD_DAYS) <= issued <= old_end:
                    errors.append(f"I22 cards {cid}: renewal must be issued within {RENEWAL_LEAD_DAYS} days "
                                  f"before {old['card_id']} expires")

    for old_id, new_ids in replaced.items():
        if len(new_ids) > 1:
            errors.append(f"I22 cards {old_id}: replaced more than once ({new_ids})")

    # I23
    for aid, account_cards in index["cards_by_account"].items():
        day = start
        while day <= as_of:
            in_service = sum(1 for c in account_cards if ledger.card_valid_on(c, day))
            if in_service > MAX_CARDS_IN_SERVICE:
                errors.append(f"I23 accounts {aid}: {in_service} cards in service on {day}")
                break
            day += timedelta(days=1)


def _check_disputes(dataset, index, meta, errors):
    """I24-I29."""

    txns, cards = index["transactions"], index["cards"]
    as_of = meta["as_of"]
    seen_transactions = set()

    for row in dataset["disputes"]:
        did = row["dispute_id"]
        txn = txns[row["transaction_id"]]
        raised = ledger.to_date(row["raised_on"])
        status, outcome = row["status"], row["resolution_outcome"]

        # I24
        if txn["transaction_type"] != ledger.DEBIT or txn["channel"] != "Card":
            errors.append(f"I24 disputes {did}: disputed transaction must be a Card Debit")
        if row["transaction_id"] in seen_transactions:
            errors.append(f"I24 disputes {did}: transaction {row['transaction_id']} is disputed more than once")
        seen_transactions.add(row["transaction_id"])
        if not ledger.to_date(txn["transaction_date"]) <= raised <= as_of:
            errors.append(f"I24 disputes {did}: raised_on must be between the transaction date and as_of_date")
        if row["reason"] == "Unauthorized" and txn["card_id"]:
            card = cards[txn["card_id"]]
            if card["blocked_on"] and ledger.to_date(card["blocked_on"]) >= ledger.to_date(txn["transaction_date"]) \
                    and card["blocked_on"] != row["raised_on"]:
                errors.append(f"I24 disputes {did}: card {card['card_id']} blocked on {card['blocked_on']}, "
                              f"expected the dispute date {row['raised_on']}")
        if row["reason"] == "Duplicate charge":
            twins = [
                t for t in index["by_account"][txn["account_id"]]
                if t["transaction_id"] < txn["transaction_id"]
                and t["transaction_type"] == ledger.DEBIT
                and (t["transaction_date"], t["merchant"], t["amount"]) ==
                    (txn["transaction_date"], txn["merchant"], txn["amount"])
            ]
            if not twins:
                errors.append(f"I24 disputes {did}: duplicate charge has no earlier identical debit")

        # I25
        if bool(row["resolved_on"]) != (status in CLOSED_STATUSES):
            errors.append(f"I25 disputes {did}: resolved_on must be set exactly for Resolved/Rejected")
        elif row["resolved_on"] and not raised <= ledger.to_date(row["resolved_on"]) <= as_of:
            errors.append(f"I25 disputes {did}: resolved_on must be between raised_on and as_of_date")

        # I26
        if bool(outcome) != (status == "Resolved"):
            errors.append(f"I26 disputes {did}: resolution_outcome must be set exactly when status is Resolved")

        # I27 / I28
        wants_refund = outcome in REFUND_OUTCOMES
        if bool(row["refund_transaction_id"]) != wants_refund:
            errors.append(f"I27 disputes {did}: refund_transaction_id must be set exactly for refund outcomes")
        elif wants_refund:
            refund = txns[row["refund_transaction_id"]]
            if refund["category"] != ledger.REFUND or refund["related_transaction_id"] != row["transaction_id"]:
                errors.append(f"I27 disputes {did}: {refund['transaction_id']} is not a refund of the disputed transaction")
            elif row["resolved_on"] and not raised <= ledger.to_date(refund["transaction_date"]) \
                    <= ledger.to_date(row["resolved_on"]):
                errors.append(f"I27 disputes {did}: refund must be dated between raised_on and resolved_on")
            else:
                others = sum((ledger.money(r["amount"]) for r in index["refunds_of"][row["transaction_id"]]
                              if r["transaction_id"] != refund["transaction_id"]), ledger.ZERO)
                remaining = ledger.money(txn["amount"]) - others
                amount = ledger.money(refund["amount"])
                if outcome == "Full refund" and amount != remaining:
                    errors.append(f"I28 disputes {did}: full refund {amount} != refundable remainder {remaining}")
                if outcome == "Partial refund" and not amount < remaining:
                    errors.append(f"I28 disputes {did}: partial refund {amount} must be below {remaining}")

        # I29: a disputed transaction is refunded only through a refund outcome.
        linked = {r["transaction_id"] for r in index["refunds_of"][row["transaction_id"]]}
        allowed = {row["refund_transaction_id"]} if wants_refund and row["refund_transaction_id"] else set()
        if linked != allowed:
            errors.append(f"I29 disputes {did}: refunds of the disputed transaction {sorted(linked)} "
                          f"do not match its outcome ({outcome or status})")
