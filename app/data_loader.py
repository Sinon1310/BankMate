"""
Banking data loader (Stage 1B).

The data folder is BANKMATE_DATA_DIR if set (a relative path is resolved
against the project root, never the working directory), else data/. It is
loaded once, at import, and validated before anything is exposed:

- v1 dataset (has dataset.json): customers, accounts, cards, transactions,
  disputes. Checked by app.data_validation (every invariant) and against
  the dataset.json SHA-256 manifest. Generated datasets must also carry
  month-end balances; hand-written fixtures (no "generator") may omit them.
- legacy format (no dataset.json): the original customers.csv +
  transactions.csv sample. Supported until the expanded dataset replaces
  data/ (Stage 1D); its headers, IDs, values and references are checked.

An invalid configured folder raises DataLoadError. There is no fallback to
another dataset.

Backward-compatible API: `customers`, `transactions`, `find_customer` and
`find_transactions` keep the original record shape (the legacy columns),
so tools and UI behave exactly as before whatever the dataset format. The
full tables are available through `get_dataset()`. Financial calculations
live in app.ledger, not here.
"""

import csv
import json
import logging
import os
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.data_validation import (
    TABLES,
    file_hashes,
    read_dataset,
    validate_dataset,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_ENV = "BANKMATE_DATA_DIR"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

FORMAT_V1 = "v1"
FORMAT_LEGACY = "legacy"

# The record shape every existing consumer (tools, UI, tests) relies on.
LEGACY_CUSTOMER_COLUMNS = ["customer_id", "customer_name", "city", "account_type"]
LEGACY_TRANSACTION_COLUMNS = ["transaction_id", "customer_id", "transaction_date",
                              "transaction_type", "amount", "merchant"]
LEGACY_FILES = {"customers.csv": LEGACY_CUSTOMER_COLUMNS, "transactions.csv": LEGACY_TRANSACTION_COLUMNS}

_CUSTOMER_ID = re.compile(r"^C\d{3}$")
_TRANSACTION_ID = re.compile(r"^T\d{3,6}$")
_AMOUNT = re.compile(r"^\d+(\.\d{1,2})?$")


class DataLoadError(RuntimeError):
    """The configured data folder is missing, incomplete or invalid."""

    def __init__(self, folder, errors):
        self.folder = Path(folder)
        self.errors = list(errors)
        preview = "\n  ".join(self.errors[:10])
        more = f"\n  ... and {len(self.errors) - 10} more" if len(self.errors) > 10 else ""
        super().__init__(f"Cannot load banking data from {self.folder}:\n  {preview}{more}")


def resolve_data_dir(value=None):
    """The configured data folder as an absolute path (deterministic, cwd-independent)."""

    raw = value if value is not None else os.getenv(DATA_DIR_ENV, "")
    if not str(raw).strip():
        return DEFAULT_DATA_DIR

    path = Path(str(raw).strip()).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class Dataset:
    """A validated dataset: full tables plus legacy-shaped views."""

    def __init__(self, folder, data_format, meta, tables):
        self.folder = Path(folder)
        self.format = data_format
        self.meta = meta
        self.tables = tables

        if data_format == FORMAT_V1:
            self.customers = _legacy_customers(tables["customers"], tables["accounts"])
            self.transactions = [{c: t[c] for c in LEGACY_TRANSACTION_COLUMNS} for t in tables["transactions"]]
        else:
            self.customers = tables["customers"]
            self.transactions = tables["transactions"]

        self._customers_by_id = {c["customer_id"]: c for c in self.customers}
        self._transactions_by_customer = {}
        for txn in self.transactions:
            self._transactions_by_customer.setdefault(txn["customer_id"], []).append(txn)

    def find_customer(self, customer_id):
        return self._customers_by_id.get(customer_id)

    def find_transactions(self, customer_id):
        return list(self._transactions_by_customer.get(customer_id, []))


def _legacy_customers(customers, accounts):
    """
    Customers in the original shape. `account_type` is derived from the
    customer's accounts (e.g. "Savings", or "Savings + Current" for several).
    """

    types = {}
    for account in accounts:
        owned = types.setdefault(account["customer_id"], [])
        if account["account_type"] not in owned:
            owned.append(account["account_type"])

    return [
        {"customer_id": c["customer_id"], "customer_name": c["customer_name"], "city": c["city"],
         "account_type": " + ".join(types.get(c["customer_id"], []))}
        for c in customers
    ]


# ---------- Loading ----------

def load_dataset(folder=None):
    """Load and validate the dataset in `folder` (default: the configured folder)."""

    folder = resolve_data_dir(None if folder is None else str(folder))
    if not folder.is_dir():
        raise DataLoadError(folder, [f"data folder does not exist (set {DATA_DIR_ENV} or create it)"])

    if (folder / "dataset.json").exists():
        dataset = _load_v1(folder)
    else:
        dataset = _load_legacy(folder)

    logger.info("Loaded %s banking dataset from %s (%d customers, %d transactions)",
                dataset.format, folder, len(dataset.customers), len(dataset.transactions))
    return dataset


def _load_v1(folder):
    missing = [f"{t}.csv" for t in TABLES if not (folder / f"{t}.csv").is_file()]
    if missing:
        raise DataLoadError(folder, [f"missing file {name}" for name in missing])

    try:
        data = read_dataset(folder)
    except json.JSONDecodeError as error:
        raise DataLoadError(folder, [f"dataset.json is not valid JSON ({error.msg}, line {error.lineno})"]) from error
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise DataLoadError(folder, [f"cannot read dataset files: {error}"]) from error

    meta = data["meta"]
    if not isinstance(meta, dict):
        raise DataLoadError(folder, ["dataset.json must contain a JSON object"])

    # The manifest guards against edited files (including edits that keep
    # every invariant true).
    expected = meta.get("sha256")
    if not isinstance(expected, dict):
        raise DataLoadError(folder, ["dataset.json has no sha256 manifest"])
    actual = file_hashes(folder)
    changed = sorted(name for name in actual if expected.get(name) != actual[name])
    if changed:
        raise DataLoadError(folder, [f"{name} does not match the dataset.json sha256 manifest" for name in changed])

    # Generated datasets carry the generator's month-end balances; a
    # hand-written fixture (generator null) may omit them.
    errors = validate_dataset(data, require_monthly_closings=bool(meta.get("generator")))
    if errors:
        raise DataLoadError(folder, errors)

    tables = {t: data[t] for t in TABLES}
    return Dataset(folder, FORMAT_V1, meta, tables)


def _load_legacy(folder):
    extra = [f"{t}.csv" for t in TABLES if f"{t}.csv" not in LEGACY_FILES and (folder / f"{t}.csv").exists()]
    if extra:
        raise DataLoadError(folder, [f"{', '.join(extra)} found but dataset.json is missing: "
                                     "a v1 dataset needs its dataset.json"])

    missing = [name for name in LEGACY_FILES if not (folder / name).is_file()]
    if missing:
        raise DataLoadError(folder, [f"missing file {name}" for name in missing])

    tables, errors = {}, []
    for name, columns in LEGACY_FILES.items():
        try:
            with open(folder / name, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = [dict(row) for row in reader]
                header = list(reader.fieldnames or [])
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            raise DataLoadError(folder, [f"cannot read {name}: {error}"]) from error
        if header != columns:
            errors.append(f"I1 {name}: header must be {columns}, got {header}")
        tables[name.split(".")[0]] = rows

    if errors:
        raise DataLoadError(folder, errors)

    errors = validate_legacy(tables["customers"], tables["transactions"])
    if errors:
        raise DataLoadError(folder, errors)

    return Dataset(folder, FORMAT_LEGACY, {}, tables)


def validate_legacy(customers, transactions):
    """Checks for the original two-file format (IDs, values, references)."""

    errors = []
    ids = set()
    for row in customers:
        cid = row.get("customer_id") or ""
        if not _CUSTOMER_ID.match(cid):
            errors.append(f"I1 customers {cid!r}: malformed customer_id")
        if cid in ids:
            errors.append(f"I1 customers {cid}: duplicate customer_id")
        ids.add(cid)
        if not (row.get("customer_name") or "").strip():
            errors.append(f"I1 customers {cid}: customer_name is required")

    seen = set()
    for row in transactions:
        tid = row.get("transaction_id") or ""
        if not _TRANSACTION_ID.match(tid):
            errors.append(f"I1 transactions {tid!r}: malformed transaction_id")
        if tid in seen:
            errors.append(f"I1 transactions {tid}: duplicate transaction_id")
        seen.add(tid)
        if row.get("customer_id") not in ids:
            errors.append(f"I2 transactions {tid}: customer_id={row.get('customer_id')!r} does not exist in customers")
        if row.get("transaction_type") not in ("Debit", "Credit"):
            errors.append(f"I1 transactions {tid}: transaction_type must be Debit or Credit")
        amount = row.get("amount") or ""
        try:
            valid_amount = bool(_AMOUNT.match(amount)) and Decimal(amount) > 0
        except InvalidOperation:
            valid_amount = False
        if not valid_amount:
            errors.append(f"I1 transactions {tid}: amount={amount!r} must be a positive decimal")
        try:
            date.fromisoformat(row.get("transaction_date") or "")
        except ValueError:
            errors.append(f"I1 transactions {tid}: transaction_date is not an ISO date")
        if not (row.get("merchant") or "").strip():
            errors.append(f"I1 transactions {tid}: merchant is required")

    return errors


# ---------- Module-level dataset (backward-compatible API) ----------

DATA_DIR = resolve_data_dir()
_dataset = load_dataset(DATA_DIR)

customers = _dataset.customers
transactions = _dataset.transactions


def get_dataset():
    """The dataset loaded at import (full tables in `.tables`, metadata in `.meta`)."""

    return _dataset


def find_customer(customer_id):
    return _dataset.find_customer(customer_id)


def find_transactions(customer_id):
    return _dataset.find_transactions(customer_id)
