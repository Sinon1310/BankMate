"""
Argument validation for customer-facing tools (Stage 1C).

The single source of truth for which arguments each customer tool accepts,
which are required, and what values are allowed. It validates and
normalises; it never calculates (figures live in app.ledger) and never
reads data (account ownership is supplied by the caller), so it has no
dependency on tools, the assistant or the data loader.

Authorization (unchanged from the existing model):
- `customer_id` is required. It must be a well-formed ID (else
  INVALID_CUSTOMER_ID) and one the *user* typed in the conversation
  (`allowed_customers`, from security.referenced_customer_ids), else
  ACCESS_DENIED. Nothing from documents, tool output or the model can
  widen that set.
- No other identifier can stand in for it. An `account_id` filter must
  belong to the authorised customer (else ACCESS_DENIED, identical whether
  or not the account exists). Any other identifier-like argument
  (card_id, transaction_id, other *_id, *_number) is rejected.

Unknown arguments that are not identifier-like are dropped, as before
(they are reported in `dropped` for logging). Invalid values are rejected
with INVALID_ARGUMENT; they are never corrected or reinterpreted, and raw
values are never echoed back.
"""

import json
import logging
import math
import os
import re
from datetime import date
from decimal import Decimal, InvalidOperation

from app.data_validation import DISPUTE_STATUSES
from app.ledger import CREDIT, CREDIT_CATEGORIES, DEBIT, DEBIT_CATEGORIES
from app.security import normalize_customer_id


logger = logging.getLogger(__name__)

# ---------- Limits ----------

DEFAULT_TRANSACTION_LIMIT = 20
HARD_MAX_TRANSACTION_LIMIT = 100        # upper bound for the configurable maximum
MAX_OFFSET = 10_000
MAX_AMOUNT = Decimal("1000000000")      # INR 100 crore: above any value in the data
MAX_STRING_LENGTH = 64                  # longer argument strings are rejected outright
MAX_LIMIT_ENV = "BANKMATE_MAX_TRANSACTION_LIMIT"


def _max_transaction_limit():
    raw = os.getenv(MAX_LIMIT_ENV, "").strip()
    if not raw:
        return 50
    if raw.isascii() and raw.isdigit() and 1 <= int(raw) <= HARD_MAX_TRANSACTION_LIMIT:
        return int(raw)
    logger.warning("Ignoring invalid %s; using 50 (allowed 1-%d)", MAX_LIMIT_ENV, HARD_MAX_TRANSACTION_LIMIT)
    return 50


MAX_TRANSACTION_LIMIT = _max_transaction_limit()


# ---------- Fixed responses (shared with the assistant) ----------

INVALID_CUSTOMER_ID = "INVALID_CUSTOMER_ID"
ACCESS_DENIED = "ACCESS_DENIED"
INVALID_ARGUMENT = "INVALID_ARGUMENT"

INVALID_CUSTOMER_ID_MESSAGE = "Customer IDs look like C101. Ask the user for a valid customer ID."
CUSTOMER_ACCESS_DENIED_MESSAGE = (
    "This customer has not been referenced by the user in this "
    "conversation. Ask the user which customer ID they mean."
)
ACCOUNT_ACCESS_DENIED_MESSAGE = "That account is not available for this customer. Use an account from the customer's profile."


def error_payload(status, message, argument=None):
    payload = {"status": status, "message": message}
    if argument is not None:
        payload["argument"] = argument
    return json.dumps(payload)


class ToolArgs:
    """Outcome of validation: `args` (normalised) when ok, else a fixed `error` payload."""

    __slots__ = ("ok", "args", "error", "status", "dropped")

    def __init__(self, ok, args=None, error=None, status=None, dropped=()):
        self.ok, self.args, self.error, self.status, self.dropped = ok, args, error, status, tuple(dropped)

    def __repr__(self):
        return f"ToolArgs(ok={self.ok}, status={self.status}, args={self.args}, dropped={self.dropped})"


class _Invalid(Exception):
    def __init__(self, argument, message):
        super().__init__(message)
        self.argument, self.message = argument, message


# ---------- Value validators (return the normalised value or raise _Invalid) ----------

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$", re.ASCII)
_ACCOUNT_ID = re.compile(r"^ACC\d{4}$", re.ASCII)
_DIGITS = re.compile(r"^\d{1,6}$", re.ASCII)
_AMOUNT_TEXT = re.compile(r"^\d{1,10}(\.\d{1,2})?$", re.ASCII)


def _string(name, value):
    if not isinstance(value, str):
        raise _Invalid(name, "must be a string")
    value = value.strip()
    if not value or len(value) > MAX_STRING_LENGTH:
        raise _Invalid(name, f"must be a non-empty string of at most {MAX_STRING_LENGTH} characters")
    return value


def _iso_date(name, value):
    text = _string(name, value)
    if not _ISO_DATE.match(text):
        raise _Invalid(name, "must be a date in YYYY-MM-DD format")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise _Invalid(name, "must be a real calendar date in YYYY-MM-DD format") from None


def _choice(allowed):
    canonical = {item.lower(): item for item in allowed}

    def check(name, value):
        text = _string(name, value)
        if text.lower() not in canonical:
            raise _Invalid(name, f"must be one of: {', '.join(allowed)}")
        return canonical[text.lower()]

    return check


def _account_id(name, value):
    text = _string(name, value).upper()
    if not _ACCOUNT_ID.match(text):
        raise _Invalid(name, "must look like ACC1001")
    return text


def _amount(name, value):
    if isinstance(value, bool):
        raise _Invalid(name, "must be a number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _Invalid(name, "must be a finite number")
        value = repr(value)
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str) or not _AMOUNT_TEXT.match(value.strip()):
        raise _Invalid(name, "must be a non-negative amount with at most 2 decimal places")
    try:
        amount = Decimal(value.strip())
    except InvalidOperation:
        raise _Invalid(name, "must be a non-negative amount") from None
    if amount > MAX_AMOUNT:
        raise _Invalid(name, f"must not exceed {MAX_AMOUNT}")
    return format(amount, "f")


def _integer(low, high_fn):
    def check(name, value):
        if isinstance(value, bool):
            raise _Invalid(name, "must be a whole number")
        if isinstance(value, str) and _DIGITS.match(value.strip()):
            value = int(value.strip())
        if not isinstance(value, int):
            raise _Invalid(name, "must be a whole number")
        high = high_fn()
        if not low <= value <= high:
            raise _Invalid(name, f"must be between {low} and {high}")
        return value

    return check


TRANSACTION_TYPES = (DEBIT, CREDIT)
CATEGORIES = tuple(sorted(DEBIT_CATEGORIES + CREDIT_CATEGORIES))

_limit = _integer(1, lambda: MAX_TRANSACTION_LIMIT)
_offset = _integer(0, lambda: MAX_OFFSET)


# ---------- Tool specifications ----------

DATE_RANGE = {"start_date": _iso_date, "end_date": _iso_date}

TOOL_SPECS = {
    "get_customer_profile": {},
    "get_customer_transactions": dict(DATE_RANGE, **{
        "transaction_type": _choice(TRANSACTION_TYPES),
        "category": _choice(CATEGORIES),
        "account_id": _account_id,
        "min_amount": _amount,
        "limit": _limit,
        "offset": _offset,
    }),
    "get_customer_metrics": dict(DATE_RANGE),
    "get_customer_disputes": {"status": _choice(DISPUTE_STATUSES)},
}

# Every customer tool requires customer_id; all other arguments are optional.
REQUIRED = ("customer_id",)
CUSTOMER_TOOLS = frozenset(TOOL_SPECS)

# Identifier-like names that are never accepted unless explicitly allowed.
_IDENTIFIER_LIKE = re.compile(r"(?:_id|_ids|_number|_no)$|^(?:card|account|transaction|dispute)s?$", re.IGNORECASE)


def allowed_arguments(tool_name):
    return REQUIRED + tuple(TOOL_SPECS[tool_name])


def validate_customer_tool_args(tool_name, raw_args, allowed_customers, customer_accounts=None):
    """
    Validate a customer tool call. `allowed_customers`: IDs the user typed in
    this conversation. `customer_accounts(customer_id)` -> set of that
    customer's account IDs; required only if an account_id filter is given
    (without it, account filters fail closed).
    """

    if tool_name not in TOOL_SPECS:
        raise ValueError(f"{tool_name!r} is not a customer tool")

    # 1. Authorization: a well-formed customer ID the user has referenced.
    raw_id = raw_args.get("customer_id") if isinstance(raw_args, dict) else None
    customer_id = normalize_customer_id(raw_id)
    if customer_id is None:
        return ToolArgs(False, error=error_payload(INVALID_CUSTOMER_ID, INVALID_CUSTOMER_ID_MESSAGE),
                        status=INVALID_CUSTOMER_ID)
    if customer_id not in allowed_customers:
        return ToolArgs(False, error=error_payload(ACCESS_DENIED, CUSTOMER_ACCESS_DENIED_MESSAGE),
                        status=ACCESS_DENIED)

    spec = TOOL_SPECS[tool_name]
    args, dropped = {"customer_id": customer_id}, []

    # 2. Unexpected arguments: identifier-like ones are rejected, others dropped.
    for name in raw_args:
        if name == "customer_id" or name in spec:
            continue
        if not isinstance(name, str) or _IDENTIFIER_LIKE.search(name):
            return ToolArgs(False, error=error_payload(
                INVALID_ARGUMENT, "Only customer_id identifies the customer; other identifiers are not accepted here."),
                status=INVALID_ARGUMENT)
        dropped.append(name)

    # 3. Optional arguments: validate and normalise; never reinterpret.
    try:
        for name, check in spec.items():
            value = raw_args.get(name)
            if value is None:
                continue
            args[name] = check(name, value)
    except _Invalid as invalid:
        return ToolArgs(False, error=error_payload(INVALID_ARGUMENT, f"{invalid.argument} {invalid.message}.",
                                                   argument=invalid.argument),
                        status=INVALID_ARGUMENT, dropped=dropped)

    if "start_date" in args and "end_date" in args and args["start_date"] > args["end_date"]:
        return ToolArgs(False, error=error_payload(INVALID_ARGUMENT, "start_date must not be after end_date.",
                                                   argument="start_date"),
                        status=INVALID_ARGUMENT, dropped=dropped)

    # 4. Account filters must belong to the authorised customer.
    if "account_id" in args:
        owned = customer_accounts(customer_id) if customer_accounts else set()
        if args["account_id"] not in owned:
            return ToolArgs(False, error=error_payload(ACCESS_DENIED, ACCOUNT_ACCESS_DENIED_MESSAGE),
                            status=ACCESS_DENIED, dropped=dropped)

    return ToolArgs(True, args=args, dropped=dropped)
