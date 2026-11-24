"""
Evaluation cases (evals/datasets/assistant_cases.jsonl), one JSON object per line.

{
  "id": "metrics_period_c102",            unique, stable (results are compared by id)
  "category": "metrics",                  see CATEGORIES
  "description": "...",
  "fail_tools": ["get_customer_metrics"], optional: these tools raise when executed
  "turns": [                              one or more turns in the same conversation
    {
      "question": "...",
      "expect": {                         model-independent expectations (also used live)
        "outcome": "answer",              answer | ask_customer_id | refused | no_policy |
                                          tool_limit | error:<AssistantError kind>
        "tools": ["get_customer_metrics"],          set of tools the model should request
        "args": [{"tool": "...", "match": {...}}],  required arguments (subset match)
        "figures": [{"figure": "period", ...,       computed from app.ledger at run time
                     "required": true}],            false = optional context: reported, never fails
        "facts": [{"fact": "dispute_status",        non-financial facts derived from the dataset/ledger:
                   "customer_id": "C111"}],         dispute_status | dispute_outcome | transaction_ids |
                                                    transaction_count | period_range
        "citations": ["POL-001"],                   default [] (no sources shown)
        "blocked": ["ACCESS_DENIED"],               refusals the assistant must return
        "contains": ["..."]                         required phrases (use sparingly)
      },
      "offline": {                        offline only: the scripted model and retrieval
        "script": [{"tool_calls": [{"name": "...", "args": {...}}]},
                   {"answer": "text with {{tool[#n]:path|inr}} from real tool results"}],
        "policy": ["POL-001/REQUIRED INFORMATION"]  sections the offline policy search returns
      }
    }
  ]
}
"""

import hashlib
import json
from pathlib import Path

import re

from evals.expected import FACT_KINDS, FIGURE_KINDS, PERIOD_FIELDS, SELECTION_FILTERS
from evals.scorers import BLOCKED_STATUSES, OUTCOMES

CASES_FILE = Path(__file__).resolve().parent / "datasets" / "assistant_cases.jsonl"

CATEGORIES = ("profile", "transactions", "metrics", "disputes", "policy", "out_of_scope",
              "authorization", "injection", "multi_turn", "clarification", "reliability")
TOOLS = ("get_customer_profile", "get_customer_transactions", "get_customer_metrics",
         "get_customer_disputes", "search_banking_policies")
EXPECT_KEYS = {"outcome", "tools", "args", "figures", "facts", "citations", "blocked", "contains"}
CUSTOMER_ID = re.compile(r"C\d{3}$")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}$")
# {kind: (required keys, optional keys)}; "required" and "label" are allowed on every figure.
FIGURE_KEYS = {
    "period": ({"field", "customer_id"}, {"start", "end"}),
    "balance": ({"customer_id"}, {"date"}),
    "account_balance": ({"account_id"}, {"date"}),
    "selection_total": ({"customer_id", "type"}, {"filters"}),
    "dispute_amount": ({"dispute_id"}, set()),
    "refund_amount": ({"dispute_id"}, set()),
}
FACT_KEYS = {
    "dispute_status": (set(), {"customer_id", "dispute_id"}),
    "dispute_outcome": (set(), {"customer_id", "dispute_id"}),
    "transaction_ids": ({"customer_id"}, {"filters"}),
    "transaction_count": ({"customer_id"}, {"filters"}),
    "period_range": (set(), {"customer_id", "start", "end"}),
}


class CaseError(ValueError):
    pass


def _check_turn(case_id, number, turn):
    where = f"{case_id} turn {number}"
    if not isinstance(turn.get("question"), str) or not turn["question"].strip():
        raise CaseError(f"{where}: question is required")
    expect = turn.get("expect", {})
    unknown = set(expect) - EXPECT_KEYS
    if unknown:
        raise CaseError(f"{where}: unknown expectation keys {sorted(unknown)}")
    outcome = expect.get("outcome", "answer")
    if outcome not in OUTCOMES and not outcome.startswith("error:"):
        raise CaseError(f"{where}: unknown outcome {outcome!r}")
    for name in expect.get("tools", []) + [a.get("tool") for a in expect.get("args", [])]:
        if name not in TOOLS:
            raise CaseError(f"{where}: unknown tool {name!r}")
    for spec in expect.get("figures", []):
        _check_figure(where, spec)
    for spec in expect.get("facts", []):
        _check_fact(where, spec)
    for status in expect.get("blocked", []):
        if status not in BLOCKED_STATUSES:
            raise CaseError(f"{where}: unknown blocked status {status!r}")
    script = turn.get("offline", {}).get("script")
    if not script:
        raise CaseError(f"{where}: offline.script is required")
    for step in script:
        if ("answer" in step) == ("tool_calls" in step):
            raise CaseError(f"{where}: each script step needs exactly one of answer/tool_calls")
        for call in step.get("tool_calls", []):
            if call.get("name") not in TOOLS:
                raise CaseError(f"{where}: script calls unknown tool {call.get('name')!r}")


def _check_keys(where, what, spec, required, optional):
    missing = required - set(spec)
    unknown = set(spec) - required - optional
    if missing:
        raise CaseError(f"{where}: {what} is missing {sorted(missing)}")
    if unknown:
        raise CaseError(f"{where}: {what} has unknown keys {sorted(unknown)}")
    if "customer_id" in spec and not CUSTOMER_ID.match(str(spec["customer_id"])):
        raise CaseError(f"{where}: {what} has an invalid customer_id {spec['customer_id']!r}")
    for key in ("start", "end", "date"):
        if key in spec and not ISO_DATE.match(str(spec[key])):
            raise CaseError(f"{where}: {what} {key} must be an ISO date")
    filters = spec.get("filters", {})
    if not isinstance(filters, dict) or set(filters) - set(SELECTION_FILTERS):
        raise CaseError(f"{where}: {what} filters must use {list(SELECTION_FILTERS)}")


def _check_figure(where, spec):
    kind = spec.get("figure")
    if kind not in FIGURE_KINDS:
        raise CaseError(f"{where}: unknown figure kind {kind!r}")
    required, optional = FIGURE_KEYS[kind]
    _check_keys(where, f"figure {kind}", spec, required | {"figure"}, optional | {"required", "label"})
    if "required" in spec and not isinstance(spec["required"], bool):
        raise CaseError(f"{where}: figure {kind} 'required' must be true or false")
    if kind == "period" and spec["field"] not in PERIOD_FIELDS:
        raise CaseError(f"{where}: unknown period field {spec['field']!r}")
    if kind == "selection_total" and spec["type"] not in ("Debit", "Credit"):
        raise CaseError(f"{where}: selection_total type must be Debit or Credit")
    if "label" in spec:
        try:
            re.compile(spec["label"])
        except (re.error, TypeError) as error:
            raise CaseError(f"{where}: figure {kind} label is not a valid regex ({error})") from None


def _check_fact(where, spec):
    kind = spec.get("fact")
    if kind not in FACT_KINDS:
        raise CaseError(f"{where}: unknown fact kind {kind!r}")
    required, optional = FACT_KEYS[kind]
    _check_keys(where, f"fact {kind}", spec, required | {"fact"}, optional)
    if kind.startswith("dispute_") and ("customer_id" in spec) == ("dispute_id" in spec):
        raise CaseError(f"{where}: fact {kind} needs exactly one of customer_id / dispute_id")


def load_cases(path=CASES_FILE, only=None):
    text = Path(path).read_text(encoding="utf-8")
    cases, seen = [], set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            case = json.loads(line)
        except ValueError as error:
            raise CaseError(f"line {line_number}: invalid JSON ({error})") from None
        case_id = case.get("id")
        if not case_id or case_id in seen:
            raise CaseError(f"line {line_number}: missing or duplicate id {case_id!r}")
        seen.add(case_id)
        if case.get("category") not in CATEGORIES:
            raise CaseError(f"{case_id}: unknown category {case.get('category')!r}")
        if not case.get("turns"):
            raise CaseError(f"{case_id}: at least one turn is required")
        for name in case.get("fail_tools", []):
            if name not in TOOLS:
                raise CaseError(f"{case_id}: unknown tool in fail_tools {name!r}")
        for number, turn in enumerate(case["turns"], 1):
            _check_turn(case_id, number, turn)
        cases.append(case)
    if only:
        missing = set(only) - seen
        if missing:
            raise CaseError(f"unknown case id(s): {sorted(missing)}")
        cases = [c for c in cases if c["id"] in set(only)]
    return cases


def file_sha256(path=CASES_FILE):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
