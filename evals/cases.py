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
        "figures": [{"figure": "period", ...}],     computed from app.ledger at run time
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

from evals.expected import FIGURE_KINDS
from evals.scorers import BLOCKED_STATUSES, OUTCOMES

CASES_FILE = Path(__file__).resolve().parent / "datasets" / "assistant_cases.jsonl"

CATEGORIES = ("profile", "transactions", "metrics", "disputes", "policy", "out_of_scope",
              "authorization", "injection", "multi_turn", "clarification", "reliability")
TOOLS = ("get_customer_profile", "get_customer_transactions", "get_customer_metrics",
         "get_customer_disputes", "search_banking_policies")
EXPECT_KEYS = {"outcome", "tools", "args", "figures", "citations", "blocked", "contains"}


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
        if spec.get("figure") not in FIGURE_KINDS:
            raise CaseError(f"{where}: unknown figure kind {spec.get('figure')!r}")
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
