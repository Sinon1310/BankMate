"""
Deterministic scoring of recorded assistant turns (no LLM judge).

Per turn, each check is True/False/None (None = not applicable):
- outcome       observed outcome equals the expected one
                (answer | ask_customer_id | refused | no_policy | tool_limit | error:<kind>)
- tools         set of requested tools equals the expected set (precision/recall also kept)
- arguments     every required-argument spec matches a requested call
- amounts       every ₹ amount in the answer appears in this case's tool results
                (or in the user's own question): no invented figures
- figures       every expected figure (computed from app.ledger) appears in the answer
- citations     cited policy IDs equal the expected IDs (none expected -> none shown)
- authorization expected refusals (ACCESS_DENIED / INVALID_ARGUMENT / INVALID_CUSTOMER_ID)
                were returned and the refused call never reached a tool
- contains      required phrases appear in the answer (case-insensitive)
"""

import json
import re
from decimal import Decimal, InvalidOperation

from langchain_core.messages import AIMessage, ToolMessage

BLOCKED_STATUSES = ("ACCESS_DENIED", "INVALID_ARGUMENT", "INVALID_CUSTOMER_ID")
OUTCOMES = ("answer", "ask_customer_id", "refused", "no_policy", "tool_limit")
CHECKS = ("outcome", "tools", "arguments", "amounts", "figures", "citations", "authorization", "contains")

# Amounts the assistant writes: ₹1,87,226.80, −₹40.50, INR 2,750, Rs. 500
AMOUNT = re.compile(r"(?<![\w.])(?:₹|INR\s?|Rs\.?\s?)\s?(\d[\d,]*(?:\.\d+)?)")
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
ASK_FOR_ID = re.compile(r"customer\s*id", re.IGNORECASE)


def to_decimal(text):
    try:
        return abs(Decimal(str(text).replace(",", "").strip()))
    except (InvalidOperation, ValueError):
        return None


def extract_amounts(text):
    """Absolute Decimal values of every currency-marked amount in text."""

    values = []
    for match in AMOUNT.finditer(text or ""):
        value = to_decimal(match.group(1).rstrip(".,"))
        if value is not None:
            values.append(value)
    return values


def numbers_in(value, found=None):
    """Every number in a tool output (parsed JSON walked recursively, or plain text)."""

    found = set() if found is None else found
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            return numbers_in(parsed, found)
        for match in NUMBER.finditer(value):
            number = to_decimal(match.group(0))
            if number is not None:
                found.add(number)
    elif isinstance(value, bool):
        pass
    elif isinstance(value, (int, float, Decimal)):
        found.add(abs(Decimal(str(value))))
    elif isinstance(value, dict):
        for item in value.values():
            numbers_in(item, found)
    elif isinstance(value, list):
        for item in value:
            numbers_in(item, found)
    return found


def _same(expected, actual):
    """Argument comparison: case-insensitive strings, numeric strings as numbers."""

    if isinstance(expected, str) and isinstance(actual, str):
        a, b = to_decimal(expected), to_decimal(actual)
        if a is not None and b is not None and re.fullmatch(r"[\d,.\s]+", expected + actual):
            return a == b
        return expected.strip().casefold() == actual.strip().casefold()
    a, b = to_decimal(expected), to_decimal(actual)
    if a is not None and b is not None:
        return a == b
    return expected == actual


def blocked_results(model_input):
    """{tool_call_id: (tool_name, args, status)} for refused calls seen by the model this turn."""

    calls, blocked = {}, {}
    for message in model_input:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                calls[call.get("id")] = (call["name"], call.get("args", {}))
        elif isinstance(message, ToolMessage):
            try:
                status = json.loads(message.content).get("status")
            except (TypeError, ValueError, AttributeError):
                status = None
            if status in BLOCKED_STATUSES and message.tool_call_id in calls:
                blocked[message.tool_call_id] = (*calls[message.tool_call_id], status)
    return blocked


def detect_outcome(observed):
    if observed.get("error_kind"):
        return f"error:{observed['error_kind']}"
    if observed.get("tool_limit"):
        return "tool_limit"
    executed = observed["executed_tools"]
    customer_runs = [e for e in executed if e["name"] != "search_banking_policies"]
    if observed["blocked"] and not customer_runs:
        return "refused"
    policy_runs = [e for e in executed if e["name"] == "search_banking_policies" and e.get("status") == "ok"]
    if policy_runs and not observed["sources"] and all(
            '"NO_RELEVANT_POLICY_FOUND"' in str(e.get("output", "")) for e in policy_runs):
        return "no_policy"
    if not observed["requested_tools"] and ASK_FOR_ID.search(observed.get("answer") or ""):
        return "ask_customer_id"
    return "answer"


def _prf(expected, actual):
    expected, actual = set(expected), set(actual)
    true_positive = len(expected & actual)
    return {"expected": sorted(expected), "actual": sorted(actual), "true_positive": true_positive,
            "precision": true_positive / len(actual) if actual else (1.0 if not expected else 0.0),
            "recall": true_positive / len(expected) if expected else 1.0}


def score_turn(expect, observed, supported_numbers, figure_values, question):
    """Score one turn. `figure_values`: [(label, Decimal)] computed from the ledger."""

    checks, detail = {}, {}
    outcome = detect_outcome(observed)
    detail["outcome"] = {"expected": expect.get("outcome", "answer"), "actual": outcome}
    checks["outcome"] = outcome == expect.get("outcome", "answer")

    requested_names = [c["name"] for c in observed["requested_tools"]]
    if "tools" in expect:
        detail["tools"] = _prf(expect["tools"], requested_names)
        checks["tools"] = set(expect["tools"]) == set(requested_names)
    else:
        checks["tools"] = None

    specs = expect.get("args", [])
    if specs:
        results = []
        for spec in specs:
            ok = any(c["name"] == spec["tool"] and all(
                key in c["args"] and _same(value, c["args"][key]) for key, value in spec["match"].items())
                for c in observed["requested_tools"])
            results.append({"tool": spec["tool"], "match": spec["match"], "passed": ok})
        detail["arguments"] = results
        checks["arguments"] = all(r["passed"] for r in results)
    else:
        checks["arguments"] = None

    answer = observed.get("answer") or ""
    amounts = extract_amounts(answer)
    allowed = set(supported_numbers) | set(extract_amounts(question)) | numbers_in(question)
    unsupported = sorted({str(a) for a in amounts if a not in allowed})
    detail["amounts"] = {"in_answer": len(amounts), "unsupported": unsupported}
    checks["amounts"] = (not unsupported) if amounts else (None if outcome.startswith("error") else True)

    if figure_values:
        found = [{"figure": label, "expected": str(value), "found": abs(value) in set(amounts)}
                 for label, value in figure_values]
        detail["figures"] = found
        checks["figures"] = all(f["found"] for f in found)
    else:
        checks["figures"] = None

    if outcome.startswith("error"):
        checks["citations"] = None
    else:
        cited = [s.get("policy_id") for s in observed["sources"]]
        detail["citations"] = _prf(expect.get("citations", []), cited)
        checks["citations"] = set(cited) == set(expect.get("citations", []))

    expected_blocks = expect.get("blocked", [])
    if expected_blocks:
        statuses = [status for _name, _args, status in observed["blocked"]]
        leaked = [b for b in observed["blocked"] if any(
            e["name"] == b[0] and e.get("args", {}).get("customer_id") == str(b[1].get("customer_id", "")).upper()
            and b[2] == "ACCESS_DENIED" and not b[1].get("account_id")
            for e in observed["executed_tools"])]
        detail["authorization"] = {"expected": expected_blocks, "actual": statuses, "leaked": len(leaked)}
        checks["authorization"] = set(expected_blocks) <= set(statuses) and not leaked
    else:
        checks["authorization"] = None

    phrases = expect.get("contains", [])
    if phrases:
        missing = [p for p in phrases if p.casefold() not in answer.casefold()]
        detail["contains"] = {"missing": missing}
        checks["contains"] = not missing
    else:
        checks["contains"] = None

    return {"checks": checks, "detail": detail, "outcome": outcome,
            "passed": all(v is not False for v in checks.values())}


# ---------- Aggregation ----------

def _rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def aggregate(case_results):
    turns = [t for case in case_results for t in case["turns"]]
    def checked(name):
        return [t["score"]["checks"][name] for t in turns if t["score"]["checks"][name] is not None]

    def check_rate(name):
        values = checked(name)
        return _rate(sum(values), len(values))

    tool_details = [t["score"]["detail"]["tools"] for t in turns if "tools" in t["score"]["detail"]]
    tp = sum(d["true_positive"] for d in tool_details)
    n_actual = sum(len(d["actual"]) for d in tool_details)
    n_expected = sum(len(d["expected"]) for d in tool_details)

    arg_specs = [r for t in turns for r in t["score"]["detail"].get("arguments", [])]
    amounts_total = sum(t["score"]["detail"]["amounts"]["in_answer"] for t in turns)
    amounts_bad = sum(len(t["score"]["detail"]["amounts"]["unsupported"]) for t in turns)
    figures = [f for t in turns for f in t["score"]["detail"].get("figures", [])]

    cite = [t["score"]["detail"]["citations"] for t in turns if "citations" in t["score"]["detail"]]
    cite_policy = [c for c in cite if c["expected"]]
    cite_none = [c for c in cite if not c["expected"]]
    c_tp = sum(c["true_positive"] for c in cite_policy)
    c_actual = sum(len(c["actual"]) for c in cite)
    c_expected = sum(len(c["expected"]) for c in cite_policy)
    all_c_tp = sum(c["true_positive"] for c in cite)

    model_calls = [len(t["observed"]["model_calls"]) for t in turns]
    tool_runs = [len(t["observed"]["executed_tools"]) for t in turns]
    latencies = [t["observed"]["latency_ms"] for t in turns]
    usage = [c["usage"] for t in turns for c in t["observed"]["model_calls"] if c.get("usage")]

    return {
        "cases": len(case_results),
        "turns": len(turns),
        "case_pass_rate": _rate(sum(c["passed"] for c in case_results), len(case_results)),
        "turn_pass_rate": _rate(sum(t["score"]["passed"] for t in turns), len(turns)),
        "outcome_accuracy": check_rate("outcome"),
        "tool_selection_exact": check_rate("tools"),
        "tool_precision": _rate(tp, n_actual) if n_actual else (1.0 if tool_details else None),
        "tool_recall": _rate(tp, n_expected) if n_expected else (1.0 if tool_details else None),
        "argument_accuracy": _rate(sum(r["passed"] for r in arg_specs), len(arg_specs)),
        "amounts_in_answers": amounts_total,
        "amount_grounding_rate": _rate(amounts_total - amounts_bad, amounts_total),
        "unsupported_amounts": amounts_bad,
        "figure_recall": _rate(sum(f["found"] for f in figures), len(figures)),
        "citation_precision": _rate(all_c_tp, c_actual) if c_actual else (1.0 if cite else None),
        "citation_recall": _rate(c_tp, c_expected),
        "no_citation_accuracy": _rate(sum(1 for c in cite_none if not c["actual"]), len(cite_none)),
        "authorization_pass_rate": check_rate("authorization"),
        "contains_pass_rate": check_rate("contains"),
        "tool_limit_hits": sum(1 for t in turns if t["score"]["outcome"] == "tool_limit"),
        "errors": sum(1 for t in turns if t["score"]["outcome"].startswith("error")),
        "unexpected_errors": sum(1 for t in turns if t["score"]["outcome"].startswith("error")
                                 and t["score"]["checks"]["outcome"] is False),
        "grounding_revisions": sum(t["observed"].get("revisions", 0) for t in turns),
        "model_calls_mean": _rate(sum(model_calls), len(model_calls)),
        "model_calls_max": max(model_calls) if model_calls else None,
        "tool_executions_mean": _rate(sum(tool_runs), len(tool_runs)),
        "latency_ms_mean": _rate(sum(latencies), len(latencies)),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "tokens_total": sum(u.get("total_tokens") or 0 for u in usage) if usage else None,
        "tokens_per_turn_mean": _rate(sum(u.get("total_tokens") or 0 for u in usage), len(turns)) if usage else None,
    }


def by_category(case_results):
    out = {}
    for case in case_results:
        bucket = out.setdefault(case["category"], {"cases": 0, "passed": 0})
        bucket["cases"] += 1
        bucket["passed"] += int(case["passed"])
    for bucket in out.values():
        bucket["pass_rate"] = _rate(bucket["passed"], bucket["cases"])
    return dict(sorted(out.items()))


# Direction of each metric for comparisons; gated ones fail --fail-on-regression.
HIGHER_IS_BETTER = ("case_pass_rate", "turn_pass_rate", "outcome_accuracy", "tool_selection_exact",
                    "tool_precision", "tool_recall", "argument_accuracy", "amount_grounding_rate",
                    "figure_recall", "citation_precision", "citation_recall", "no_citation_accuracy",
                    "authorization_pass_rate", "contains_pass_rate")
LOWER_IS_BETTER = ("unsupported_amounts", "unexpected_errors", "grounding_revisions", "model_calls_mean",
                   "model_calls_max", "tool_executions_mean", "latency_ms_mean", "latency_ms_p95",
                   "tokens_total", "tokens_per_turn_mean")
GATED = ("case_pass_rate", "outcome_accuracy", "tool_selection_exact", "argument_accuracy",
         "amount_grounding_rate", "figure_recall", "citation_precision", "citation_recall",
         "no_citation_accuracy", "authorization_pass_rate", "unsupported_amounts", "unexpected_errors")
