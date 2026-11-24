"""
Deterministic scoring of recorded assistant turns (no LLM judge).

Per turn, each check is True/False/None (None = not applicable):
- outcome       observed outcome equals the expected one
                (answer | ask_customer_id | refused | no_policy | tool_limit | error:<kind>)
- tools         set of requested tools equals the expected set (precision/recall also kept)
- arguments     every required-argument spec matches a requested call
- amounts       every money amount in the answer (currency-marked, in a money column of a
                table, or money-shaped such as 1,05,465.46 / 4 686,00) appears in this
                case's tool results (or in the user's own question): no invented figures.
                None when no amount was checked: grounding is never claimed vacuously.
- figures       every REQUIRED expected figure (computed from app.ledger) appears in the
                answer and is not attached to a conflicting label (credits shown as debits).
                Optional figures are reported separately and never fail a turn.
- facts         non-financial facts derived from the dataset/ledger: dispute status and
                outcome, transaction IDs and counts, period ranges
Formatting (₹ on each amount, two decimals, no comma decimal separator) is
reported per turn and in the metrics but does not fail a turn.
- citations     cited policy IDs equal the expected IDs (none expected -> none shown)
- authorization expected refusals (ACCESS_DENIED / INVALID_ARGUMENT / INVALID_CUSTOMER_ID)
                were returned and the refused call never reached a tool
- contains      required phrases appear in the answer (case-insensitive)
"""

import json
import re
from decimal import Decimal, InvalidOperation

from langchain_core.messages import AIMessage, ToolMessage

from evals import answers

BLOCKED_STATUSES = ("ACCESS_DENIED", "INVALID_ARGUMENT", "INVALID_CUSTOMER_ID")
OUTCOMES = ("answer", "ask_customer_id", "refused", "no_policy", "tool_limit")
CHECKS = ("outcome", "tools", "arguments", "amounts", "figures", "facts", "citations", "authorization",
          "contains")

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


def score_turn(expect, observed, supported_numbers, figure_values, question, facts=None):
    """
    Score one turn.
    `figure_values`: expected figures from app.ledger, each a dict
        {"label", "value": Decimal, "spec": figure spec, "required": bool}
        (a bare (label, value) tuple is a required figure with no label binding);
    `facts`: expected facts from evals.expected.fact().
    """

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
    reading = answers.read(answer)
    allowed = set(supported_numbers) | set(extract_amounts(question)) | numbers_in(question)
    unsupported = sorted({str(a.value) for a in reading.amounts if a.value not in allowed})
    issues = {}
    for amount in reading.amounts:
        for issue in amount.issues:
            issues[issue] = issues.get(issue, 0) + 1
    detail["amounts"] = {
        "checked": len(reading.amounts),
        "by_source": {src: sum(1 for a in reading.amounts if a.source == src)
                      for src in ("marked", "table", "shaped", "context")},
        "unsupported": unsupported,
        # Numbers that could be money but could not be classified: they lower coverage, never vanish.
        "unclassified_numbers": reading.unclassified,
        "candidates": len(reading.amounts) + len(reading.unclassified),
        "non_financial": {k: v for k, v in reading.other_numbers.items() if k != "unclassified"},
        "coverage": _rate(len(reading.amounts), len(reading.amounts) + len(reading.unclassified)),
    }
    detail["formatting"] = {"issues": issues,
                            "amounts_with_issues": sum(1 for a in reading.amounts if a.issues),
                            "examples": [a.text for a in reading.amounts if a.issues][:5]}
    checks["amounts"] = (not unsupported) if reading.amounts else None   # nothing checked: never "grounded"

    if figure_values:
        found = [_score_figure(item, reading, figure_values) for item in figure_values]
        detail["figures"] = found
        required = [f for f in found if f["required"]]
        checks["figures"] = all(f["passed"] for f in required) if required else None
    else:
        checks["figures"] = None

    if facts:
        results = [check_fact(item, answer) for item in facts]
        detail["facts"] = results
        checks["facts"] = all(r["passed"] for r in results)
    else:
        checks["facts"] = None

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


# ---------- Figures and their labels ----------

CREDIT = r"credit|inflow|received|deposit"
DEBIT = r"debit|outflow|withdraw|charged"
FIELD_LABELS = {
    "total_credits": CREDIT, "total_debits": DEBIT,
    "net_cash_flow": r"net\s*(transaction\s*|cash\s*)?flow|credits?\s*(minus|less|-)\s*debits?",
    "income": r"income|salary|pension|earn", "other_inflows": r"transfers?\s*in|other\s*inflow",
    "transfer_out_total": r"transfers?\s*out", "refund_total": r"refund",
    "spending_gross": r"gross|before\s*(\w+\s*){0,2}refunds?",
    "spending_net": r"net\s*spend|spending\s*net|after\s*(\w+\s*){0,3}refunds?|net\s*of\s*refunds?|less\s*refunds?",
}
OPPOSITES = {"total_credits": "total_debits", "total_debits": "total_credits",
             "spending_gross": "spending_net", "spending_net": "spending_gross",
             "Debit": "Credit", "Credit": "Debit"}


def _label_key(spec):
    kind = spec.get("figure")
    if kind == "period":
        return spec.get("field")
    if kind == "selection_total":
        return spec.get("type")
    return kind


def label_pattern(spec):
    """Regex for the words that name a figure (None when no label can be expected)."""

    if spec is None:
        return None
    if spec.get("label"):
        return spec["label"]
    kind, key = spec.get("figure"), _label_key(spec)
    if kind == "period":
        return FIELD_LABELS.get(key)
    if kind == "selection_total":
        return DEBIT if key == "Debit" else CREDIT + r"|refund"
    if kind == "balance":
        return r"balance|total"
    if kind == "account_balance":
        return re.escape(spec["account_id"]) + r"|balance"
    if kind == "dispute_amount":
        return r"amount|disputed|dispute|charge|transaction|" + re.escape(spec["dispute_id"])
    if kind == "refund_amount":
        return r"refund"
    return None


def _conflict_patterns(item, figure_values):
    """Labels of the figure's opposite (credits vs debits, gross vs net)."""

    spec = item.get("spec")
    if spec is None:
        return []
    opposite = OPPOSITES.get(_label_key(spec))
    if not opposite:
        return []
    patterns = [label_pattern(o["spec"]) for o in figure_values
                if isinstance(o, dict) and o.get("spec") is not None and o["spec"] is not spec
                and _label_key(o["spec"]) == opposite]
    if not patterns:
        patterns = [FIELD_LABELS.get(opposite) if spec.get("figure") == "period"
                    else (DEBIT if opposite == "Debit" else CREDIT)]
    return [p for p in patterns if p]


def _matches(pattern, text):
    return [m for m in re.finditer(pattern, text, re.IGNORECASE)] if pattern else []


def _without_parentheticals(text):
    text = re.sub(r"\([^()]*\)", " ", text)                 # complete asides
    text = re.sub(r"^[^()]*\)", " ", text)                   # the end of an aside opened before this segment
    return re.sub(r"\([^()]*$", " ", text)                   # an aside still open at the end of this segment


def _has_label(text, own, conflicts):
    return bool(_matches(own, text) or any(_matches(c, text) for c in conflicts))


def _occurrence_status(amount, own, conflicts):
    """
    The label of one occurrence of a figure's value. The text BEFORE the amount
    decides first ("debits of ₹X"); the text after it is used only when nothing
    relevant precedes it ("₹X in debits"). Within one side, the label nearest
    to the amount wins, so "Credits of ₹X and debits of ₹Y" gives X the credit
    label even though "debits" follows it.
    """

    before = getattr(amount, "before", None)
    after = getattr(amount, "after", None)
    if before is None and after is None:                      # older Amount objects: one context string
        before, _, after = amount.context.partition(" ¦ ")
    for side, full in (("before", before or ""), ("after", after or "")):
        # A parenthetical aside ("Total debits (credits excluded) were ₹X") never outranks the
        # label outside it; it is consulted only when nothing outside it names the amount.
        outer = _without_parentheticals(full)
        text = outer if _has_label(outer, own, conflicts) else full
        own_hits = _matches(own, text)
        conflict_hits = [m for c in conflicts for m in _matches(c, text)]
        if not own_hits and not conflict_hits:
            continue
        if own_hits and not conflict_hits:
            return "labelled"
        if conflict_hits and not own_hits:
            return "mislabelled"
        if side == "before":                                   # both present: the closer one names the amount
            return "labelled" if max(m.end() for m in own_hits) > max(m.end() for m in conflict_hits) \
                else "mislabelled"
        return "labelled" if min(m.start() for m in own_hits) < min(m.start() for m in conflict_hits) \
            else "mislabelled"
    return "unlabelled"


def _score_figure(item, reading, figure_values):
    """
    One expected figure against the amounts in the answer:
    labelled    the value appears next to the figure's own label;
    unlabelled  the value appears, but with no recognisable label (accepted);
    mislabelled the value appears under its opposite's label and nowhere under its own (fails);
    missing     the value does not appear (fails when required).
    """

    if isinstance(item, tuple):
        item = {"label": item[0], "value": item[1], "spec": None, "required": True}
    value = abs(item["value"])
    own = label_pattern(item.get("spec"))
    conflicts = _conflict_patterns(item, figure_values)
    matches = [a for a in reading.amounts if a.value == value]
    statuses = [_occurrence_status(a, own, conflicts) for a in matches]
    status = ("missing" if not matches else "labelled" if "labelled" in statuses
              else "mislabelled" if "mislabelled" in statuses else "unlabelled")
    return {"figure": item["label"], "expected": str(item["value"]), "required": item.get("required", True),
            "found": bool(matches), "label_status": status, "passed": status in ("labelled", "unlabelled")}


# ---------- Facts ----------

STATUS_PATTERNS = {
    "Open": r"\b(is|remains|still|status\W{0,3})\s*\**\s*open\b|\bopen\s+dispute",
    "Under Investigation": r"\bunder\s+investigation\b",
    "Resolved": r"\bresolved\b",
    "Rejected": r"\brejected\b",
}
REFUND_CLAIM = (r"\b(full|partial)\s+refund\b|\brefund(ed)?\b[^.\n;]{0,30}\b(issued|credited|processed|approved|"
                r"paid|received|made|given)\b|\b(was|were|has been|have been|got|received)\s+(a\s+)?refund(ed)?\b")
# Negated refund statements, removed before looking for a refund claim:
# "No refund was issued", "without a refund", "no full refund", "not refunded",
# "the refund was not issued", "did not receive a refund", "no refund or adjustment".
NEGATED_REFUND = re.compile(
    r"\b(no|without|never|nor|neither)\s+(?:any\s+|a\s+|an\s+)?(?:full\s+|partial\s+)?refunds?\b"
    r"(?:[^.\n;,]{0,30}?\b(?:issued|credited|processed|approved|paid|received|made|given)\b)?"
    # "not" / "never" governing the refund: up to two verb or article words in between, but not
    # "not only was a refund issued" or "never before had a refund been processed" (affirmative).
    r"|\b(?:not|never|n't)\s+(?:(?!only\b|before\b|long\b|just\b|merely\b|simply\b|until\b|since\b|once\b|"
    r"sooner\b|surprisingly\b)\w+\s+){0,2}?(?:a\s+|any\s+)?(?:full\s+|partial\s+)?refund(?:ed|s)?\b"
    r"|\brefunds?\s+(?:was|were|is|has|have|had)\s+(?:not|never)\b(?:\s+been)?(?:\s+\w+)?"
    r"|\brefunds?\s+(?:wasn't|weren't|isn't|hasn't|haven't|hadn't)\b(?:\s+been)?(?:\s+\w+)?",
    re.IGNORECASE)


def without_negated_refunds(text):
    """Text with negated refund statements blanked out (same length)."""

    return NEGATED_REFUND.sub(lambda m: " " * len(m.group(0)), text)
OUTCOME_PATTERNS = {
    "Full refund": r"\bfull(y)?\s+refund(ed)?\b|\brefunded\s+in\s+full\b",
    "Partial refund": r"\bpartial(ly)?\s+refund(ed)?\b",
    "No financial adjustment": r"\bno\s+(financial\s+)?adjustment\b|\bno\s+refund\b|\bnot\s+refunded\b",
}
PAGE_WORDS = re.compile(r"\b(showing|shown|show|displaying|displayed|listing|listed|returned|returning|first|"
                        r"latest|last|top|recent|newest|page)\b(?:\W+\w+){0,2}\W*$", re.IGNORECASE)
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def _present(pattern, text):
    return re.search(pattern, text, re.IGNORECASE) is not None


def check_fact(item, answer):
    """{"fact", "kind", "expected", "passed", "problems": [...]} for one expected fact."""

    kind, expected = item["kind"], item["expected"]
    text = (answer or "").translate(answers.DASHES)
    problems = []

    if kind == "dispute_status":
        wanted = {d["status"] for d in expected}
        for status in sorted(wanted):
            if not _present(STATUS_PATTERNS[status], text):
                problems.append(f"status {status!r} not stated")
        for status, pattern in STATUS_PATTERNS.items():
            if status not in wanted and _present(pattern, text):
                problems.append(f"conflicting status {status!r} stated")

    elif kind == "dispute_outcome":
        wanted = {d["outcome"] for d in expected if d["outcome"]}
        for outcome in sorted(wanted):
            if not _present(OUTCOME_PATTERNS.get(outcome, re.escape(outcome)), text):
                problems.append(f"outcome {outcome!r} not stated")
        affirmative = without_negated_refunds(text)
        if not any(d["refunded"] for d in expected) and _present(REFUND_CLAIM, affirmative):
            problems.append("claims a refund the dispute does not have")
        stated = {o for o, pattern in OUTCOME_PATTERNS.items()
                  if _present(pattern, text if o == "No financial adjustment" else affirmative)}
        for outcome in sorted(stated - wanted):
            if outcome != "No financial adjustment" or wanted:
                problems.append(f"conflicting outcome {outcome!r} stated")

    elif kind == "transaction_ids":
        listed = answers.transaction_ids_in(text)
        missing, extra = sorted(set(expected) - listed), sorted(listed - set(expected))
        if missing:
            problems.append(f"missing transaction IDs {missing}")
        if extra:
            problems.append(f"transaction IDs not in the selection {extra}")

    elif kind == "transaction_count":
        count = expected
        words = rf"(?<![-/.,\d])(\b{count}\b|\b{NUMBER_WORDS.get(count, '@@')}\b)"
        stated = _present(words + r"[ \t]+([\w-]+[ \t]+){0,3}?transactions?\b", text) or \
            _present(r"\btransactions?\b[ \t:]+([\w-]+[ \t]+){0,3}?" + words, text)
        if not stated and len(answers.transaction_ids_in(text)) != count:
            problems.append(f"count {count} neither stated nor shown as {count} listed transactions")
        for match in re.finditer(r"(?<![-/.,\d])\b(\d+)[ \t]+([\w-]+[ \t]+){0,3}?transactions?\b", text,
                                 re.IGNORECASE):
            if re.fullmatch(r"(19|20)\d\d", match.group(1)):
                continue
            # "20 of 36 transactions": 20 is the page shown, 36 the claimed total.
            of_total = re.match(r"[ \t]+of[ \t]+(\d+)\b", text[match.end(1):match.end()])
            if of_total:
                claimed = int(of_total.group(1))
            elif PAGE_WORDS.search(text[max(0, match.start() - 40):match.start()]):
                continue                      # "showing the latest 20 transactions": a page size, not a total
            else:
                claimed = int(match.group(1))
            if claimed != count:
                problems.append(f"conflicting count {claimed} stated")

    elif kind == "period_range":
        mentioned = answers.dates_in(text)
        for label, day in zip(("start", "end"), expected):
            if answers.date.fromisoformat(day) not in mentioned:
                problems.append(f"period {label} {day} not stated")

    else:
        problems.append(f"unknown fact kind {kind!r}")

    return {"fact": item["label"], "kind": kind, "expected": expected, "passed": not problems,
            "problems": problems}


# ---------- Aggregation ----------

def _rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def aggregate(case_results, mode="offline"):
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
    amount_details = [t["score"]["detail"]["amounts"] for t in turns]
    amounts_total = sum(d.get("checked", d.get("in_answer", 0)) for d in amount_details)
    amounts_bad = sum(len(d["unsupported"]) for d in amount_details)
    unclassified = sum(len(d.get("unclassified_numbers", [])) for d in amount_details)
    format_details = [t["score"]["detail"].get("formatting", {}) for t in turns]
    format_counts = {}
    for d in format_details:
        for issue, count in d.get("issues", {}).items():
            format_counts[issue] = format_counts.get(issue, 0) + count
    figures = [f for t in turns for f in t["score"]["detail"].get("figures", [])]
    required = [f for f in figures if f.get("required", True)]
    optional = [f for f in figures if not f.get("required", True)]
    found = [f for f in figures if f.get("found")]
    facts = [f for t in turns for f in t["score"]["detail"].get("facts", [])]

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
    provider = [c["latency_ms"] for t in turns for c in t["observed"]["model_calls"]
                if c.get("latency_ms") is not None and not c.get("error")]
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
        "amounts_checked": amounts_total,
        "amount_candidates": amounts_total + unclassified,
        "unclassified_numbers": unclassified,
        "amount_check_coverage": _rate(amounts_total, amounts_total + unclassified),
        "amount_grounding_rate": _rate(amounts_total - amounts_bad, amounts_total),   # None if nothing checked
        "unsupported_amounts": amounts_bad,
        "amount_format_issues": sum(d.get("amounts_with_issues", 0) for d in format_details),
        "format_missing_currency": format_counts.get("missing_currency", 0),
        "format_decimal_precision": format_counts.get("decimal_precision", 0),
        "format_comma_decimal": format_counts.get("comma_decimal", 0),
        "figure_recall": _rate(sum(f.get("passed", f.get("found")) for f in required), len(required)),
        "optional_figure_recall": _rate(sum(f.get("passed", f.get("found")) for f in optional), len(optional)),
        "figure_label_rate": _rate(sum(f.get("label_status") == "labelled" for f in found), len(found)),
        "mislabelled_figures": sum(f.get("label_status") == "mislabelled" for f in figures),
        "fact_accuracy": _rate(sum(f["passed"] for f in facts), len(facts)),
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
        "latency_ms_mean": _rate(sum(latencies), len(latencies)),       # per turn, pacing waits excluded
        "latency_ms_p95": _percentile(latencies, 0.95),
        "provider_latency_ms_mean": _rate(sum(provider), len(provider)) if mode == "live" else None,
        "provider_latency_ms_p95": _percentile(provider, 0.95) if mode == "live" else None,
        "pacing_ms_total": round(sum(t["observed"].get("paced_ms", 0) or 0 for t in turns), 3),
        # Usage reported for these turns: by the provider (live) or as recorded in the cassette (replay).
        "tokens_total": sum(u.get("total_tokens") or 0 for u in usage) if usage else None,
        "tokens_per_turn_mean": _rate(sum(u.get("total_tokens") or 0 for u in usage), len(turns)) if usage else None,
        # Tokens this run actually spent with the provider: only a live run spends any.
        "tokens_consumed": sum(u.get("total_tokens") or 0 for u in usage) if mode == "live" else 0,
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
                    "amount_check_coverage", "figure_recall", "optional_figure_recall", "figure_label_rate",
                    "fact_accuracy", "citation_precision", "citation_recall", "no_citation_accuracy",
                    "authorization_pass_rate", "contains_pass_rate")
LOWER_IS_BETTER = ("unsupported_amounts", "unexpected_errors", "grounding_revisions", "model_calls_mean",
                   "model_calls_max", "tool_executions_mean", "latency_ms_mean", "latency_ms_p95",
                   "provider_latency_ms_mean", "provider_latency_ms_p95", "pacing_ms_total",
                   "tokens_total", "tokens_per_turn_mean", "tokens_consumed", "amount_format_issues",
                   "format_missing_currency", "format_decimal_precision", "format_comma_decimal",
                   "mislabelled_figures", "unclassified_numbers")
# Gated: a worse value is a regression (compare --fail-on-regression).
GATED = ("case_pass_rate", "outcome_accuracy", "tool_selection_exact", "argument_accuracy",
         "amount_grounding_rate", "figure_recall", "fact_accuracy", "citation_precision", "citation_recall",
         "no_citation_accuracy", "authorization_pass_rate", "unsupported_amounts", "unexpected_errors",
         "mislabelled_figures")
# Depend on where the run happened (provider, pacing, local replay): comparable only between runs of the same mode.
MODE_DEPENDENT = ("latency_ms_mean", "latency_ms_p95", "provider_latency_ms_mean", "provider_latency_ms_p95",
                  "pacing_ms_total", "tokens_consumed")
