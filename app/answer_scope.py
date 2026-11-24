"""
Data-scope notes for customer answers (pure, no I/O).

Every period figure (metrics, transaction totals) is only meaningful with
its date range, and every balance with its as-of date. The system prompt
asks the model to state them (rule 50), but a model can drop them from a
short answer. This module makes that deterministic, from the tool results
only (never computed or guessed):

- Relevance: a tool result's scope is used only if the answer contains a
  money value from THAT result (tools return amounts as strings with two
  decimals). Any number format counts: ₹1,87,226.80, 187226.80, 1,87,226.8,
  4 686,00, or a bare 5890, marked or not. So a period is never attached to
  an answer about something else, or about another customer.
- Period: `period.start` / `period.end` (and `clamped`) of
  get_customer_metrics / get_customer_transactions results.
- As-of date: `as_of_date` of get_customer_profile results, for balances.
- Contradictions: if the answer states a date range that differs from the
  reported period, or a date that is neither inside a reported period nor
  present in the relevant results, the dates cannot be corrected safely.
  No range is appended then; a clear caution note is appended instead.

A date counts as mentioned in ISO form (2026-05-01), "1 May 2026",
"May 1, 2026", "01-May-2026" or day-first "01/05/2026".
"""

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation

MARKER_LINE = re.compile(r"^\s*CITED_POLICIES\s*:", re.IGNORECASE | re.MULTILINE)
BALANCE = re.compile(r"\bbalances?\b", re.IGNORECASE)
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
SPACES = "[    ]"
MONEY_STRING = re.compile(r"-?\d+\.\d{2}$")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}$")
NOT_MONEY_KEYS = {"card_last4", "last4"}
# Any number in the answer, in every format the evaluator recognises.
NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:" + SPACES + r"\d{3})+(?:,\d{1,2})?(?!\d)|\d{1,3}(?:,\d{2,3})+(?:\.\d+)?"
                    r"|\d+(?:[.,]\d+)?)(?![\w])")
MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|" \
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_FORMS = (
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s-]+" + MONTH + r"[\s,-]+(\d{4})\b", re.I),
     lambda m: (m[3], MONTHS[m[2][:3].lower()], m[1])),
    (re.compile(r"\b" + MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I),
     lambda m: (m[3], MONTHS[m[1][:3].lower()], m[2])),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), lambda m: (m[3], m[2], m[1])),
)
RANGE_JOIN = re.compile(r"^\s*(?:to|and|through|until|till|-|-{2})\s*$", re.IGNORECASE)
CONFLICT_NOTE = ("Note: the dates stated above differ from the period reported by BankMate's data tools, "
                 "so they could not be confirmed; please check the period before relying on these figures.")


def _normalise(text):
    return (text or "").translate(DASHES).replace(" ", " ").replace(" ", " ")


def _date_spans(text):
    """[(start, end, iso)] for every calendar date written in the (normalised) text, in order."""

    spans = []
    for pattern, parts in DATE_FORMS:
        for match in pattern.finditer(text):
            try:
                year, month, day = (int(p) for p in parts(match))
                spans.append((match.start(), match.end(), date(year, month, day).isoformat()))
            except (ValueError, KeyError):
                pass
    spans.sort()
    kept, last_end = [], -1
    for span in spans:                       # one date per position (formats can overlap)
        if span[0] >= last_end:
            kept.append(span)
            last_end = span[1]
    return kept


def dates_mentioned(text):
    """Every calendar date written in the text, as ISO strings."""

    return {iso for _s, _e, iso in _date_spans(_normalise(text))}


def stated_ranges(text):
    """Date ranges the answer states explicitly ("2026-05-01 to 2026-11-21", "1 May 2026 – 21 Nov 2026")."""

    text = _normalise(text)
    spans = _date_spans(text)
    return [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if RANGE_JOIN.match(text[a[1]:b[0]])]


def answer_numbers(text):
    """Absolute Decimal value of every number in the answer, whatever its format."""

    values = set()
    for match in NUMBER.finditer(_normalise(text)):
        token = match.group(1)
        if re.search(SPACES, token) or re.fullmatch(r"\d+,\d{1,2}", token):     # 4 686,00 / 686,00
            whole, _, frac = token.partition(",")
            token = re.sub(SPACES, "", whole) + ("." + frac if frac else "")
        try:
            values.add(abs(Decimal(token.replace(",", ""))))
        except InvalidOperation:
            pass
    return values


def _walk(data, key=None):
    if isinstance(data, dict):
        for k, v in data.items():
            yield from _walk(v, k)
    elif isinstance(data, list):
        for item in data:
            yield from _walk(item, key)
    else:
        yield key, data


def _parse(raw):
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("status"):          # refusals and errors carry a status
        return None
    return data


def result_customers(results):
    """Customer IDs the (raw) results belong to."""

    return {str(data["customer_id"]).upper() for data in map(_parse, results)
            if data is not None and data.get("customer_id")}


def money_values(data):
    return {abs(Decimal(v)) for k, v in _walk(data)
            if isinstance(v, str) and k not in NOT_MONEY_KEYS and MONEY_STRING.match(v)}


def relevant_results(answer, results):
    """Parsed results that contributed at least one money value to the answer."""

    numbers = answer_numbers(answer)
    relevant = []
    for raw in results:
        data = _parse(raw)
        if data is not None and money_values(data) & numbers:
            relevant.append(data)
    return relevant


def scopes_from_results(results):
    """(periods [(start, end, clamped)], as_of dates) reported by parsed or raw customer-tool results."""

    periods, as_of = [], []
    for item in results:
        data = item if isinstance(item, dict) else _parse(item)
        if data is None:
            continue
        period = data.get("period")
        if isinstance(period, dict) and period.get("start") and period.get("end") and not period.get("empty"):
            entry = (period["start"], period["end"], bool(period.get("clamped")))
            if entry not in periods:
                periods.append(entry)
        if "accounts" in data and data.get("as_of_date") and "total_balance" in data:
            if data["as_of_date"] not in as_of:
                as_of.append(data["as_of_date"])
    return periods, as_of


def _conflicting(answer, relevant, periods, as_of):
    """True when the answer states dates that contradict the reported scope."""

    reported = {(start, end) for start, end, _clamped in periods}
    if any(rng not in reported for rng in stated_ranges(answer)):
        return True
    known = {v for data in relevant for _k, v in _walk(data) if isinstance(v, str) and ISO_DATE.match(v)}
    known |= set(as_of)
    for iso in dates_mentioned(answer):
        inside = any(start <= iso <= end for start, end, _clamped in periods)
        if not inside and iso not in known:
            return True
    return False


def scope_note(answer, results):
    """The note to append, or "" when the answer already states its scope or no result is relevant to it."""

    relevant = relevant_results(answer, results)
    if not relevant:
        return ""
    periods, as_of = scopes_from_results(relevant)
    if _conflicting(answer, relevant, periods, as_of):
        return CONFLICT_NOTE
    mentioned = dates_mentioned(answer)
    lines = []
    missing = [p for p in periods if not {p[0], p[1]} <= mentioned]
    if missing:
        label = "Period covered" if len(periods) == 1 else "Periods covered"
        ranges = "; ".join(f"{start} to {end}" + (" (limited to the available data)" if clamped else "")
                           for start, end, clamped in missing)
        lines.append(f"{label}: {ranges} (inclusive).")
    if BALANCE.search(answer) and as_of and not set(as_of) <= mentioned:
        lines.append(f"Balances as of {', '.join(as_of)}.")
    return "\n".join(lines)


def add_scope_note(answer, results):
    """Answer with the scope note appended (before a CITED_POLICIES line, which must stay last)."""

    note = scope_note(answer, results)
    if not note:
        return answer
    marker = MARKER_LINE.search(answer)
    if marker:
        return answer[:marker.start()].rstrip() + "\n\n" + note + "\n" + answer[marker.start():]
    return answer.rstrip() + "\n\n" + note
