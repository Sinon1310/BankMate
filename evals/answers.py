"""
Reading an assistant answer for scoring: money amounts, their labels, dates
and transaction IDs. Pure functions, no I/O.

Money amounts are found whether or not they carry a currency marker:
- marked:  ₹1,87,226.80   −₹40.50   INR 2,750   Rs. 500
- table:   any number in a markdown-table column whose header names money
           (₹, INR, Rs, amount, balance, debit, credit, total, spending, refund...)
- shaped:  unmarked numbers that look like money: digit grouping
           (1,05,465.46 / 105,465.46 / 4 686,00) or exactly two decimals.
- context: any other unmarked number shortly after a financial term in the
           same clause ("Refund of 500", "Debits were 5890 in total").
- list:    an unmarked number joined to the previous amount only by a list
           separator ("₹1,953.78 and 500", "₹4,686.00, 999 and ₹8,081.00").
Each amount records its format: currency marker, decimals, decimal
separator and grouping style, so formatting rules can be checked.

Numbers inside dates (2026-09-16, 12-Apr-2019, 03/09/2026), identifiers
(T01603, ACC1001, C106), times and percentages are never amounts. Other
bare numbers need evidence to be called non-financial: a year, card digits
("ending 8323"), a count noun after it ("5 transactions", "4 debits") or a
measure term before it ("threshold 1.2", "k=3"), a count or identifier word
before it ("debit count 12", "refund ID 7788") or a page word ("latest 20").
Small whole numbers are ordinary counts only in sentences with no money in
them; in a sentence about money they are unclassified instead. A number that fits none of
these is "unclassified" and lowers the amount-check coverage, so an
answer cannot look fully checked while ignoring a figure.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
SPACES = "[    ]"

# A number token. Lookarounds keep date parts, IDs, times and versions out.
NUMBER = re.compile(
    r"(?<![\w.,])(?<!\d[/:\-])"
    r"(?P<num>\d{1,3}(?:" + SPACES + r"\d{3})+(?:,\d{1,2})?(?!\d)"   # space grouping: 4 686,00
    r"|\d{1,3}(?:,\d{2,3})+(?:\.\d+)?"                                 # comma grouping: 1,05,465.46
    r"|\d+(?:[.,]\d+)?)"                                                # plain: 4686.00 / 686,00 / 2026
    r"(?!\w|%|[.,/:\-]\d)"
)
CURRENCY_BEFORE = re.compile(r"(₹|\bINR|\bRs\.?)\s?[−\-]?\s?$")
MONEY_HEADER = re.compile(r"₹|\bINR\b|\bRs\b|amount|balance|debit|credit|total|spend|refund|income|"
                          r"inflow|outflow|net|value|paid", re.IGNORECASE)
CARD_CONTEXT = re.compile(r"(ending|last\s*(4|four)|card|xx+|\*{2,})\W*$", re.IGNORECASE)
SEPARATOR = r"[^\w\n.;!?|]"      # within one clause: no sentence or table-cell break
FINANCIAL_TERMS = (r"(refund\w*|debit\w*|credit\w*|total\w*|spent|spend\w*|payment\w*|paid|pay|balance\w*|"
                   r"amount\w*|charge\w*|fee\w*|income|salary|withdrawal\w*|withdrew|deposit\w*|transfer\w*|"
                   r"cost\w*|price\w*|worth|sum|net|gross|inflow\w*|outflow\w*)")
FINANCIAL_BEFORE = re.compile(r"\b" + FINANCIAL_TERMS + r"\b(?:" + SEPARATOR + r"+[\w-]+){0,3}" + SEPARATOR + r"*$",
                              re.IGNORECASE)
COUNT_AFTER = re.compile(
    r"^\s+(?:[\w-]+\s+)?(transactions?|disputes?|accounts?|cards?|days?|weeks?|months?|years?|items?|"
    r"customers?|records?|rows?|times|entries|merchants?|categor(?:y|ies)|digits?|hours?|polic(?:y|ies)|"
    r"sections?|results?|matches|debits?|credits?|refunds?|payments?|withdrawals?|deposits?|transfers?|"
    r"calls?|steps?|chunks?|documents?)\b", re.IGNORECASE)
MEASURE_BEFORE = re.compile(r"\b(k|top[- ]?k|threshold|distance|score|similarity|version|v|rate|ratio|percent|"
                            r"chunks?|sections?|steps?|page|rank|limit|turns?|step)\b\W{0,3}$", re.IGNORECASE)
COUNT_OR_ID_BEFORE = re.compile(r"(?:#|\b(?P<word>count|counts|number|numbers|id|ids|reference|ref|code|serial)\b)"
                                r"(?:" + SEPARATOR + r"+[\w-]+){0,2}" + SEPARATOR + r"*$", re.IGNORECASE)
PAGE_BEFORE = re.compile(r"\b(showing|shown|displaying|listing|first|latest|last|top|recent|newest|page)\b"
                         r"(?:\W+\w+){0,2}\W*$", re.IGNORECASE)
LIST_JOIN = re.compile(r"^\s*(?:,\s*and|,\s*or|,|and|&|or|plus)\s*$", re.IGNORECASE)
MONEY_SENTENCE = re.compile(r"₹|\bINR\b|\bRs\b|\b" + FINANCIAL_TERMS + r"\b", re.IGNORECASE)
TRANSACTION_ID = re.compile(r"\bT\d{5}\b")
SENTENCE_BREAK = re.compile(r"[.!?;]\s|\n|\|")


@dataclass
class Amount:
    value: Decimal
    text: str
    start: int
    end: int
    marked: bool                 # ₹ / INR / Rs immediately before the number
    source: str                  # marked | table | shaped | context | list
    decimals: int
    comma_decimal: bool
    grouping: str                # indian | western | indian_or_western | space | none
    context: str = ""            # before + " ¦ " + after (kept for reports)
    issues: list = field(default_factory=list)
    before: str = ""             # label text preceding the amount (column header + earlier cells, or prose)
    after: str = ""              # label text following it, up to the next amount or sentence end
    in_table: bool = False


@dataclass
class Reading:
    amounts: list
    other_numbers: dict          # {"year", "count", "card", "measure", "unclassified": n}
    unclassified: list


def _parse(token):
    """(Decimal, decimals, comma_decimal, grouping) for one number token, or None."""

    if re.search(SPACES, token):
        whole, _, frac = token.partition(",")
        digits = re.sub(SPACES, "", whole)
        text = digits + ("." + frac if frac else "")
        return _decimal(text), len(frac), bool(frac), "space"
    if "," in token and "." not in token and re.fullmatch(r"\d+,\d{1,2}", token):
        whole, frac = token.split(",")
        return _decimal(f"{whole}.{frac}"), len(frac), True, "none"
    whole, _, frac = token.partition(".")
    if "," in whole:
        indian = re.fullmatch(r"\d{1,2}(,\d{2})*,\d{3}", whole) is not None
        western = re.fullmatch(r"\d{1,3}(,\d{3})+", whole) is not None
        grouping = ("indian_or_western" if indian and western else "indian" if indian
                    else "western" if western else "irregular")
    else:
        grouping = "none"
    return _decimal(whole.replace(",", "") + ("." + frac if frac else "")), len(frac), False, grouping


def _decimal(text):
    try:
        return abs(Decimal(text))
    except (InvalidOperation, ValueError):
        return None


def _table_cells(text):
    """{(cell start, cell end): (header, row text before the cell, row text after it)} for markdown tables."""

    cells, header = {}, None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            parts = stripped.strip("|").split("|")
            if all(re.fullmatch(r"\s*:?-{2,}:?\s*", p) for p in parts):
                pass                                   # separator row
            elif header is None:
                header = [p.strip() for p in parts]
            else:
                position = offset + line.index("|") + 1
                for index, part in enumerate(parts):
                    begin, end = position, position + len(part)
                    cells[(begin, end)] = (header[index] if index < len(header) else "",
                                           line[:begin - offset], line[end - offset:].rstrip("\n"))
                    position += len(part) + 1
        else:
            header = None
        offset += len(line)
    return cells


def _cell_at(cells, start):
    for (begin, end), value in cells.items():
        if begin <= start < end:
            return value
    return None


def read(text):
    """Every amount (with format and label context) and a classification of the other numbers."""

    text = (text or "").translate(DASHES)
    cells = _table_cells(text)
    amounts, unclassified = [], []
    others = {"year": 0, "count": 0, "card": 0, "measure": 0, "identifier": 0, "unclassified": 0}
    scan = _mask_dates(text)          # same length: positions still index into `text`

    for match in NUMBER.finditer(scan):
        token, start, end = match.group("num"), match.start("num"), match.end("num")
        parsed = _parse(token)
        if parsed is None or parsed[0] is None:
            continue
        value, decimals, comma_decimal, grouping = parsed
        before = text[max(0, start - 6):start]
        marked = CURRENCY_BEFORE.search(before) is not None
        cell = _cell_at(cells, start)
        clause = text[max(0, start - 60):start]
        counted = COUNT_AFTER.match(text[end:end + 40]) is not None and decimals == 0
        if marked:
            source = "marked"
        elif cell and MONEY_HEADER.search(cell[0]):
            source = "table"
        elif (grouping != "none" or decimals == 2 or comma_decimal) and not counted:
            source = "shaped"
        elif re.fullmatch(r"(19|20)\d\d", token) or counted \
                or CARD_CONTEXT.search(text[max(0, start - 16):start]) or (cell and "card" in cell[0].lower()) \
                or MEASURE_BEFORE.search(clause) or COUNT_OR_ID_BEFORE.search(clause) or PAGE_BEFORE.search(clause):
            source = None
        elif FINANCIAL_BEFORE.search(clause):
            source = "context"
        elif amounts and amounts[-1].end > _sentence_start(text, start) and LIST_JOIN.match(text[amounts[-1].end:start]):
            source = "list"                   # "₹1,953.78 and 500": the next item of a money list
        else:
            source = None

        if source is None:
            id_word = COUNT_OR_ID_BEFORE.search(clause)
            if re.fullmatch(r"(19|20)\d\d", token):
                others["year"] += 1
            elif CARD_CONTEXT.search(text[max(0, start - 16):start]) or (cell and "card" in cell[0].lower()):
                others["card"] += 1
            elif counted:
                others["count"] += 1
            elif MEASURE_BEFORE.search(clause):
                others["measure"] += 1
            elif id_word:
                others["count" if (id_word.group("word") or "").lower().startswith(("count", "number"))
                       else "identifier"] += 1
            elif PAGE_BEFORE.search(clause):
                others["count"] += 1          # "showing the latest 20": a page size
            elif decimals == 0 and value <= 1000 and not MONEY_SENTENCE.search(_sentence(text, start, end)):
                others["count"] += 1          # small whole number in a sentence with no money in it
            else:
                others["unclassified"] += 1   # could be money: lowers coverage instead of disappearing
                unclassified.append(token)
            continue

        amount = Amount(value, token, start, end, marked, source, decimals, comma_decimal, grouping)
        if not marked:
            amount.issues.append("missing_currency")
        if decimals != 2:
            amount.issues.append("decimal_precision")
        if comma_decimal:
            amount.issues.append("comma_decimal")
        if cell:
            amount.before, amount.after = f"{cell[0]} || {cell[1]}", cell[2]
            amount.in_table = True
        amounts.append(amount)

    for index, amount in enumerate(amounts):
        if not (amount.before or amount.after):
            amount.before, amount.after = _prose_context(text, amounts, index)
    _apply_respectively(text, amounts)
    for amount in amounts:
        amount.context = f"{amount.before} ¦ {amount.after}"
    return Reading(amounts, others, unclassified)


def _apply_respectively(text, amounts):
    """
    "Debits and credits were ₹A and ₹B respectively": the labels listed before
    the first amount are paired with the amounts in order. Applied only when the
    number of listed labels equals the number of amounts in the sentence;
    otherwise the ordinary before/after labels stand.
    """

    for match in re.finditer(r"\brespectively\b", text, re.IGNORECASE):
        start = _sentence_start(text, match.start())
        group = [a for a in amounts if not a.in_table and start <= a.start < match.start()]
        if len(group) < 2:
            continue
        lead = text[start:group[0].start]
        parts = [part for part in re.split(r",\s*(?:and\s+)?|\band\b|&", lead, flags=re.IGNORECASE) if part.strip()]
        if len(parts) == len(group):
            for amount, part in zip(group, parts):
                amount.before, amount.after = part, ""


def _sentence_start(text, position):
    breaks = [m.end() for m in SENTENCE_BREAK.finditer(text, 0, position)]
    return breaks[-1] if breaks else 0


def _sentence(text, start, end):
    following = SENTENCE_BREAK.search(text, end)
    return text[_sentence_start(text, start):following.start() if following else len(text)]


def _mask_dates(text):
    """Blank out every recognised date so its day, month and year never count as numbers."""

    spaced = text.replace("\u00a0", " ").replace("\u202f", " ")
    chars = list(text)
    for pattern, _parts in DATE_PATTERNS + PARTIAL_DATES:
        for match in pattern.finditer(spaced):
            chars[match.start():match.end()] = " " * (match.end() - match.start())
    return "".join(chars)


CLAUSE_SEPARATOR = re.compile(r",|\band\b|\bwhile\b|\bbut\b|\bwhereas\b|\bplus\b|\bminus\b|\bless\b|"
                              r"\bversus\b|\bvs\b|\bthen\b|\(", re.IGNORECASE)


def _prose_context(text, amounts, index):
    """
    (label text before the amount, label text after it). Text between two
    amounts of one sentence is split at its first clause separator: the part
    before it trails the earlier amount ("₹X in credits | and had ₹Y"), the
    part after it leads the later one ("₹X and | debits of ₹Y").
    """

    amount = amounts[index]
    breaks = [m.end() for m in SENTENCE_BREAK.finditer(text, 0, amount.start)]
    sentence_start = breaks[-1] if breaks else 0
    previous = amounts[index - 1] if index and amounts[index - 1].end > sentence_start else None
    if previous is None:
        before = text[sentence_start:amount.start]
    else:
        between = text[previous.end:amount.start]
        separator = CLAUSE_SEPARATOR.search(between)
        before = between[separator.end():] if separator else between
    tail_end = amount.end + 40
    following = SENTENCE_BREAK.search(text, amount.end)
    if following:
        tail_end = min(tail_end, following.start())
    if index + 1 < len(amounts):
        tail_end = min(tail_end, amounts[index + 1].start)
    after = text[amount.end:tail_end]
    separator = CLAUSE_SEPARATOR.search(after)
    return before, (after[:separator.start()] if separator else after)


# ---------- Dates and identifiers ----------

MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|" \
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_PATTERNS = (
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s-]+" + MONTH + r"[\s,-]+(\d{4})\b", re.I),
     lambda m: (m[3], MONTHS[m[2][:3].lower()], m[1])),
    (re.compile(r"\b" + MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I),
     lambda m: (m[3], MONTHS[m[1][:3].lower()], m[2])),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), lambda m: (m[3], m[2], m[1])),     # day first (India)
)


def dates_in(text):
    """Every calendar date mentioned in text, as datetime.date."""

    text = (text or "").translate(DASHES).replace(" ", " ").replace(" ", " ")
    found = set()
    for pattern, parts in DATE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                year, month, day = (int(p) for p in parts(match))
                found.add(date(year, month, day))
            except (ValueError, KeyError):
                pass
    return found


# Day and month without a year ("From 1 June to 31 August 2026"): masked so the day is never a number,
# but not dates for the period_range fact (which needs full dates).
PARTIAL_DATES = (
    (re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + MONTH + r"\b", re.I), None),
    (re.compile(r"\b" + MONTH + r"\s+\d{1,2}(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{3})", re.I), None),
)


def transaction_ids_in(text):
    return set(TRANSACTION_ID.findall(text or ""))
