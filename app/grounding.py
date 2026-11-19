
"""
Deterministic check that policy answers do not introduce contact
channels (phone, chat, branch, email, app, ...) that the session's
supporting text does not mention.

Supporting text = retrieved policy evidence plus customer tool results,
accumulated for the whole session so follow-up answers are checked too.
A channel is "supported" only if the same pattern matches that text.

Patterns deliberately avoid bare everyday words such as "call" or "app"
to limit false positives.
"""

import re

from app.citations import parse_policy_result


CHANNEL_PATTERNS = {
    "phone": (
        r"\b(?:tele)?phone\b|\bphone call\b|\bhelpline\b|\bhotline\b"
        r"|\btoll[- ]free\b|\bcustomer care number\b"
        r"|\bcall(?:ing)? (?:the |your |our )?"
        r"(?:bank|us|helpline|hotline|customer care|support|branch)\b"
    ),
    "chat": r"\b(?:live[- ])?chat(?:bot)?\b",
    "branch": r"\b(?:in[- ])?branch(?:es)?\b",
    "email": r"\be-?mail\b",
    "SMS": r"\bsms\b|\btext messages?\b",
    "mobile app": r"\bmobile (?:banking )?app\b|\bbanking app\b",
    "website": (
        r"\bwebsite\b|\bweb portal\b|\bonline banking\b"
        r"|\binternet banking\b|\bnet ?banking\b"
    ),
    "WhatsApp": r"\bwhatsapp\b",
    "social media": r"\bsocial media\b",
}

_COMPILED = {
    label: re.compile(pattern, re.IGNORECASE)
    for label, pattern in CHANNEL_PATTERNS.items()
}

_PARENTHETICAL = re.compile(r"[ \t]*\([^()\n]*\)")

# An example list introduced by "e.g." / "for example" / "such as" /
# "including", running to the end of the clause.
_EXAMPLE_CLAUSE = re.compile(
    r",?[ \t]*\b(?:e\.g\.|for example|for instance|such as|including)"
    r"[^.;:\n]*",
    re.IGNORECASE
)


# Clauses that say the policy does NOT give a detail ("The policy does not
# specify any fees") are the grounded behaviour we ask for, so they must
# not count as claims. Matching is per clause, so "doesn't specify fees,
# but it is credited back to your card" is still caught.
_NOT_SPECIFIED = re.compile(
    r"\b(?:does not|doesn't|do not|don't|did not|didn't)\s+"
    r"(?:specify|mention|state|say|include|cover|provide|list|describe|indicate|define|contain)\b"
    r"|\bnot\s+(?:specified|mentioned|stated|covered|provided|defined|included|listed)\b"
    r"|\bno\s+(?:information|details?|mention)\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\n|;|,\s*(?:but|however|although|though|while|whereas)\b|\s+but\s+",
    re.IGNORECASE,
)


def _claims_only(text):
    """Text with 'the policy does not specify ...' clauses removed."""

    return " ".join(
        clause for clause in _CLAUSE_SPLIT.split(text or "")
        if clause and not _NOT_SPECIFIED.search(clause)
    )


def collect_evidence_text(evidence_texts, tool_result):
    """Append evidence content from a search_banking_policies result."""

    data = parse_policy_result(tool_result)

    for item in (data or {}).get("evidence") or []:
        if isinstance(item, dict) and item.get("content"):
            evidence_texts.append(str(item["content"]))

    return evidence_texts


def find_unsupported_channels(answer, support_text):
    """Channel labels mentioned in the answer but absent from support."""

    support_text = support_text or ""
    claims = _claims_only(answer)

    return [
        label for label, pattern in _COMPILED.items()
        if pattern.search(claims) and not pattern.search(support_text)
    ]


def remove_unsupported_examples(answer, labels):
    """
    Conservative cleanup: remove only parenthetical asides and example
    clauses ("e.g. ...", "such as ...", "including ...") that name an
    unsupported channel. Whole sentences are never removed, so the
    supported instruction around an invented example is kept.
    """

    patterns = [_COMPILED[label] for label in labels if label in _COMPILED]

    def drop_if_unsupported(match):
        text = match.group(0)
        return "" if any(p.search(text) for p in patterns) else text

    cleaned = _PARENTHETICAL.sub(drop_if_unsupported, answer)
    cleaned = _EXAMPLE_CLAUSE.sub(drop_if_unsupported, cleaned)

    return cleaned.strip()


# ---------- Procedural details (policy answers) ----------
#
# Policy answers must not complete a procedure with outcomes the
# retrieved policy text does not state. These categories are details a
# model commonly fills in from general banking knowledge. A category is
# "supported" only if the same pattern matches the retrieved policy
# evidence (not customer data).

DETAIL_PATTERNS = {
    "refund destination": (
        r"\boriginal (?:payment )?(?:method|mode|source|card|account)\b"
        r"|\bcredited back\b"
        r"|\bcredit(?:s|ed|ing)?\b[^.\n]{0,40}\bback\b"
        r"|\b(?:credited|refunded|returned|reversed|deposited|transferred)\s+(?:back\s+)?(?:to|into)\b"
        r"|\bback (?:to|into) (?:the |your |their )?(?:customer'?s?|card|account|bank|original)"
        # "once the funds are returned", "the amount is reimbursed": a money
        # word followed by a settlement verb, even without a destination.
        r"|\b(?:funds|money|amount|refund)\b[^.\n]{0,40}?\b(?:are|is|be|been|being|get|gets|got)\s+"
        r"(?:returned|credited|reimbursed|deposited|reversed|paid back|sent back)\b"
    ),
    "timeframe": (
        r"\bwithin\s+(?:\d+|a few|one|two|three|seven|ten|thirty)\b"
        r"|\b\d+\s*(?:-|–|to)\s*\d+\s*(?:business |working |calendar )?(?:days?|hours?|weeks?)\b"
        r"|\b\d+\s*(?:business |working |calendar )?(?:days?|hours?|weeks?)\b"
        r"|\bsame[- ]day\b"
    ),
    "fees": (
        r"\bfees?\b|\bcharges? (?:may |will )?apply\b|\bpenalt(?:y|ies)\b|\bfree of charge\b"
    ),
}

DETAIL_DESCRIPTIONS = {
    "refund destination": "where the refunded amount is credited",
    "timeframe": "a timeframe",
    "fees": "any fees or charges",
}

_DETAILS = {
    label: re.compile(pattern, re.IGNORECASE)
    for label, pattern in DETAIL_PATTERNS.items()
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")


def find_unsupported_details(answer, evidence_text):
    """Detail categories stated in the answer but absent from evidence."""

    evidence_text = evidence_text or ""
    claims = _claims_only(answer)

    return [
        label for label, pattern in _DETAILS.items()
        if pattern.search(claims) and not pattern.search(evidence_text)
    ]


def not_specified_statement(label):
    return f"The policy does not specify {DETAIL_DESCRIPTIONS.get(label, label)}."


def replace_unsupported_detail_sentences(answer, labels):
    """
    Last-resort fallback when the model will not revise: each sentence
    stating an unsupported detail is replaced by an explicit
    "The policy does not specify ..." statement (once per category), so
    an invented procedure is never shown as policy.
    """

    patterns = [(label, _DETAILS[label]) for label in labels if label in _DETAILS]
    announced = set()
    lines = []

    for line in answer.split("\n"):
        prefix_match = _LIST_PREFIX.match(line)
        prefix = prefix_match.group(1) if prefix_match else ""
        body = line[len(prefix):]

        sentences = _SENTENCE_SPLIT.split(body) if body.strip() else [body]
        kept = []

        for sentence in sentences:
            hits = [label for label, pattern in patterns if pattern.search(sentence)]

            if not hits:
                kept.append(sentence)
                continue

            for label in hits:
                if label not in announced:
                    announced.add(label)
                    kept.append(not_specified_statement(label))

        text = " ".join(part for part in kept if part)

        if text.strip() or not body.strip():
            lines.append(prefix + text if text.strip() else line)

    return "\n".join(lines).strip()
