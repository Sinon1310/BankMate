
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

    return [
        label for label, pattern in _COMPILED.items()
        if pattern.search(answer or "") and not pattern.search(support_text)
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
