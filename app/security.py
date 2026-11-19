"""
Security helpers (pure functions, no I/O).

- Input: questions are normalised (control characters removed) and
  length-limited before any model call.
- Customer scope: the assistant may only fetch customers whose IDs the
  user has typed in the current conversation, so text in retrieved
  documents or tool results cannot trigger lookups of other customers.
- Output: answers are rendered as markdown, so links and images are
  neutralised (no clickable phishing links, no image-URL data exfiltration).
- Secrets: API-key-like strings and the configured key are redacted from
  anything logged or shown.
"""

import os
import re


# ---------- Questions ----------

MAX_QUESTION_CHARS = 2000

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_question(text):
    """Return the question without control characters, trimmed (may be empty)."""

    return _CONTROL.sub("", str(text or "")).strip()


# ---------- Customer IDs ----------

CUSTOMER_ID = re.compile(r"^C\d{3,6}$")
_CUSTOMER_ID_IN_TEXT = re.compile(r"(?<![A-Za-z0-9])[Cc]\d{3,6}(?![A-Za-z0-9])")


def normalize_customer_id(value):
    """'c101 ' -> 'C101'; None for anything that is not a customer ID."""

    if not isinstance(value, str):
        return None

    candidate = value.strip().upper()

    if len(candidate) > 7 or not CUSTOMER_ID.match(candidate):
        return None

    return candidate


def referenced_customer_ids(texts):
    """Customer IDs typed in the given (user) texts, normalised."""

    found = set()

    for text in texts:
        for match in _CUSTOMER_ID_IN_TEXT.findall(text or ""):
            found.add(match.upper())

    return found


# ---------- Rendering model output ----------

_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\((?:[^)(]|\([^)]*\))*\)")
_HTML_IMG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_URL = re.compile(r"\b(https?)://", re.IGNORECASE)


def sanitize_answer(text):
    """
    Neutralise markdown that could exfiltrate data or mislead users:
    images are removed, links keep only their text, and bare URLs are
    defanged so they are not auto-linked. Banking answers never need them.
    """

    text = _IMAGE.sub(lambda m: f"[image removed: {m.group(1)}]" if m.group(1) else "[image removed]", text or "")
    text = _HTML_IMG.sub("[image removed]", text)
    text = _LINK.sub(lambda m: m.group(1), text)
    return _URL.sub(lambda m: m.group(1) + "[:]//", text)


# ---------- Secrets ----------

_KEY_LIKE = re.compile(r"\b(gsk|sk|pk|rk)_[A-Za-z0-9]{8,}\b")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}")


def redact_secrets(text, extra_secrets=None):
    """Redact API-key-like tokens and the configured GROQ_API_KEY value."""

    text = str(text)
    secrets = [os.getenv("GROQ_API_KEY", "")] + list(extra_secrets or [])

    for secret in secrets:
        secret = (secret or "").strip()
        if len(secret) >= 8:
            text = text.replace(secret, "[REDACTED]")

    text = _KEY_LIKE.sub("[REDACTED]", text)
    return _BEARER.sub("Bearer [REDACTED]", text)
