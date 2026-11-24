"""
Policy index readiness for the web app.

`initialize()` runs on each app run (from streamlit_app.py): it reuses a
valid index instantly (app.vector_store caches readiness per process) and
builds one on a fresh install, showing a spinner only while that happens.
A failure is shown as a clear banner on the policy-dependent pages (AI
Assistant and Policy Intelligence); the other pages keep working.

A failed attempt is not cached for the life of the server: later runs
retry it (at most once every RETRY_SECONDS), so a transient problem at
startup (for example the index being rebuilt by `python -m
app.vector_store` in another process, or a briefly locked database)
clears by itself. Every failure is logged with its full traceback; the
UI never shows raw error text. Policy searches also retry the build.

Set BANKMATE_INDEX_AUTOINIT=0 to skip startup initialisation (tests).
"""

import logging
import os
import threading
import time

import streamlit as st


logger = logging.getLogger(__name__)

_status = {"ok": None, "error": None, "chunks": 0, "built": False, "attempted_at": None}

# Minimum time between automatic retries after a failure (seconds).
RETRY_SECONDS = 30

_attempt_lock = threading.Lock()

INDEX_ERROR_MESSAGE = (
    "BankMate's policy index isn't available, so policy search and policy "
    "answers are disabled. Customer and transaction features still work."
)


def _enabled():
    return os.getenv("BANKMATE_INDEX_AUTOINIT", "1").strip() != "0"


def _initialize_once():
    """One attempt to make the index available (instant when it is valid). Never raises."""

    from app.vector_store import ensure_policy_index

    try:
        status = ensure_policy_index()
        return {"ok": True, "error": None, "chunks": status.count, "built": status.built}
    except Exception as error:  # noqa: BLE001 - reported in the UI and logs
        # Full traceback and exception type in the server log, for diagnosis.
        logger.exception("Policy index initialisation failed: %s", error)
        return {"ok": False, "error": str(error) or error.__class__.__name__,
                "chunks": 0, "built": False}


def _forget_attempts():
    _status["attempted_at"] = None


# Kept for callers that cleared the former st.cache_resource: forgets the
# last attempt so the next run retries immediately.
_initialize_once.clear = _forget_attempts


def _attempt(show_spinner):
    """Try to initialise, unless a failed attempt happened within RETRY_SECONDS."""

    with _attempt_lock:
        last = _status.get("attempted_at")
        if _status.get("ok") is False and last is not None and time.monotonic() - last < RETRY_SECONDS:
            return
        _status["attempted_at"] = time.monotonic()
        if show_spinner:
            with st.spinner("Preparing the policy index… this only happens on first launch."):
                result = _initialize_once()
        else:
            result = _initialize_once()
        _status.update(result)


def initialize():
    """Prepare the policy index (reused instantly once valid). Never raises."""

    if not _enabled():
        return None

    from app.vector_store import is_index_ready

    if is_index_ready():
        _status.update(ok=True, error=None)
        return dict(_status)

    _attempt(show_spinner=True)
    return dict(_status)


def index_error():
    """The current error if the index is still unavailable, else None."""

    if _status.get("ok") is not False:
        return None

    from app.vector_store import is_index_ready

    if is_index_ready():          # a later search or retry made it available
        _status.update(ok=True, error=None)
        return None

    if _enabled():
        _attempt(show_spinner=False)          # rate-limited retry
        if _status.get("ok"):
            return None

    return _status.get("error") or "unknown error"


def show_index_banner():
    """Render the unavailable-index banner on policy-dependent pages."""

    error = index_error()

    if error:
        st.error(
            f"{INDEX_ERROR_MESSAGE} Rebuild it with `python -m app.vector_store`. "
            "Details are in the server logs.",
            icon=":material/error:",
        )

    return error
