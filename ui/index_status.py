"""
Policy index readiness for the web app.

`initialize()` runs once per server process (from streamlit_app.py):
it reuses a valid index instantly and builds one on a fresh install,
showing a spinner only while that happens. A failure is remembered and
shown as a clear banner on the policy-dependent pages (AI Assistant and
Policy Intelligence); Home, Customer Explorer and Transaction Analytics
keep working. Policy searches still retry the build, so the banner
disappears once the index becomes available.

Set BANKMATE_INDEX_AUTOINIT=0 to skip startup initialisation (tests).
"""

import logging
import os

import streamlit as st


logger = logging.getLogger(__name__)

_status = {"ok": None, "error": None, "chunks": 0, "built": False}

INDEX_ERROR_MESSAGE = (
    "BankMate's policy index isn't available, so policy search and policy "
    "answers are disabled. Customer and transaction features still work."
)


def _enabled():
    return os.getenv("BANKMATE_INDEX_AUTOINIT", "1").strip() != "0"


@st.cache_resource(show_spinner=False)
def _initialize_once():
    from app.vector_store import ensure_policy_index

    try:
        status = ensure_policy_index()
        return {"ok": True, "error": None, "chunks": status.count, "built": status.built}
    except Exception as error:  # noqa: BLE001 - reported in the UI and logs
        logger.error("Policy index initialisation failed: %s", error)
        return {"ok": False, "error": str(error) or error.__class__.__name__,
                "chunks": 0, "built": False}


def initialize():
    """Prepare the policy index once per process. Never raises."""

    if not _enabled():
        return None

    from app.vector_store import is_index_ready

    if is_index_ready():
        _status.update(ok=True, error=None)
        return dict(_status)

    with st.spinner("Preparing the policy index… this only happens on first launch."):
        _status.update(_initialize_once())

    return dict(_status)


def index_error():
    """The startup error if the index is still unavailable, else None."""

    if _status.get("ok") is not False:
        return None

    from app.vector_store import is_index_ready

    if is_index_ready():          # a later search rebuilt it successfully
        _status.update(ok=True, error=None)
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
