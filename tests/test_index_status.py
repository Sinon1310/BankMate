"""
Policy index readiness in the web app (ui/index_status.py).

Regression: a failed startup attempt used to be cached (st.cache_resource)
for the life of the server process, so a transient problem kept the
"policy index isn't available" banner up until a policy search happened
to rebuild the index, and the log carried only the message.
"""

import logging

import pytest

import app.vector_store as vector_store
from ui import index_status


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(index_status, "_status", dict(index_status._status, ok=None, error=None, attempted_at=None))
    monkeypatch.setattr(vector_store, "is_index_ready", lambda persist_dir=None: False)
    monkeypatch.setenv("BANKMATE_INDEX_AUTOINIT", "1")


class Flaky:
    """ensure_policy_index stand-in: fails `failures` times, then succeeds."""

    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise vector_store.PolicyIndexError("database is locked")
        return vector_store.IndexStatus(state="valid", count=19)


def test_transient_startup_failure_recovers_without_a_restart(fresh, monkeypatch):
    flaky = Flaky(failures=1)
    monkeypatch.setattr(vector_store, "ensure_policy_index", flaky)
    monkeypatch.setattr(index_status, "RETRY_SECONDS", 0)

    index_status.initialize()
    assert index_status._status["ok"] is False and index_status._status["error"] == "database is locked"

    assert index_status.index_error() is None                # the next page load retries and succeeds
    assert index_status._status["ok"] is True and index_status._status["chunks"] == 19
    assert flaky.calls == 2


def test_retries_are_rate_limited_while_the_failure_persists(fresh, monkeypatch):
    flaky = Flaky(failures=99)
    monkeypatch.setattr(vector_store, "ensure_policy_index", flaky)
    monkeypatch.setattr(index_status, "RETRY_SECONDS", 30)

    index_status.initialize()
    for _ in range(5):
        assert index_status.index_error() == "database is locked"
        index_status.initialize()
    assert flaky.calls == 1                                  # no retry storm within the cooldown

    index_status._initialize_once.clear()                    # cooldown forgotten: next load retries
    index_status.index_error()
    assert flaky.calls == 2


def test_failures_are_logged_with_the_full_traceback(fresh, monkeypatch, caplog):
    monkeypatch.setattr(vector_store, "ensure_policy_index", Flaky(failures=1))
    with caplog.at_level(logging.ERROR, logger="ui.index_status"):
        index_status.initialize()
    record = next(r for r in caplog.records if "initialisation failed" in r.getMessage())
    assert record.exc_info and record.exc_info[0] is vector_store.PolicyIndexError


def test_no_automatic_retries_when_autoinit_is_off(fresh, monkeypatch):
    flaky = Flaky(failures=0)
    monkeypatch.setattr(vector_store, "ensure_policy_index", flaky)
    monkeypatch.setenv("BANKMATE_INDEX_AUTOINIT", "0")
    index_status._status.update(ok=False, error="startup failed")

    assert index_status.initialize() is None
    assert index_status.index_error() == "startup failed"
    assert flaky.calls == 0                                  # tests never touch the real index


def test_a_valid_index_is_reused_without_retrying(fresh, monkeypatch):
    flaky = Flaky(failures=0)
    monkeypatch.setattr(vector_store, "ensure_policy_index", flaky)
    monkeypatch.setattr(vector_store, "is_index_ready", lambda persist_dir=None: True)

    assert index_status.initialize()["ok"] is True and index_status.index_error() is None
    assert flaky.calls == 0
