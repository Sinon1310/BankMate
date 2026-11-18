"""
UI tests using Streamlit's AppTest harness (Stages 1-2).

The assistant backend is replaced with a fake that records every call, so
no Groq request is made and ChromaDB is never touched. Assertions focus
on behaviour: which question reaches the backend, with which session_id,
and what the user sees.
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.assistant as assistant_module
from ui import navigation, state


APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
ASSISTANT = navigation.ASSISTANT["path"]

REFUND_CITATION = {
    "policy_id": "POL-002",
    "policy_name": "TRANSACTION REFUND POLICY",
    "version": "1.0",
    "source_files": ["refund_policy.txt"],
}


class FakeBackend:
    """Stands in for ask_bankmate_with_sources; records calls."""

    def __init__(self):
        self.calls = []
        self.fail_next = 0
        self.sources = [REFUND_CITATION]

    def __call__(self, question, session_id="default", model=None, tools=None):
        self.calls.append({"question": question, "session_id": session_id})

        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("backend down")

        return {"answer": f"Answer to: {question}", "sources": list(self.sources)}

    def questions(self):
        return [call["question"] for call in self.calls]

    def sessions(self):
        return [call["session_id"] for call in self.calls]


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(assistant_module, "ask_bankmate_with_sources", fake)
    return fake


def run_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def open_assistant():
    at = run_app()
    at.switch_page(ASSISTANT).run()
    assert not at.exception, at.exception
    return at


def send(at, text):
    at.chat_input(key="assistant-input").set_value(text).run()
    assert not at.exception, at.exception
    return at


def page_text(at):
    """Rendered markdown, excluding the theme's <style> block."""

    return "\n".join(
        block.value for block in at.markdown
        if not block.value.lstrip().startswith("<style>")
    )


def conversations(at):
    return list(at.session_state[state.CONVERSATIONS].values())


def active(at):
    return at.session_state[state.CONVERSATIONS][at.session_state[state.ACTIVE_ID]]


# ---------- Home (Stage 1) ----------

def test_home_renders_brand_and_live_workspace_stats():
    at = run_app()
    text = page_text(at)

    assert "Banking answers" in text
    # Values come from data/*.csv and knowledge/policies, not hardcoded.
    assert '>3</div><div class="bm-stat-label">Customers' in text
    assert '>10</div><div class="bm-stat-label">Transactions' in text
    assert '>3</div><div class="bm-stat-label">Policy documents' in text
    assert "1–5 Nov 2026" in text


def test_only_implemented_pages_are_registered():
    paths = [page["path"] for page in navigation.PAGES]
    root = Path(APP).parent

    assert paths == ["ui/views/home.py", "ui/views/assistant.py"]
    for path in paths:
        assert (root / path).is_file()


def test_every_home_button_has_a_real_action():
    at = run_app()
    # 4 suggestion chips + 3 capability actions; nothing decorative.
    assert len(at.main.button) == 7


def test_suggestion_chip_opens_assistant_and_sends_prompt(backend):
    at = run_app()
    at.button(key="chip-btn-refund").click().run()

    assert not at.exception, at.exception
    assert backend.questions() == ["How are refunds handled for duplicate transactions?"]
    assert "Answer to: How are refunds handled" in page_text(at)


def test_capability_card_opens_assistant_with_its_prompt(backend):
    at = run_app()
    at.button(key="card-btn-metrics").click().run()

    assert backend.questions() == ["What are C101's transaction metrics?"]


def test_home_composer_sends_typed_question(backend):
    at = run_app()
    at.chat_input(key="home-composer-input").set_value("Show C103's profile").run()

    assert not at.exception, at.exception
    assert backend.questions() == ["Show C103's profile"]


# ---------- Assistant: core chat ----------

def test_empty_state_offers_working_suggestions(backend):
    at = open_assistant()

    assert "How can I help?" in page_text(at)
    at.button(key="chip-btn-empty-refund").click().run()

    assert backend.questions() == ["What happens when a refund is approved?"]
    assert "How can I help?" not in page_text(at)


def test_follow_ups_share_one_session(backend):
    at = open_assistant()
    send(at, "Show me C101")
    send(at, "What was his total debit?")

    assert backend.questions() == ["Show me C101", "What was his total debit?"]
    assert backend.sessions()[0] == backend.sessions()[1]
    assert backend.sessions()[0] == active(at)["session_id"]


def test_reply_and_citations_are_rendered_from_backend(backend):
    at = open_assistant()
    send(at, "Refund question")
    text = page_text(at)

    assert "Answer to: Refund question" in text
    assert "POL-002" in text
    assert "Transaction Refund Policy" in text
    assert "Version 1.0" in text and "refund_policy.txt" in text
    # The fictional-policy disclaimer travels with every sources block.
    assert "fictional BankMate training policies" in text


def test_no_sources_block_when_backend_returns_none(backend):
    backend.sources = []
    at = open_assistant()
    send(at, "Show me C101")

    assert "Answer to: Show me C101" in page_text(at)
    assert "bm-sources" not in page_text(at)


def test_ui_never_adds_citations_the_backend_did_not_return(backend):
    at = open_assistant()
    send(at, "Refund question")
    backend.sources = []
    send(at, "Customer question")

    stored = active(at)["messages"]
    assert stored[1]["sources"] == [REFUND_CITATION]
    assert stored[3]["sources"] == []


def test_user_text_is_not_rendered_as_html(backend):
    at = open_assistant()
    send(at, "<b>bold</b> question")

    header = next(
        b.value for b in at.markdown
        if 'class="bm-thread-title"' in b.value
    )
    assert "&lt;b&gt;bold&lt;/b&gt;" in header


# ---------- Assistant: conversations ----------

def test_conversation_title_comes_from_first_question(backend):
    at = open_assistant()
    send(at, "What happens when a refund is approved for a duplicate card payment?")
    send(at, "And how long does it take?")

    title = active(at)["title"]
    assert title.startswith("What happens when a refund is approved")
    assert title.endswith("…") and len(title) <= state.TITLE_LENGTH


def test_new_chat_starts_isolated_session_and_keeps_history(backend):
    at = open_assistant()
    send(at, "First conversation question")
    first_id = active(at)["id"]

    at.button(key="assistant-new-chat").click().run()
    assert "First conversation question" not in page_text(at)
    send(at, "Second conversation question")

    assert backend.sessions()[0] != backend.sessions()[1]
    assert len(conversations(at)) == 2
    assert first_id in at.session_state[state.CONVERSATIONS]


def test_switching_conversations_restores_thread_and_session(backend):
    at = open_assistant()
    send(at, "Question in A")
    conversation_a = active(at)["id"]

    at.button(key="assistant-new-chat").click().run()
    send(at, "Question in B")

    at.button(key=f"conv-btn-{conversation_a}").click().run()
    assert "Question in A" in page_text(at)
    assert "Question in B" not in page_text(at)

    send(at, "Follow-up in A")
    assert backend.sessions()[2] == backend.sessions()[0]
    assert backend.sessions()[2] != backend.sessions()[1]


def test_new_chat_reuses_an_empty_conversation(backend):
    at = open_assistant()
    at.button(key="assistant-new-chat").click().run()
    at.button(key="assistant-new-chat").click().run()

    assert len(conversations(at)) == 1


def test_home_prompt_starts_new_conversation_instead_of_appending(backend):
    at = open_assistant()
    send(at, "Existing thread question")
    existing = active(at)["id"]

    at.switch_page(navigation.HOME["path"]).run()
    at.button(key="chip-btn-dispute").click().run()

    assert active(at)["id"] != existing
    assert len(active(at)["messages"]) == 2
    assert len(at.session_state[state.CONVERSATIONS][existing]["messages"]) == 2


# ---------- Assistant: failure states ----------

def test_error_state_and_retry_resends_same_question(backend):
    backend.fail_next = 1
    at = open_assistant()
    send(at, "Flaky question")

    messages = active(at)["messages"]
    assert messages[-1].get("error") is True
    assert "Something went wrong" in page_text(at)

    at.button(key="assistant-retry").click().run()
    assert not at.exception, at.exception

    messages = active(at)["messages"]
    assert backend.questions() == ["Flaky question", "Flaky question"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert not messages[-1].get("error")
    assert backend.sessions()[0] == backend.sessions()[1]


def test_retry_is_only_offered_after_a_failure(backend):
    at = open_assistant()
    send(at, "Fine question")

    assert not [b for b in at.button if b.key == "assistant-retry"]


def test_unavailable_backend_disables_input_without_dead_controls(monkeypatch):
    # Simulate app.assistant failing to import (e.g. missing GROQ_API_KEY).
    monkeypatch.setitem(sys.modules, "app.assistant", None)

    at = open_assistant()

    assert "Assistant unavailable" in page_text(at)
    assert at.chat_input(key="assistant-input").disabled
    assert not [b for b in at.main.button if b.key and b.key.startswith("chip-btn-empty")]
