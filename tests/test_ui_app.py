"""
UI tests using Streamlit's AppTest harness (Stages 1-2).

The assistant backend is replaced with a fake that records every call, so
no Groq request is made and ChromaDB is never touched. Assertions focus
on behaviour: which question reaches the backend, with which session_id,
and what the user sees.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.assistant as assistant_module
import app.tools as tools_module
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
        self.remembered = set()

    def __call__(self, question, session_id="default", model=None, tools=None, request_id=None):
        self.calls.append({"question": question, "session_id": session_id, "request_id": request_id})

        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("backend down")

        # Like the real backend: a successful answer means it now holds
        # memory of this conversation.
        self.remembered.add(session_id)
        return {"answer": f"Answer to: {question}", "sources": list(self.sources)}

    def has_memory(self, session_id):
        return session_id in self.remembered

    def questions(self):
        return [call["question"] for call in self.calls]

    def sessions(self):
        return [call["session_id"] for call in self.calls]


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(assistant_module, "ask_bankmate_with_sources", fake)
    monkeypatch.setattr(assistant_module, "has_conversation_memory", fake.has_memory)
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

    assert paths == [
        "ui/views/home.py",
        "ui/views/assistant.py",
        "ui/views/customers.py",
        "ui/views/analytics.py",
        "ui/views/policies.py",
    ]
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


# ---------- Customer Explorer (Stage 3) ----------

CUSTOMERS = navigation.CUSTOMERS["path"]
ANALYTICS = navigation.ANALYTICS["path"]


def open_page(path):
    at = run_app()
    at.switch_page(path).run()
    assert not at.exception, at.exception
    return at


def test_explorer_defaults_to_first_customer_with_tool_metrics():
    at = open_page(CUSTOMERS)
    text = page_text(at)

    assert "Arjun Mehta" in text
    # C101 per get_customer_metrics: debit 9000, credit 25000, net 16000, 5 rows.
    assert "₹9,000" in text and "₹25,000" in text and "+₹16,000" in text
    assert "not an account balance" in text
    assert text.count("<tr>") == 1 + 5  # header + 5 transactions


def test_explorer_selecting_customer_updates_detail():
    at = open_page(CUSTOMERS)
    at.button(key="cust-btn-C103").click().run()
    text = page_text(at)

    assert not at.exception, at.exception
    assert "Rahul Nair" in text and "₹3,500" in text and "₹40,000" in text
    assert at.query_params.get("customer") in ("C103", ["C103"])


def test_explorer_search_filters_list():
    at = open_page(CUSTOMERS)
    at.text_input(key="explorer-search").set_value("pune").run()

    customer_buttons = [b.key for b in at.button if b.key and b.key.startswith("cust-btn-")]
    assert customer_buttons == ["cust-btn-C102"]

    at.text_input(key="explorer-search").set_value("zzz").run()
    assert "No customers match" in page_text(at)


def test_explorer_ask_button_opens_assistant_for_that_customer(backend):
    at = open_page(CUSTOMERS)
    at.button(key="cust-btn-C102").click().run()
    at.button(key="explorer-ask").click().run()

    assert not at.exception, at.exception
    assert len(backend.calls) == 1
    assert "C102" in backend.questions()[0]


def test_explorer_works_without_assistant_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "app.assistant", None)
    at = open_page(CUSTOMERS)
    assert "Arjun Mehta" in page_text(at)


# ---------- Transaction Analytics (Stage 3) ----------

def open_analytics():
    """
    Open Analytics and normalise the Type segmented control.

    Workaround for Streamlit 1.50 AppTest: an untouched single-select
    segmented control keeps its default as a plain string, which AppTest
    iterates character by character on the next rerun. Setting it as a
    one-item list (as AppTest expects) avoids that; it must be done again
    before each rerun. App code is unaffected.
    """

    at = open_page(ANALYTICS)
    set_type(at, "All", run=False)
    return at


def set_type(at, value, run=True):
    control = next(g for g in at.get("button_group") if g.key == "analytics-type")
    control.set_value([value])
    return control.run() if run else at

def test_analytics_defaults_cover_all_transactions():
    at = open_page(ANALYTICS)
    text = page_text(at)

    assert "₹14,000" in text and "₹95,000" in text and "+₹81,000" in text
    assert "not an account balance" in text
    assert len(at.get("arrow_vega_lite_chart")) == 3
    assert text.count("<tr>") == 1 + 10


def test_analytics_type_filter_scopes_kpis_charts_and_table():
    at = open_analytics()
    set_type(at, "Credit")
    text = page_text(at)

    assert not at.exception, at.exception
    assert "₹0" in text and "₹95,000" in text
    assert "No debits" in text  # merchant chart empty state
    assert len(at.get("arrow_vega_lite_chart")) == 2
    assert text.count("<tr>") == 1 + 3


def test_analytics_customer_filter():
    at = open_analytics()
    at.multiselect(key="analytics-customers").set_value(["C101"]).run()
    text = page_text(at)

    assert "₹9,000" in text and "₹25,000" in text and "+₹16,000" in text
    assert text.count("<tr>") == 1 + 5


def test_analytics_empty_selection_shows_empty_state():
    at = open_analytics()
    at.multiselect(key="analytics-customers").set_value(["C102"]).run()
    # Set the date, then the type (which reruns) — AppTest needs the
    # segmented control re-normalised before every rerun.
    at.date_input(key="analytics-dates").set_value((date(2026, 11, 4), date(2026, 11, 5)))
    set_type(at, "Credit")

    assert not at.exception, at.exception
    assert "No transactions match these filters" in page_text(at)
    assert not at.get("arrow_vega_lite_chart")


# ---------- Policy Intelligence (Stage 4) ----------

POLICIES = navigation.POLICIES["path"]


class FakeSearchTool:
    """Stands in for app.tools.search_banking_policies (no ChromaDB)."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def invoke(self, args):
        self.calls.append(args["question"])
        if self.error:
            raise self.error
        return json.dumps(self.result)


DISPUTE_EVIDENCE = {
    "status": "SUCCESS",
    "evidence": [{
        "policy_id": "POL-001",
        "policy_name": "CARD DISPUTE HANDLING POLICY",
        "version": "1.0",
        "section": "REQUIRED INFORMATION",
        "source_file": "card_dispute_policy.txt",
        "distance": 0.65,
        "content": "The customer should provide:\n1. Customer ID.\n2. Card or account reference.",
    }],
}


@pytest.fixture
def search_tool(monkeypatch):
    tool = FakeSearchTool(DISPUTE_EVIDENCE)
    monkeypatch.setattr(tools_module, "search_banking_policies", tool)
    return tool


def open_policy(policy_id):
    at = run_app()
    at.query_params["policy"] = policy_id
    at.switch_page(POLICIES).run()
    assert not at.exception, at.exception
    return at


def test_policy_library_lists_documents_from_files():
    at = open_page(POLICIES)
    text = page_text(at)

    for policy_id, name in (
        ("POL-001", "Card Dispute Handling Policy"),
        ("POL-002", "Transaction Refund Policy"),
        ("POL-003", "Transaction Security Guidelines"),
    ):
        assert policy_id in text and name in text
    assert "3 policies" in text
    assert "card_dispute_policy.txt" in text


def test_policy_filter_and_scope():
    at = open_page(POLICIES)
    at.text_input(key="policy-filter").set_value("otp").run()

    assert "1 policy" in page_text(at)
    assert [b.key for b in at.button if b.key and b.key.startswith("policy-open-")] == ["policy-open-POL-003"]
    assert "Customer Protection" in page_text(at)  # matched section snippet

    at.selectbox(key="policy-scope").set_value("Titles & IDs").run()
    assert "No policies match" in page_text(at)


def test_read_policy_opens_detail_and_back_returns():
    at = open_page(POLICIES)
    at.button(key="policy-open-POL-001").click().run()
    text = page_text(at)

    assert not at.exception, at.exception
    assert 'class="bm-policy-title">Card Dispute Handling Policy' in text
    for section in ("Dispute Initiation", "Required Information", "Status Tracking", "Resolution"):
        assert f"<h2>{section}</h2>" in text
    assert "<li>Merchant name.</li>" in text
    assert "Fictional policy" in text
    assert "DOCUMENT ID" not in text  # header shown as metadata, not content

    at.button(key="policy-back").click().run()
    assert "3 policies" in page_text(at)


def test_deep_link_opens_policy_and_unknown_id_falls_back():
    at = open_policy("POL-003")
    assert 'class="bm-policy-title">Transaction Security Guidelines' in page_text(at)

    at = open_policy("POL-999")
    assert any("No policy matches POL-999" in w.value for w in at.warning)
    assert "3 policies" in page_text(at)


def test_policy_suggestion_opens_assistant(backend):
    at = open_policy("POL-002")
    at.button(key="chip-btn-policy-q-0").click().run()

    assert not at.exception, at.exception
    assert backend.questions() == ["Summarise the Transaction Refund Policy."]


def test_policy_question_box_includes_policy_context(backend):
    at = open_policy("POL-001")
    at.chat_input(key="policy-ask-input").set_value("How is status tracked?").run()

    assert backend.questions() == [
        "Using the Card Dispute Handling Policy (POL-001): How is status tracked?"
    ]


def test_policy_card_ask_button(backend):
    at = open_page(POLICIES)
    at.button(key="policy-ask-POL-003").click().run()

    assert backend.questions() == ["Summarise the Transaction Security Guidelines (POL-003)."]


def test_index_search_shows_passages_and_hands_off(search_tool, backend):
    at = open_page(POLICIES)
    at.text_input(key="policy-retrieval-input").set_value("What is needed to raise a dispute?").run()
    text = page_text(at)

    assert search_tool.calls == ["What is needed to raise a dispute?"]
    assert "1 passage" in text and "Required Information" in text and "distance 0.65" in text

    at.button(key="passage-ask").click().run()
    assert backend.questions() == ["What is needed to raise a dispute?"]


def test_index_search_result_opens_policy(search_tool):
    at = open_page(POLICIES)
    at.text_input(key="policy-retrieval-input").set_value("dispute details").run()
    at.button(key="passage-open-0").click().run()

    assert 'class="bm-policy-title">Card Dispute Handling Policy' in page_text(at)


def test_index_search_reruns_do_not_repeat_retrieval(search_tool):
    at = open_page(POLICIES)
    at.text_input(key="policy-retrieval-input").set_value("dispute details").run()
    at.text_input(key="policy-filter").set_value("refund").run()

    assert search_tool.calls == ["dispute details"]


def test_index_search_no_results_state(monkeypatch):
    monkeypatch.setattr(
        tools_module, "search_banking_policies",
        FakeSearchTool({"status": "NO_RELEVANT_POLICY_FOUND", "evidence": []}),
    )
    at = open_page(POLICIES)
    at.text_input(key="policy-retrieval-input").set_value("home loan rates").run()

    assert "No sufficiently relevant passages" in page_text(at)
    assert not [b for b in at.button if b.key == "passage-ask"]


def test_index_search_error_state(monkeypatch):
    monkeypatch.setattr(
        tools_module, "search_banking_policies",
        FakeSearchTool(error=RuntimeError("collection missing")),
    )
    at = open_page(POLICIES)
    at.text_input(key="policy-retrieval-input").set_value("refunds").run()

    assert not at.exception, at.exception
    assert any("could not be searched" in e.value for e in at.error)
    # The raw failure detail goes to the server log, never to the page.
    assert not any("collection missing" in e.value for e in at.error)
    assert "collection missing" not in page_text(at)


def test_assistant_source_opens_policy_and_conversation_survives(backend):
    at = open_assistant()
    send(at, "Refund question")

    at.button(key="src-open-1-POL-002").click().run()
    assert not at.exception, at.exception
    assert 'class="bm-policy-title">Transaction Refund Policy' in page_text(at)

    at.switch_page(ASSISTANT).run()
    assert "Answer to: Refund question" in page_text(at)


def test_policy_pages_work_without_assistant_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "app.assistant", None)
    at = open_policy("POL-002")
    assert 'class="bm-policy-title">Transaction Refund Policy' in page_text(at)


# ---------- Stage 5.2: friendly provider errors in the UI ----------

def test_rate_limit_shows_friendly_message_and_retry(monkeypatch):
    from app.errors import RATE_LIMITED, AssistantError

    def limited(*args, **kwargs):
        raise AssistantError(
            RATE_LIMITED,
            "Rate limit reached for model on tokens per minute (TPM): Limit 8000",
            retry_after=13,
        )

    monkeypatch.setattr(assistant_module, "ask_bankmate_with_sources", limited)
    at = open_assistant()
    send(at, "Hello")
    text = page_text(at)

    assert "reached its usage limit" in text and "wait a minute, then try again" in text
    assert "13 seconds" not in text and "1 seconds" not in text
    assert "TPM" not in text and "8000" not in text
    assert [b for b in at.button if b.key == "assistant-retry"]


# ---------- Stage 5.2.1: request IDs, interrupted runs, expired memory ----------

def test_each_message_gets_a_stable_request_id_reused_by_retry(backend):
    backend.fail_next = 1
    at = open_assistant()
    send(at, "Flaky question")
    at.button(key="assistant-retry").click().run()

    ids = [call["request_id"] for call in backend.calls]
    assert len(ids) == 2 and ids[0] and ids[0] == ids[1]

    send(at, "Another question")
    assert backend.calls[-1]["request_id"] not in (None, ids[0])


def test_interrupted_run_resend_returns_saved_answer_without_model_call(monkeypatch):
    """Real engine, fake model: the UI re-sends the same request ID."""

    from langchain_core.messages import AIMessage, HumanMessage

    class CountingModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content=f"Answer #{self.calls}")

    model = CountingModel()
    monkeypatch.setattr(assistant_module, "_get_model", lambda: model)

    at = open_assistant()
    send(at, "What was his total debit?")
    conversation = active(at)
    session_id = conversation["session_id"]

    try:
        assert model.calls == 1
        assert "Answer #1" in page_text(at)

        # Simulate a run stopped after the backend saved the answer but
        # before the UI recorded it: the reply is missing from UI state.
        conversation["messages"].pop()
        at.run()

        assert not at.exception, at.exception
        assert model.calls == 1                      # saved result returned
        assert "Answer #1" in page_text(at)
        stored = assistant_module.conversation_sessions[session_id]
        assert [m.content for m in stored if isinstance(m, HumanMessage)] == ["What was his total debit?"]
    finally:
        for store in (assistant_module.conversation_sessions, assistant_module.grounding_context,
                      assistant_module.session_last_active, assistant_module.conversation_versions,
                      assistant_module.request_results, assistant_module.requests_in_flight):
            store.pop(session_id, None)


def test_expired_memory_shows_notice_and_does_not_resend(backend):
    at = open_assistant()
    send(at, "First question")
    calls_before = len(backend.calls)

    backend.remembered.clear()          # backend memory expired
    at.run()
    text = page_text(at)

    assert "Earlier context has expired" in text
    assert len(backend.calls) == calls_before        # nothing reprocessed
    assert [b for b in at.button if b.key == "assistant-fresh-start"]
    assert active(at)["memory_lost_at"] == 2

    # Continuing works, and the notice stays where context ended.
    send(at, "New question")
    assert backend.calls[-1]["question"] == "New question"
    assert "Earlier context has expired" in page_text(at)
    assert active(at)["memory_lost_at"] == 2


def test_expired_memory_fresh_start_opens_new_conversation(backend):
    at = open_assistant()
    send(at, "First question")
    old_id = active(at)["id"]

    backend.remembered.clear()
    at.run()
    at.button(key="assistant-fresh-start").click().run()

    assert active(at)["id"] != old_id
    assert "How can I help?" in page_text(at)


def test_unanswered_question_is_not_silently_resent_after_expiry(backend):
    at = open_assistant()
    send(at, "Answered question")
    # An interrupted, never-answered question, then the memory expires.
    active(at)["messages"].append({"role": "user", "content": "Lost question", "request_id": "lost-1"})
    backend.remembered.clear()
    calls_before = len(backend.calls)
    at.run()

    assert len(backend.calls) == calls_before
    assert "wasn't answered before BankMate's memory" in page_text(at)
    assert [b for b in at.button if b.key == "assistant-retry"]
    assert active(at)["memory_lost_at"] == 2           # before the lost question

    at.button(key="assistant-retry").click().run()     # explicit user action
    assert backend.calls[-1]["question"] == "Lost question"

    # The notice stays above the re-asked question, not below its answer.
    texts = [b.value for b in at.markdown if not b.value.lstrip().startswith("<style>")]
    notice = next(i for i, t in enumerate(texts) if "Earlier context has expired" in t)
    lost_q = next(i for i, t in enumerate(texts) if t == "Lost question")
    answer = next(i for i, t in enumerate(texts) if t == "Answer to: Lost question")
    assert notice < lost_q < answer


# ---------- Stage 5.3: policy index readiness in the UI ----------

from ui import index_status  # noqa: E402


@pytest.fixture
def failed_index(monkeypatch):
    import app.vector_store as vector_store

    monkeypatch.setitem(index_status._status, "ok", False)
    monkeypatch.setitem(index_status._status, "error", "Building the policy index failed: disk full")
    monkeypatch.setattr(vector_store, "is_index_ready", lambda persist_dir=None: False)


def error_texts(at):
    return [e.value for e in at.error]


def test_failed_index_banner_on_policy_pages_only(failed_index):
    assistant_page = open_assistant()
    assert any("policy index isn't available" in e for e in error_texts(assistant_page))

    policies_page = open_page(navigation.POLICIES["path"])
    assert any("policy index isn't available" in e for e in error_texts(policies_page))
    assert "Card Dispute Handling Policy" in page_text(policies_page)    # library still works

    for path in (navigation.HOME["path"], CUSTOMERS, ANALYTICS):
        page = open_page(path)
        assert not error_texts(page), path


def test_banner_never_shows_raw_build_error(failed_index):
    at = open_assistant()
    banner = next(e for e in error_texts(at) if "policy index" in e)
    assert "disk full" not in banner


def test_banner_clears_once_a_search_rebuilds_the_index(monkeypatch, failed_index):
    import app.vector_store as vector_store

    monkeypatch.setattr(vector_store, "is_index_ready", lambda persist_dir=None: True)
    at = open_assistant()
    assert not [e for e in error_texts(at) if "policy index" in e]


def test_startup_initialisation_failure_is_recorded_not_raised(monkeypatch):
    import app.vector_store as vector_store

    def broken(*args, **kwargs):
        raise vector_store.PolicyIndexError("cannot build")

    monkeypatch.setenv("BANKMATE_INDEX_AUTOINIT", "1")
    monkeypatch.setattr(vector_store, "ensure_policy_index", broken)
    monkeypatch.setattr(vector_store, "is_index_ready", lambda persist_dir=None: False)
    monkeypatch.setattr(index_status, "_status", dict(index_status._status))
    index_status._initialize_once.clear()

    try:
        at = run_app()                        # app starts despite the failure
        assert "Banking answers" in page_text(at)
        assert index_status._status["ok"] is False
        assert index_status._status["error"] == "cannot build"
    finally:
        index_status._initialize_once.clear()


# ---------- Stage 5.4: application security ----------

def test_malicious_policy_deep_link_is_not_rendered(monkeypatch):
    at = run_app()
    at.query_params["policy"] = "[Verify your account](https://evil.example/login)"
    at.switch_page(POLICIES).run()

    assert not at.exception, at.exception
    warnings = [w.value for w in at.warning]
    assert warnings and all("evil.example" not in w and "Verify your account" not in w for w in warnings)
    assert "3 policies" in page_text(at)


def test_unknown_but_well_formed_policy_id_is_named():
    at = open_policy("POL-123")
    assert any("No policy matches POL-123" in w.value for w in at.warning)


def test_malicious_customer_deep_link_falls_back_safely():
    at = run_app()
    at.query_params["customer"] = "../../etc/passwd"
    at.switch_page(CUSTOMERS).run()

    assert not at.exception, at.exception
    assert "Arjun Mehta" in page_text(at)                 # default customer
    assert "passwd" not in page_text(at)


def test_chat_inputs_enforce_length_limits():
    from app.security import MAX_QUESTION_CHARS

    home = run_app()
    assert home.chat_input(key="home-composer-input").proto.max_chars == MAX_QUESTION_CHARS

    chat = open_assistant()
    assert chat.chat_input(key="assistant-input").proto.max_chars == MAX_QUESTION_CHARS

    detail = open_policy("POL-001")
    assert 0 < detail.chat_input(key="policy-ask-input").proto.max_chars <= MAX_QUESTION_CHARS - 100


def test_backend_load_failure_shows_generic_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "app.assistant", None)     # ImportError, not a config error
    at = open_assistant()
    text = page_text(at)

    assert "Assistant unavailable" in text
    assert "could not be started" in text
    assert "sys.modules" not in text and "Traceback" not in text


def test_policy_load_failure_hides_internal_paths(monkeypatch):
    import streamlit as st
    from ui import policy_data

    def broken():
        raise OSError("Permission denied: '/Users/someone/secret/knowledge/policies'")

    monkeypatch.setattr(policy_data, "load_catalog", broken)
    st.cache_data.clear()
    try:
        at = open_page(POLICIES)
        errors_shown = " ".join(e.value for e in at.error)
        assert "could not be loaded" in errors_shown
        assert "/Users/someone" not in errors_shown and "Permission denied" not in errors_shown
    finally:
        st.cache_data.clear()
