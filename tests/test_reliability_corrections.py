"""
Stage 5.2.1 regression tests.

Part A: grounding scope follows the kind of each turn (policy,
customer-data, no-tool) and only the relevant policy turn's evidence.
Part B: request idempotency, in-flight duplicates, version conflicts and
isolation, with deterministic thread coordination (Events, no sleeps).

Mocked models and tools only; no Groq calls, no ChromaDB.
"""

import json
import threading
import time
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import assistant, errors
from app.history import is_complete_history


NOT_SPECIFIED_DESTINATION = "The policy does not specify where the refunded amount is credited."
NOT_SPECIFIED_TIMEFRAME = "The policy does not specify a timeframe."


def policy_result(policy_id, text):
    return json.dumps({
        "status": "SUCCESS",
        "sources": [{"policy_id": policy_id, "policy_name": "POLICY " + policy_id, "version": "1.0",
                     "section": "SECTION", "source_file": policy_id.lower() + ".txt"}],
        "evidence": [{"policy_id": policy_id, "section": "SECTION", "content": text}],
    })


POL_002 = policy_result(
    "POL-002",
    "Once approved, the refund should be processed through the appropriate "
    "payment processing workflow.",
)
POL_X_WITH_TIMEFRAME = policy_result("POL-901", "Disputes are resolved within 10 days.")
TRANSACTIONS = json.dumps([
    {"transaction_id": "T001", "customer_id": "C101", "transaction_date": "2026-11-01",
     "transaction_type": "Debit", "amount": "2500", "merchant": "Amazon"},
    {"transaction_id": "T003", "customer_id": "C101", "transaction_date": "2026-11-03",
     "transaction_type": "Credit", "amount": "25000", "merchant": "Salary"},
    {"transaction_id": "T005", "customer_id": "C101", "transaction_date": "2026-11-05",
     "transaction_type": "Debit", "amount": "800", "merchant": "Uber"},
])


class FakeModel:
    """Scripted model. An item may be a callable run before returning."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        item = self.responses.pop(0)
        if callable(item):
            item = item()
        if isinstance(item, Exception):
            raise item
        return item


class FakeTool:
    def __init__(self, result):
        self.result = result

    def invoke(self, args):
        return self.result


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


TOOLS = {
    "search_banking_policies": FakeTool(POL_002),
    "get_customer_transactions": FakeTool(TRANSACTIONS),
}


@pytest.fixture
def sid():
    session_id = "fix-" + uuid.uuid4().hex
    yield session_id
    clear(session_id)


def clear(session_id):
    for store in (assistant.conversation_sessions, assistant.grounding_context,
                  assistant.session_last_active, assistant.conversation_versions,
                  assistant.request_results, assistant.requests_in_flight):
        store.pop(session_id, None)


def ask(session_id, responses, tools=TOOLS, question="question", request_id=None):
    model = FakeModel(responses)
    result = assistant.ask_bankmate_with_sources(
        question, session_id=session_id, model=model, tools=tools, request_id=request_id,
    )
    return result, model


def policy_turn(session_id, tools=TOOLS, answer="It is processed through the workflow.\nCITED_POLICIES: POL-002"):
    return ask(session_id, [tool_call("search_banking_policies", {"question": "refund"}, "p-" + uuid.uuid4().hex[:6]),
                            AIMessage(content=answer)], tools=tools, question="refund?")


def questions(session_id):
    return [m.content for m in assistant.conversation_sessions[session_id] if isinstance(m, HumanMessage)]


# =====================================================================
# Part A — grounding scope
# =====================================================================

def test_customer_transaction_question_after_policy_question_is_not_policy_checked(sid):
    policy_turn(sid)

    answer = "C101's salary of INR 25,000 was credited to his account on 3 Nov, over 5 days of activity."
    result, model = ask(sid, [
        tool_call("get_customer_transactions", {"customer_id": "C101"}, "t1"),
        AIMessage(content=answer),
    ], question="Show C101's transactions")

    assert model.calls == 2                 # no grounding revision request
    assert result["answer"] == answer


def test_customer_days_question_answered_from_memory_is_kept(sid):
    policy_turn(sid)
    ask(sid, [tool_call("get_customer_transactions", {"customer_id": "C101"}, "t1"),
              AIMessage(content="Three transactions.")], question="C101 transactions?")

    answer = "C101's transactions span 5 days, from 1 to 5 Nov."
    result, model = ask(sid, [AIMessage(content=answer)], question="Over how many days did he transact?")

    assert model.calls == 1
    assert result["answer"] == answer      # not replaced by "The policy does not specify a timeframe."


def test_salary_credited_to_account_from_memory_is_kept(sid):
    ask(sid, [tool_call("get_customer_transactions", {"customer_id": "C101"}, "t1"),
              AIMessage(content="Three transactions.")], question="C101 transactions?")
    policy_turn(sid)
    ask(sid, [tool_call("get_customer_transactions", {"customer_id": "C101"}, "t2"),
              AIMessage(content="Here are the credits.")], question="And his credits?")

    answer = "His salary of INR 25,000 was credited to his account."
    result, model = ask(sid, [AIMessage(content=answer)], question="Where did the salary go?")

    assert model.calls == 1
    assert result["answer"] == answer


def test_policy_follow_up_from_memory_uses_most_recent_policy_turn(sid):
    policy_turn(sid)
    ask(sid, [AIMessage(content="Hello!")], question="thanks")     # a no-tool turn in between

    result, model = ask(sid, [
        AIMessage(content="It is credited back to your card within 5 days."),
        AIMessage(content="It is credited back to your card within 5 days."),   # ignores revision
    ], question="Where does the money go and how fast?")

    assert model.calls == 2                                       # revision was requested
    assert NOT_SPECIFIED_DESTINATION in result["answer"]
    assert "credited back" not in result["answer"]


def test_supported_policy_follow_up_needs_no_revision(sid):
    policy_turn(sid)
    answer = "It is processed through the appropriate payment processing workflow."
    result, model = ask(sid, [AIMessage(content=answer)], question="Remind me what happens?")

    assert model.calls == 1
    assert result["answer"] == answer


def test_older_policy_evidence_cannot_validate_a_new_policy_claim(sid):
    # Turn 1 retrieves a policy that does state a timeframe.
    tools_x = {"search_banking_policies": FakeTool(POL_X_WITH_TIMEFRAME)}
    policy_turn(sid, tools=tools_x, answer="Disputes are resolved within 10 days.\nCITED_POLICIES: POL-901")

    # Turn 2 retrieves POL-002 (no timeframe) and claims one anyway.
    claim = "Refunds are processed within 10 days.\nCITED_POLICIES: POL-002"
    result, model = ask(sid, [
        tool_call("search_banking_policies", {"question": "refund timing"}, "p2"),
        AIMessage(content=claim),
        AIMessage(content=claim),      # model ignores the revision request
    ], question="How long do refunds take?")

    assert model.calls == 3            # flagged despite turn 1's "within 10 days"
    assert NOT_SPECIFIED_TIMEFRAME in result["answer"]
    assert "10 days" not in result["answer"]
    assert [s["policy_id"] for s in result["sources"]] == ["POL-002"]


def test_memory_follow_up_does_not_reuse_evidence_from_an_older_policy_turn(sid):
    tools_x = {"search_banking_policies": FakeTool(POL_X_WITH_TIMEFRAME)}
    policy_turn(sid, tools=tools_x, answer="Disputes are resolved within 10 days.\nCITED_POLICIES: POL-901")
    policy_turn(sid)                                  # most recent policy turn: POL-002

    result, model = ask(sid, [
        AIMessage(content="Refunds take 10 days."),
        AIMessage(content="Refunds take 10 days."),
    ], question="So how long for refunds?")

    assert model.calls == 2
    assert result["answer"] == NOT_SPECIFIED_TIMEFRAME


def test_turn_kinds_are_recorded(sid):
    policy_turn(sid)
    ask(sid, [tool_call("get_customer_transactions", {"customer_id": "C101"}, "t1"),
              AIMessage(content="ok")], question="tx")
    ask(sid, [AIMessage(content="hi")], question="hello")

    kinds = [t["kind"] for t in assistant.grounding_context[sid]["turns"]]
    assert kinds == [assistant.TURN_POLICY, assistant.TURN_CUSTOMER, assistant.TURN_NONE]


# =====================================================================
# Part B — idempotency, in-flight duplicates, conflicts, isolation
# =====================================================================

def test_repeated_request_id_returns_saved_result_without_model_call(sid):
    first, model_1 = ask(sid, [AIMessage(content="INR 9,000")], question="his total debit?", request_id="r1")
    model_2 = FakeModel([AssertionError("model must not be called")])
    second = assistant.ask_bankmate_with_sources(
        "his total debit?", session_id=sid, model=model_2, tools={}, request_id="r1",
    )

    assert second == first
    assert model_1.calls == 1 and model_2.calls == 0
    assert questions(sid) == ["his total debit?"]


def test_saved_result_is_a_copy(sid):
    first, _ = ask(sid, [AIMessage(content="answer")], request_id="r1")
    first["answer"] = "mutated by caller"
    again = assistant.ask_bankmate_with_sources("q", session_id=sid, model=FakeModel([]), tools={}, request_id="r1")
    assert again["answer"] == "answer"


def test_failed_request_is_not_saved_so_retry_runs_again(sid):
    with pytest.raises(errors.AssistantError):
        ask(sid, [TimeoutError("slow")], request_id="r1")

    result, model = ask(sid, [AIMessage(content="ok now")], request_id="r1")
    assert model.calls == 1 and result["answer"] == "ok now"
    assert questions(sid) == ["question"]


def test_duplicate_request_while_first_is_running_does_not_call_model(sid):
    started, release = threading.Event(), threading.Event()
    outcome = {}

    def slow_reply():
        started.set()
        release.wait(5)
        return AIMessage(content="first answer")

    def first():
        outcome["first"] = assistant.ask_bankmate_with_sources(
            "q", session_id=sid, model=FakeModel([slow_reply]), tools={}, request_id="r1")

    worker = threading.Thread(target=first)
    worker.start()
    assert started.wait(5)

    duplicate_model = FakeModel([AssertionError("must not be called")])
    with pytest.raises(errors.AssistantError) as caught:
        assistant.ask_bankmate_with_sources("q", session_id=sid, model=duplicate_model, tools={}, request_id="r1")

    release.set()
    worker.join(5)

    assert caught.value.kind == errors.IN_PROGRESS
    assert duplicate_model.calls == 0
    assert outcome["first"]["answer"] == "first answer"
    # Once finished, a repeat gets the saved answer.
    again = assistant.ask_bankmate_with_sources("q", session_id=sid, model=FakeModel([]), tools={}, request_id="r1")
    assert again["answer"] == "first answer"
    assert questions(sid) == ["q"]


def test_concurrent_requests_same_conversation_conflict_instead_of_losing_a_turn(sid):
    ask(sid, [AIMessage(content="base answer")], question="base", request_id="r0")
    snapshot_taken, b_committed = threading.Event(), threading.Event()
    outcome = {}

    def a_reply():
        snapshot_taken.set()          # A has its snapshot; let B finish first
        b_committed.wait(5)
        return AIMessage(content="answer A")

    def request_a():
        try:
            assistant.ask_bankmate_with_sources("Q-A", session_id=sid, model=FakeModel([a_reply]),
                                                tools={}, request_id="rA")
        except errors.AssistantError as error:
            outcome["a_error"] = error

    worker = threading.Thread(target=request_a)
    worker.start()
    assert snapshot_taken.wait(5)

    ask(sid, [AIMessage(content="answer B")], question="Q-B", request_id="rB")
    b_committed.set()
    worker.join(5)

    assert outcome["a_error"].kind == errors.CONFLICT
    assert "updated by another reply" in outcome["a_error"].user_message
    assert questions(sid) == ["base", "Q-B"]          # B kept, nothing lost or interleaved
    assert is_complete_history(assistant.conversation_sessions[sid])
    assert "rA" not in assistant.request_results[sid]

    # Retrying A (same request ID) now succeeds on top of B.
    result, model = ask(sid, [AIMessage(content="answer A")], question="Q-A", request_id="rA")
    assert model.calls == 1 and result["answer"] == "answer A"
    assert questions(sid) == ["base", "Q-B", "Q-A"]


def test_version_increments_and_stale_commit_is_refused(sid):
    ask(sid, [AIMessage(content="one")])
    assert assistant.conversation_versions[sid] == 1

    def bump_version_then_reply():
        assistant.conversation_versions[sid] += 1     # another commit "happened"
        return AIMessage(content="stale")

    with pytest.raises(errors.AssistantError) as caught:
        ask(sid, [bump_version_then_reply], question="second")

    assert caught.value.kind == errors.CONFLICT
    assert questions(sid) == ["question"]


def test_independent_conversations_process_concurrently(sid):
    other = "fix-other-" + uuid.uuid4().hex
    both_started = threading.Barrier(2, timeout=5)
    results = {}

    def reply(text):
        def _reply():
            both_started.wait()      # both requests are in flight at once
            return AIMessage(content=text)
        return _reply

    def run(session_id, text):
        results[session_id] = assistant.ask_bankmate_with_sources(
            "q-" + text, session_id=session_id, model=FakeModel([reply(text)]), tools={}, request_id="same-id")

    try:
        threads = [threading.Thread(target=run, args=(sid, "A")),
                   threading.Thread(target=run, args=(other, "B"))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        assert results[sid]["answer"] == "A" and results[other]["answer"] == "B"
        assert questions(sid) == ["q-A"] and questions(other) == ["q-B"]
    finally:
        clear(other)


def test_saved_results_are_capped_per_conversation(sid, monkeypatch):
    monkeypatch.setattr(assistant, "MAX_SAVED_REQUESTS", 3)
    for i in range(5):
        ask(sid, [AIMessage(content=f"a{i}")], question=f"q{i}", request_id=f"r{i}")

    assert list(assistant.request_results[sid]) == ["r2", "r3", "r4"]


def test_expiry_clears_all_state_but_never_an_in_flight_session(sid):
    idle = "fix-idle-" + uuid.uuid4().hex
    busy = "fix-busy-" + uuid.uuid4().hex
    try:
        ask(idle, [AIMessage(content="a")], request_id="r1")
        ask(busy, [AIMessage(content="b")], request_id="r1")
        stale = time.time() - assistant.SESSION_TTL_SECONDS - 5
        assistant.session_last_active[idle] = stale
        assistant.session_last_active[busy] = stale
        assistant.requests_in_flight[busy] = {"r-running"}

        ask(sid, [AIMessage(content="c")])

        assert not assistant.has_conversation_memory(idle)
        for store in (assistant.conversation_sessions, assistant.grounding_context,
                      assistant.session_last_active, assistant.conversation_versions,
                      assistant.request_results):
            assert idle not in store
        assert assistant.has_conversation_memory(busy)
    finally:
        clear(idle)
        clear(busy)


def test_has_conversation_memory(sid):
    assert not assistant.has_conversation_memory(sid)
    ask(sid, [AIMessage(content="a")])
    assert assistant.has_conversation_memory(sid)
