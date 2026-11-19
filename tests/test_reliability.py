"""
Stage 5.2 reliability tests: atomic turns, retries, provider failures,
history trimming, idle-session cleanup and isolation.

Mocked models and tools only; no Groq calls, no ChromaDB.
"""

import json
import logging
import time
import uuid

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app import assistant, errors, llm
from app.history import is_complete_history, split_turns, trim_to_recent_turns


REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

POLICY_RESULT = json.dumps({
    "status": "SUCCESS",
    "sources": [{"policy_id": "POL-002", "policy_name": "TRANSACTION REFUND POLICY",
                 "version": "1.0", "section": "PROCESSING", "source_file": "refund_policy.txt"}],
    "evidence": [{"policy_id": "POL-002", "section": "PROCESSING",
                  "content": "Once approved, the refund should be processed through the "
                             "appropriate payment processing workflow."}],
})


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeTool:
    def __init__(self, result="{}", error=None):
        self.result = result
        self.error = error

    def invoke(self, args):
        if self.error:
            raise self.error
        return self.result


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def rate_limit_error(retry_after="13"):
    headers = {"retry-after": retry_after} if retry_after else {}
    return groq.RateLimitError(
        "Rate limit reached for model on tokens per minute (TPM): Limit 8000",
        response=httpx.Response(429, request=REQUEST, headers=headers),
        body=None,
    )


@pytest.fixture
def sid():
    session_id = "rel-" + uuid.uuid4().hex
    yield session_id
    for store in (assistant.conversation_sessions, assistant.grounding_context,
                  assistant.session_last_active):
        store.pop(session_id, None)


def ask(session_id, responses, tools=None, question="question"):
    model = FakeModel(responses)
    result = assistant.ask_bankmate_with_sources(
        question, session_id=session_id, model=model, tools=tools or {},
    )
    return result, model


def stored(session_id):
    return assistant.conversation_sessions.get(session_id)


def snapshot(session_id):
    history = list(stored(session_id) or [])
    context = json.loads(json.dumps(assistant.grounding_context.get(session_id) or {}, default=list))
    return history, context


def human_texts(session_id):
    return [m.content for m in stored(session_id) if isinstance(m, HumanMessage)]


# ---------- 1. Atomic turns: tool failure mid-response ----------

def test_tool_failure_raises_friendly_error_and_keeps_history(sid):
    ask(sid, [AIMessage(content="Hello.")], question="first")
    before = snapshot(sid)

    with pytest.raises(errors.AssistantError) as caught:
        ask(sid, [tool_call("search_banking_policies", {"question": "x"}, "c1")],
            tools={"search_banking_policies": FakeTool(error=RuntimeError("chroma unavailable"))},
            question="second")

    assert caught.value.kind == errors.TOOL_FAILURE
    assert "chroma" not in caught.value.user_message.lower()
    assert snapshot(sid) == before                     # history and grounding untouched
    assert is_complete_history(stored(sid))           # no dangling tool call
    assert human_texts(sid) == ["first"]


def test_failure_on_first_turn_leaves_no_partial_session(sid):
    with pytest.raises(errors.AssistantError):
        ask(sid, [tool_call("get_customer_profile", {"customer_id": "C101"}, "c1")],
            tools={"get_customer_profile": FakeTool(error=ValueError("boom"))})

    assert stored(sid) is None
    assert sid not in assistant.grounding_context


def test_retry_after_failure_does_not_duplicate_question(sid):
    with pytest.raises(errors.AssistantError):
        ask(sid, [rate_limit_error()], question="What happens when a refund is approved?")

    result, model = ask(sid, [AIMessage(content="It is processed.")],
                        question="What happens when a refund is approved?")

    assert result["answer"] == "It is processed."
    assert human_texts(sid) == ["What happens when a refund is approved?"]
    # The model saw the question exactly once in the retried request.
    sent = [m.content for m in model.calls[0] if isinstance(m, HumanMessage)]
    assert sent == ["What happens when a refund is approved?"]


def test_successful_tool_call_history_is_complete(sid):
    ask(sid, [
        tool_call("search_banking_policies", {"question": "refund"}, "c1"),
        AIMessage(content="Processed through the workflow.\nCITED_POLICIES: POL-002"),
    ], tools={"search_banking_policies": FakeTool(POLICY_RESULT)})

    history = stored(sid)
    assert is_complete_history(history)
    assert isinstance(history[0], SystemMessage)
    assert [type(m).__name__ for m in history[1:]] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"
    ]
    assert history[2].tool_calls[0]["id"] == history[3].tool_call_id == "c1"


def test_tool_limit_turn_is_committed_complete(sid):
    responses = [tool_call("get_customer_profile", {"customer_id": "C101"}, f"c{i}")
                 for i in range(assistant.MAX_TOOL_STEPS)]
    result, _ = ask(sid, responses, tools={"get_customer_profile": FakeTool('{"customer_id": "C101"}')})

    assert result["answer"] == assistant.TOOL_LIMIT_MESSAGE
    assert is_complete_history(stored(sid))
    assert stored(sid)[-1].content == assistant.TOOL_LIMIT_MESSAGE


# ---------- 2. Provider failures ----------

def test_rate_limit_is_classified_with_retry_after_and_not_retried_by_engine(sid):
    model = FakeModel([rate_limit_error("13")])

    with pytest.raises(errors.AssistantError) as caught:
        assistant.ask_bankmate_with_sources("q", session_id=sid, model=model, tools={})

    failure = caught.value
    assert failure.kind == errors.RATE_LIMITED
    assert failure.retry_after == 13
    assert "about 13 seconds" in failure.user_message
    assert "TPM" not in failure.user_message and "8000" not in failure.user_message
    assert len(model.calls) == 1        # no engine-level retry loop
    assert stored(sid) is None


def test_rate_limit_without_retry_after_header():
    failure = errors.classify_provider_error(rate_limit_error(retry_after=None))
    assert failure.kind == errors.RATE_LIMITED and failure.retry_after is None
    assert failure.user_message == errors.USER_MESSAGES[errors.RATE_LIMITED]


@pytest.mark.parametrize("error", [groq.APITimeoutError(request=REQUEST), TimeoutError("read timed out")])
def test_timeout_is_classified(sid, error):
    with pytest.raises(errors.AssistantError) as caught:
        ask(sid, [error])

    assert caught.value.kind == errors.TIMEOUT
    assert caught.value.user_message == errors.USER_MESSAGES[errors.TIMEOUT]
    assert stored(sid) is None


@pytest.mark.parametrize("error, kind", [
    (groq.APIConnectionError(request=REQUEST), errors.CONNECTION),
    (groq.AuthenticationError("Invalid API Key gsk_secret",
                              response=httpx.Response(401, request=REQUEST), body=None), errors.AUTHENTICATION),
    (groq.InternalServerError("upstream exploded",
                              response=httpx.Response(503, request=REQUEST), body=None), errors.SERVICE),
    (KeyError("weird"), errors.UNEXPECTED),
])
def test_other_provider_errors_are_classified(error, kind):
    failure = errors.classify_provider_error(error)
    assert failure.kind == kind
    assert "gsk_" not in failure.user_message and "exploded" not in failure.user_message


def test_failure_detail_goes_to_logs_not_users(sid, caplog):
    with caplog.at_level(logging.WARNING, logger="app.assistant"):
        with pytest.raises(errors.AssistantError) as caught:
            ask(sid, [rate_limit_error()])

    assert "Rate limit reached" in caplog.text           # developers can diagnose
    assert "Rate limit reached" not in caught.value.user_message


def test_revision_call_failure_does_not_fail_the_answer(sid):
    # A rate limit during the optional grounding revision falls back to
    # the deterministic cleanup instead of failing the whole response.
    result, model = ask(sid, [
        tool_call("search_banking_policies", {"question": "refund"}, "c1"),
        AIMessage(content="It is credited back to your card.\nCITED_POLICIES: POL-002"),
        rate_limit_error(),
    ], tools={"search_banking_policies": FakeTool(POLICY_RESULT)})

    assert result["answer"] == "The policy does not specify where the refunded amount is credited."
    assert [s["policy_id"] for s in result["sources"]] == ["POL-002"]
    assert len(model.calls) == 3


# ---------- 3. LLM client configuration ----------

def test_model_is_built_with_timeout_and_limited_retries():
    bound = llm.get_assistant_model()
    client = bound.bound

    assert client.request_timeout == llm.LLM_TIMEOUT_SECONDS == 30.0
    assert client.max_retries == llm.LLM_MAX_RETRIES == 1


def test_env_number_parsing(monkeypatch):
    monkeypatch.setenv("BANKMATE_TEST_NUMBER", "45")
    assert llm.env_number("BANKMATE_TEST_NUMBER", 30.0, float, minimum=1) == 45.0

    monkeypatch.setenv("BANKMATE_TEST_NUMBER", "not-a-number")
    assert llm.env_number("BANKMATE_TEST_NUMBER", 30.0, float, minimum=1) == 30.0

    monkeypatch.setenv("BANKMATE_TEST_NUMBER", "0")
    assert llm.env_number("BANKMATE_TEST_NUMBER", 30.0, float, minimum=1) == 30.0

    monkeypatch.delenv("BANKMATE_TEST_NUMBER")
    assert llm.env_number("BANKMATE_TEST_NUMBER", 6, int) == 6


# ---------- 4. History trimming ----------

def run_turns(session_id, count, start=0):
    tools = {"search_banking_policies": FakeTool(POLICY_RESULT),
             "get_customer_profile": FakeTool('{"customer_id": "C101"}')}
    for i in range(start, start + count):
        name = "search_banking_policies" if i % 2 == 0 else "get_customer_profile"
        ask(session_id, [
            tool_call(name, {"question": f"q{i}"} if i % 2 == 0 else {"customer_id": "C101"}, f"call-{i}"),
            AIMessage(content=f"answer {i}"),
        ], tools=tools, question=f"turn {i}")


def test_history_is_trimmed_to_recent_complete_turns(sid, monkeypatch):
    monkeypatch.setattr(assistant, "MAX_HISTORY_TURNS", 3)
    run_turns(sid, 7)

    history = stored(sid)
    prefix, turns = split_turns(history)

    assert prefix == [assistant.system_message]           # system prompt kept
    assert len(turns) == 3
    assert human_texts(sid) == ["turn 4", "turn 5", "turn 6"]
    assert is_complete_history(history)
    for turn in turns:                                    # every call has its result
        calls = [c["id"] for m in turn if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
        results = [m.tool_call_id for m in turn if isinstance(m, ToolMessage)]
        assert calls == results


def test_grounding_text_is_trimmed_with_its_turns(sid, monkeypatch):
    monkeypatch.setattr(assistant, "MAX_HISTORY_TURNS", 2)
    run_turns(sid, 5)    # policy on turns 0, 2, 4; customer data on 1, 3

    context = assistant.grounding_context[sid]
    turns = context["turns"]
    assert len(turns) == 2                                 # synchronized with kept turns
    assert [t["kind"] for t in turns] == [assistant.TURN_CUSTOMER, assistant.TURN_POLICY]
    assert len(turns[0]["customer"]) == 1 and turns[0]["evidence"] == []   # turn 3
    assert len(turns[1]["evidence"]) == 1 and turns[1]["customer"] == []   # turn 4
    assert context["policy_tool_used"] is True


def test_follow_up_still_sees_previous_turn_after_trimming(sid, monkeypatch):
    monkeypatch.setattr(assistant, "MAX_HISTORY_TURNS", 2)
    run_turns(sid, 4)

    _, model = ask(sid, [AIMessage(content="ok")], question="and what about him?")
    sent = [m.content for m in model.calls[0] if isinstance(m, HumanMessage)]
    assert sent == ["turn 2", "turn 3", "and what about him?"]


def test_trim_helper_never_splits_a_tool_pair():
    history = [SystemMessage(content="s")]
    for i in range(4):
        history += [HumanMessage(content=f"h{i}"), tool_call("t", {}, f"id{i}"),
                    ToolMessage(content="r", tool_call_id=f"id{i}"), AIMessage(content=f"a{i}")]

    for keep in range(1, 5):
        trimmed, dropped = trim_to_recent_turns(history, keep)
        assert dropped == 4 - keep
        assert is_complete_history(trimmed)
        assert trimmed[0].content == "s"


def test_is_complete_history_detects_dangling_and_orphaned_calls():
    base = [SystemMessage(content="s"), HumanMessage(content="h")]
    assert not is_complete_history(base + [tool_call("t", {}, "x")])
    assert not is_complete_history(base + [ToolMessage(content="r", tool_call_id="nope")])
    assert not is_complete_history(base + [tool_call("t", {}, "x"), HumanMessage(content="h2")])
    assert is_complete_history(base + [tool_call("t", {}, "x"),
                                       ToolMessage(content="r", tool_call_id="x"),
                                       AIMessage(content="done")])


# ---------- 5. Idle session cleanup ----------

def test_idle_sessions_expire_and_state_stays_synchronized(sid):
    idle = "rel-idle-" + uuid.uuid4().hex
    recent = "rel-recent-" + uuid.uuid4().hex
    try:
        ask(idle, [AIMessage(content="a")])
        ask(recent, [AIMessage(content="b")])
        assistant.session_last_active[idle] = time.time() - assistant.SESSION_TTL_SECONDS - 5

        ask(sid, [AIMessage(content="c")])    # any request triggers lazy cleanup

        for store in (assistant.conversation_sessions, assistant.grounding_context,
                      assistant.session_last_active):
            assert idle not in store
            assert recent in store
            assert sid in store
    finally:
        for name in (idle, recent):
            for store in (assistant.conversation_sessions, assistant.grounding_context,
                          assistant.session_last_active):
                store.pop(name, None)


def test_current_session_is_never_evicted_even_if_stale(sid):
    ask(sid, [AIMessage(content="a")], question="first")
    assistant.session_last_active[sid] = time.time() - assistant.SESSION_TTL_SECONDS - 5

    ask(sid, [AIMessage(content="b")], question="second")

    assert human_texts(sid) == ["first", "second"]


# ---------- 6. Isolation ----------

def test_conversations_stay_isolated_including_failures(sid):
    other = "rel-other-" + uuid.uuid4().hex
    try:
        ask(sid, [AIMessage(content="a1")], question="A first")
        ask(other, [AIMessage(content="b1")], question="B first")

        with pytest.raises(errors.AssistantError):
            ask(other, [rate_limit_error()], question="B failing")

        _, model = ask(sid, [AIMessage(content="a2")], question="A second")

        assert human_texts(sid) == ["A first", "A second"]
        assert human_texts(other) == ["B first"]
        sent = [m.content for m in model.calls[0] if isinstance(m, HumanMessage)]
        assert sent == ["A first", "A second"]
    finally:
        for store in (assistant.conversation_sessions, assistant.grounding_context,
                      assistant.session_last_active):
            store.pop(other, None)
