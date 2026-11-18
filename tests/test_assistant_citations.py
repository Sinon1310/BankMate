"""
End-to-end citation behaviour of ask_bankmate using a scripted fake model
and fake tools: no Groq API calls and no ChromaDB access.
"""

import json
import uuid

import pytest
from langchain_core.messages import AIMessage

from app import assistant
from app.citations import SOURCES_HEADER


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.responses.pop(0)


class FakeTool:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.result


def tool_call(name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


REFUND_RESULT = json.dumps({
    "status": "SUCCESS",
    "sources": [
        {
            "policy_id": "POL-002",
            "policy_name": "TRANSACTION REFUND POLICY",
            "version": "1.0",
            "section": "PROCESSING",
            "source_file": "refund_policy.txt",
            "distance": 0.6,
        },
        {
            "policy_id": "POL-003",
            "policy_name": "TRANSACTION SECURITY POLICY",
            "version": "1.0",
            "section": "REPORTING",
            "source_file": "transaction_security_policy.txt",
            "distance": 1.1,
        },
    ],
    "evidence": [],
})

NO_POLICY_RESULT = json.dumps({
    "status": "NO_RELEVANT_POLICY_FOUND",
    "message": "No relevant information.",
    "sources": [],
    "evidence": [],
})


@pytest.fixture
def session_id():
    sid = "test-" + uuid.uuid4().hex
    yield sid
    assistant.conversation_sessions.pop(sid, None)
    assistant.grounding_context.pop(sid, None)


def run(session_id, responses, tools):
    return assistant.ask_bankmate_with_sources(
        "question",
        session_id=session_id,
        model=FakeModel(responses),
        tools=tools,
    )


def test_policy_answer_cites_only_used_policy(session_id):
    tools = {"search_banking_policies": FakeTool(REFUND_RESULT)}
    result = run(session_id, [
        tool_call("search_banking_policies", {"question": "refund approved"}),
        AIMessage(content="Approved refunds are credited.\nCITED_POLICIES: POL-002"),
    ], tools)

    assert result["answer"] == "Approved refunds are credited."
    assert [s["policy_id"] for s in result["sources"]] == ["POL-002"]


def test_unsupported_question_has_no_sources(session_id):
    # NO_RELEVANT_POLICY_FOUND path (nothing retrieved).
    tools = {"search_banking_policies": FakeTool(NO_POLICY_RESULT)}
    result = run(session_id, [
        tool_call("search_banking_policies", {"question": "home loan rate"}),
        AIMessage(
            content="The available documents do not contain the answer.\n"
                    "CITED_POLICIES: NONE"
        ),
    ], tools)

    assert result["sources"] == []
    assert "CITED_POLICIES" not in result["answer"]


def test_customer_only_question_has_no_sources(session_id):
    profile = FakeTool(json.dumps({"customer_id": "C101", "customer_name": "Arjun Mehta"}))
    tools = {"get_customer_profile": profile}
    text = assistant.ask_bankmate(
        "Show C101",
        session_id=session_id,
        model=FakeModel([
            tool_call("get_customer_profile", {"customer_id": "C101"}),
            AIMessage(content="Customer C101 is Arjun Mehta."),
        ]),
        tools=tools,
    )

    assert text == "Customer C101 is Arjun Mehta."
    assert SOURCES_HEADER not in text
    assert profile.calls == [{"customer_id": "C101"}]


def test_sources_from_previous_turn_do_not_leak(session_id):
    policy_tool = FakeTool(REFUND_RESULT)
    profile = FakeTool(json.dumps({"customer_id": "C101"}))
    tools = {"search_banking_policies": policy_tool, "get_customer_profile": profile}

    first = run(session_id, [
        tool_call("search_banking_policies", {"question": "refund"}),
        AIMessage(content="Refunds are credited.\nCITED_POLICIES: POL-002"),
    ], tools)
    second = run(session_id, [
        tool_call("get_customer_profile", {"customer_id": "C101"}),
        AIMessage(content="Customer C101 found. As noted, POL-002 applies."),
    ], tools)

    assert first["sources"]
    assert second["sources"] == []


def test_duplicate_retrievals_produce_single_citation(session_id):
    tools = {"search_banking_policies": FakeTool(REFUND_RESULT)}
    text = assistant.ask_bankmate(
        "refund",
        session_id=session_id,
        model=FakeModel([
            tool_call("search_banking_policies", {"question": "a"}, "c1"),
            tool_call("search_banking_policies", {"question": "b"}, "c2"),
            AIMessage(content="Refunds are credited.\nCITED_POLICIES: POL-002, POL-002"),
        ]),
        tools=tools,
    )

    answer, _, block = text.partition(SOURCES_HEADER)
    assert answer.strip() == "Refunds are credited."
    assert block.count("POL-002") == 1
    assert "POL-003" not in block


def test_malformed_policy_result_is_safe(session_id):
    tools = {"search_banking_policies": FakeTool("not json")}
    result = run(session_id, [
        tool_call("search_banking_policies", {"question": "x"}),
        AIMessage(content="Answer.\nCITED_POLICIES: POL-001"),
    ], tools)

    assert result["sources"] == []


def test_unknown_tool_and_step_limit(session_id):
    responses = [tool_call("nope", {}, f"c{i}") for i in range(assistant.MAX_TOOL_STEPS)]
    result = run(session_id, responses, {})

    assert result["sources"] == []
    assert "tool-call limit" in result["answer"]


def test_system_prompt_rules_are_uniquely_numbered():
    import re

    numbers = [
        int(n) for n in re.findall(r"(?m)^(\d+)\.", assistant.system_message.content)
    ]
    assert numbers == list(range(1, len(numbers) + 1))


# --- Regression: explicit coverage signal ---------------------------------

@pytest.mark.parametrize("content", [
    "I'm sorry, but I couldn't find any policy documents that provide "
    "the current home loan interest rate.\nCITED_POLICIES: NONE",
    "Home loan rates are not part of BankMate's material.\nCITED_POLICIES: NONE",
    # Model forgot the marker entirely: fail closed.
    "Home loan rates are not part of BankMate's material.",
    # Mentions a policy ID in prose but gives no marker: still no citation.
    "POL-002 covers refunds, not home loans.",
])
def test_no_citations_without_explicit_coverage_signal(session_id, content):
    tools = {"search_banking_policies": FakeTool(REFUND_RESULT)}
    text = assistant.ask_bankmate(
        "home loan rate?",
        session_id=session_id,
        model=FakeModel([
            tool_call("search_banking_policies", {"question": "home loan rate"}),
            AIMessage(content=content),
        ]),
        tools=tools,
    )
    assert SOURCES_HEADER not in text
    assert "CITED_POLICIES" not in text


# --- Regression: invented contact channels --------------------------------

DISPUTE_RESULT = json.dumps({
    "status": "SUCCESS",
    "sources": [{
        "policy_id": "POL-001",
        "policy_name": "CARD DISPUTE HANDLING POLICY",
        "version": "1.0",
        "section": "DISPUTE INITIATION",
        "source_file": "card_dispute_policy.txt",
        "distance": 0.8,
    }],
    "evidence": [{
        "policy_id": "POL-001",
        "section": "DISPUTE INITIATION",
        "content": (
            "A customer who identifies an unauthorized or incorrect card "
            "transaction should report it through the bank's official "
            "customer support channel."
        ),
    }],
})

INVENTED_ANSWER = (
    "1. Report it through the bank's official customer support channel "
    "(e.g., call, chat, or in-branch).\n"
    "2. Provide the transaction details.\n"
    "CITED_POLICIES: POL-001"
)


def test_invented_channels_trigger_revision(session_id):
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "unauthorized"}),
        AIMessage(content=INVENTED_ANSWER),
        AIMessage(content=(
            "1. Report it through the bank's official customer support channel.\n"
            "2. Provide the transaction details.\n"
            "CITED_POLICIES: POL-001"
        )),
    ])
    result = assistant.ask_bankmate_with_sources(
        "unauthorized?",
        session_id=session_id,
        model=model,
        tools={"search_banking_policies": FakeTool(DISPUTE_RESULT)},
    )

    assert model.calls == 3
    for word in ("call", "chat", "branch"):
        assert word not in result["answer"]
    assert "official customer support channel" in result["answer"]
    assert [s["policy_id"] for s in result["sources"]] == ["POL-001"]
    # History holds the corrected answer, not the invented one.
    assert "chat" not in assistant.conversation_sessions[session_id][-1].content


def test_invented_channels_removed_if_revision_still_invents(session_id):
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "unauthorized"}),
        AIMessage(content=INVENTED_ANSWER),
        # Revision still invents a channel and drops the marker.
        AIMessage(content=(
            "1. Report it through the bank's official customer support channel "
            "(for example the mobile app).\n"
            "2. Provide the transaction details."
        )),
    ])
    result = assistant.ask_bankmate_with_sources(
        "unauthorized?",
        session_id=session_id,
        model=model,
        tools={"search_banking_policies": FakeTool(DISPUTE_RESULT)},
    )

    assert result["answer"] == (
        "1. Report it through the bank's official customer support channel.\n"
        "2. Provide the transaction details."
    )
    # Original explicit citation signal is preserved.
    assert [s["policy_id"] for s in result["sources"]] == ["POL-001"]


def test_unremovable_channel_is_kept_with_warning(session_id, capsys):
    # Option (a): when an invented channel is not in an example list or
    # parenthetical, the sentence is kept rather than deleted.
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "home loan"}),
        AIMessage(content=(
            "The available documents do not contain the answer. "
            "Please call your branch.\nCITED_POLICIES: NONE"
        )),
        AIMessage(content="Please call your branch.\nCITED_POLICIES: NONE"),
    ])
    result = assistant.ask_bankmate_with_sources(
        "home loan?",
        session_id=session_id,
        model=model,
        tools={"search_banking_policies": FakeTool(NO_POLICY_RESULT)},
    )

    assert result["answer"] == "Please call your branch."
    assert result["sources"] == []
    assert "WARNING unsupported channels remain" in capsys.readouterr().out


SECURITY_RESULT = json.dumps({
    "status": "SUCCESS",
    "sources": [{
        "policy_id": "POL-003",
        "policy_name": "TRANSACTION SECURITY GUIDELINES",
        "version": "1.0",
        "section": "GENERAL",
        "source_file": "transaction_security_policy.txt",
    }],
    "evidence": [{
        "content": (
            "Customers should never share their PIN, password, "
            "OTP or confidential authentication credentials."
        ),
    }],
})


def test_supported_instruction_survives_cleanup(session_id):
    answer = (
        "Never share your OTP with anyone, including over the phone.\n"
        "CITED_POLICIES: POL-003"
    )
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "otp"}),
        AIMessage(content=answer),
        AIMessage(content=answer),  # revision ignores the request
    ])
    result = assistant.ask_bankmate_with_sources(
        "otp?",
        session_id=session_id,
        model=model,
        tools={"search_banking_policies": FakeTool(SECURITY_RESULT)},
    )

    assert result["answer"] == "Never share your OTP with anyone."
    assert [s["policy_id"] for s in result["sources"]] == ["POL-003"]


def test_customer_data_supports_channel_like_words(session_id):
    transactions = FakeTool(json.dumps([
        {"customer_id": "C101", "merchant": "Branch Bakery", "amount": "300"}
    ]))
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "refund"}, "c1"),
        tool_call("get_customer_transactions", {"customer_id": "C101"}, "c2"),
        AIMessage(content=(
            "C101's INR 300 payment to Branch Bakery can be refunded to the "
            "original payment method.\nCITED_POLICIES: POL-002"
        )),
    ])
    result = assistant.ask_bankmate_with_sources(
        "refund C101 bakery?",
        session_id=session_id,
        model=model,
        tools={
            "search_banking_policies": FakeTool(REFUND_RESULT),
            "get_customer_transactions": transactions,
        },
    )

    assert model.calls == 3  # no revision request
    assert "Branch Bakery" in result["answer"]


def test_follow_up_answer_is_also_grounding_checked(session_id):
    tools = {"search_banking_policies": FakeTool(DISPUTE_RESULT)}
    run(session_id, [
        tool_call("search_banking_policies", {"question": "unauthorized"}),
        AIMessage(content="Report it via the official channel.\nCITED_POLICIES: POL-001"),
    ], tools)

    model = FakeModel([
        AIMessage(content="He can also email the bank (or use live chat)."),
        AIMessage(content="He should use the official customer support channel."),
    ])
    result = assistant.ask_bankmate_with_sources(
        "anything else?", session_id=session_id, model=model, tools=tools
    )

    assert model.calls == 2
    assert result["answer"] == "He should use the official customer support channel."
    assert result["sources"] == []


def test_inline_marker_is_never_shown(session_id):
    tools = {"search_banking_policies": FakeTool(REFUND_RESULT)}
    text = assistant.ask_bankmate(
        "refund?",
        session_id=session_id,
        model=FakeModel([
            tool_call("search_banking_policies", {"question": "refund"}),
            AIMessage(content="Approved refunds are credited. CITED_POLICIES: POL-002"),
        ]),
        tools=tools,
    )

    assert "CITED_POLICIES" not in text
    assert text.startswith("Approved refunds are credited.\n\n" + SOURCES_HEADER)
    assert "POL-002" in text


def test_grounding_check_skipped_without_policy_search(session_id):
    model = FakeModel([AIMessage(content="I can't place a call for you.")])
    result = assistant.ask_bankmate_with_sources(
        "call me", session_id=session_id, model=model, tools={}
    )
    assert model.calls == 1
    assert result["answer"] == "I can't place a call for you."
