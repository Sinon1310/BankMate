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
    assistant.session_last_active.pop(sid, None)


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
            # Supported claim only: the point of this test is that the
            # merchant name "Branch Bakery" is not treated as a channel.
            "C101's INR 300 payment to Branch Bakery can be raised as a "
            "refund request.\nCITED_POLICIES: POL-002"
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


# --- Regression: unsupported procedural details (POL-002) -----------------

from pathlib import Path

from app.document_loader import split_into_sections

_REFUND_TEXT = (Path(__file__).resolve().parent.parent / "knowledge" / "policies"
                / "refund_policy.txt").read_text(encoding="utf-8")
_POL_002 = dict(split_into_sections(_REFUND_TEXT))

POL_002_RESULT = json.dumps({
    "status": "SUCCESS",
    "sources": [
        {"policy_id": "POL-002", "policy_name": "TRANSACTION REFUND POLICY", "version": "1.0",
         "section": section, "source_file": "refund_policy.txt"}
        for section in ("PROCESSING", "PURPOSE", "TRACKING")
    ],
    "evidence": [
        {"policy_id": "POL-002", "section": section, "content": _POL_002[section]}
        for section in ("PROCESSING", "PURPOSE", "TRACKING")
    ],
})

INVENTED_REFUND_ANSWER = (
    "When a refund is approved, it is processed through the payment processing "
    "workflow. The refund is assigned a unique reference ID and its status is "
    "updated to Processing before it is completed and credited back to the "
    "customer's original payment method.\nCITED_POLICIES: POL-002"
)

NOT_SPECIFIED = "The policy does not specify where the refunded amount is credited."


def refund_tools():
    return {"search_banking_policies": FakeTool(POL_002_RESULT)}


def test_direct_question_unsupported_refund_destination_is_revised(session_id):
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "refund approved"}),
        AIMessage(content=INVENTED_REFUND_ANSWER),
        AIMessage(content=(
            "When a refund is approved, it is processed through the appropriate "
            "payment processing workflow. Each refund has a unique reference ID and "
            "is tracked as Pending, Processing, Completed or Rejected. "
            + NOT_SPECIFIED + "\nCITED_POLICIES: POL-002"
        )),
    ])
    result = assistant.ask_bankmate_with_sources(
        "What happens when a refund is approved?", session_id=session_id,
        model=model, tools=refund_tools(),
    )

    assert model.calls == 3
    assert "original payment method" not in result["answer"]
    assert "credited back" not in result["answer"]
    assert NOT_SPECIFIED in result["answer"]
    assert [s["policy_id"] for s in result["sources"]] == ["POL-002"]
    # History holds the grounded answer, so follow-ups don't build on the claim.
    assert "original payment method" not in assistant.conversation_sessions[session_id][-1].content


def test_revision_prompt_asks_for_explicit_not_specified(session_id):
    seen = []

    class RecordingModel(FakeModel):
        def invoke(self, messages):
            seen.append(messages[-1].content)
            return super().invoke(messages)

    model = RecordingModel([
        tool_call("search_banking_policies", {"question": "refund"}),
        AIMessage(content=INVENTED_REFUND_ANSWER),
        AIMessage(content="Processed through the payment processing workflow.\nCITED_POLICIES: POL-002"),
    ])
    assistant.ask_bankmate_with_sources("refund?", session_id=session_id, model=model, tools=refund_tools())

    correction = seen[-1]
    assert "where the refunded amount is credited" in correction
    assert "The policy does not specify" in correction


def test_unrevised_claim_is_replaced_with_not_specified(session_id):
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "refund"}),
        AIMessage(content=INVENTED_REFUND_ANSWER),
        AIMessage(content=INVENTED_REFUND_ANSWER),  # model ignores the revision request
    ])
    result = assistant.ask_bankmate_with_sources(
        "refund?", session_id=session_id, model=model, tools=refund_tools(),
    )

    assert result["answer"] == (
        "When a refund is approved, it is processed through the payment processing "
        "workflow. " + NOT_SPECIFIED
    )
    assert [s["policy_id"] for s in result["sources"]] == ["POL-002"]


def test_question_about_unspecified_detail_gets_explicit_answer(session_id):
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "where is refund credited"}),
        AIMessage(content=NOT_SPECIFIED + "\nCITED_POLICIES: POL-002"),
    ])
    result = assistant.ask_bankmate_with_sources(
        "Where is the refunded amount credited?", session_id=session_id,
        model=model, tools=refund_tools(),
    )

    assert model.calls == 2  # already grounded: no revision
    assert result["answer"] == NOT_SPECIFIED


def test_follow_up_from_memory_is_checked(session_id):
    tools = refund_tools()
    run(session_id, [
        tool_call("search_banking_policies", {"question": "refund approved"}),
        AIMessage(content="It is processed through the payment processing workflow.\nCITED_POLICIES: POL-002"),
    ], tools)

    model = FakeModel([
        # Follow-up answered from memory, no new tool call.
        AIMessage(content="It will be credited to the original payment method within 5-7 business days."),
        AIMessage(content=NOT_SPECIFIED + " The policy does not specify a timeframe."),
    ])
    result = assistant.ask_bankmate_with_sources(
        "And where does the money go, and how long does it take?",
        session_id=session_id, model=model, tools=tools,
    )

    assert model.calls == 2
    assert "original payment method" not in result["answer"]
    assert "5-7" not in result["answer"]
    assert NOT_SPECIFIED in result["answer"]


def test_follow_up_that_searches_again_is_checked(session_id):
    tools = refund_tools()
    run(session_id, [
        tool_call("search_banking_policies", {"question": "refund"}, "c1"),
        AIMessage(content="It is processed through the workflow.\nCITED_POLICIES: POL-002"),
    ], tools)

    model = FakeModel([
        tool_call("search_banking_policies", {"question": "refund destination"}, "c2"),
        AIMessage(content="Refunds are returned to your card.\nCITED_POLICIES: POL-002"),
        AIMessage(content="Refunds are returned to your card.\nCITED_POLICIES: POL-002"),
    ])
    result = assistant.ask_bankmate_with_sources(
        "Where do refunds go?", session_id=session_id, model=model, tools=tools,
    )

    assert result["answer"] == NOT_SPECIFIED
    assert [s["policy_id"] for s in result["sources"]] == ["POL-002"]


def test_customer_data_turn_is_not_detail_checked(session_id):
    tools = {
        **refund_tools(),
        "get_customer_transactions": FakeTool(json.dumps([
            {"transaction_id": "T003", "customer_id": "C101", "transaction_type": "Credit",
             "amount": "25000", "merchant": "Salary"}
        ])),
    }
    run(session_id, [
        tool_call("search_banking_policies", {"question": "refund"}),
        AIMessage(content="Processed through the workflow.\nCITED_POLICIES: POL-002"),
    ], tools)

    model = FakeModel([
        tool_call("get_customer_transactions", {"customer_id": "C101"}),
        AIMessage(content="His salary of INR 25,000 was credited to his account on 3 Nov."),
    ])
    result = assistant.ask_bankmate_with_sources(
        "Show C101's credits", session_id=session_id, model=model, tools=tools,
    )

    assert model.calls == 2  # no revision request on customer-data answers
    assert result["answer"] == "His salary of INR 25,000 was credited to his account on 3 Nov."


def test_policy_grounding_rules_are_in_system_prompt():
    prompt = assistant.system_message.content
    assert "POLICY GROUNDING RULES" in prompt
    assert "The policy does not specify <detail>." in prompt


def test_grounded_not_specified_answer_needs_no_revision(session_id):
    answer = (
        "Once approved, the refund is processed through the appropriate payment "
        "processing workflow. The policy does not specify any fees or charges, "
        "a timeframe, or where the refunded amount is credited.\nCITED_POLICIES: POL-002"
    )
    model = FakeModel([
        tool_call("search_banking_policies", {"question": "refund fees and timing"}),
        AIMessage(content=answer),
    ])
    result = assistant.ask_bankmate_with_sources(
        "Are there refund fees, and how long does it take?", session_id=session_id,
        model=model, tools=refund_tools(),
    )

    assert model.calls == 2  # no revision request
    assert result["answer"] == answer.split("\nCITED_POLICIES")[0]
    assert [s["policy_id"] for s in result["sources"]] == ["POL-002"]


def test_core_safety_and_grounding_instructions_remain_in_prompt():
    """Guards prompt clean-ups from dropping a safety or grounding rule."""

    prompt = assistant.system_message.content
    for instruction in (
        "Never invent customer details or financial figures.",
        "Do not recommend financial products.",
        "Do not make credit eligibility or approval decisions.",
        "Never describe transaction flow as account balance.",
        "Do not invent policy information.",
        "do not attempt to answer using general knowledge.",
        "Never invent policy IDs, sections or document names.",
        "Do not mix information belonging to different customers.",
        "state only what the retrieved policy text says.",
    ):
        assert instruction in prompt, instruction

    # Each instruction appears once (no duplicated rules).
    assert prompt.count("recommend financial products") == 1
    assert prompt.count("credit eligibility") == 1
