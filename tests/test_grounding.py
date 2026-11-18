import json

from app.grounding import (
    collect_evidence_text,
    find_unsupported_channels,
    remove_unsupported_examples,
)


EVIDENCE = (
    "A customer who identifies an unauthorized or incorrect card "
    "transaction should report it through the bank's official customer "
    "support channel."
)

INVENTED = (
    "1. Report the transaction immediately through the bank's official "
    "customer support channel (e.g., call, chat, or in-branch).\n"
    "2. Provide the transaction details."
)


def test_detects_invented_channels():
    assert find_unsupported_channels(INVENTED, EVIDENCE) == ["chat", "branch"]


def test_detects_specific_phone_phrases():
    for answer in (
        "Call the bank immediately.",
        "Contact the helpline.",
        "Report it over the phone.",
        "Use the toll-free number.",
    ):
        assert find_unsupported_channels(answer, EVIDENCE) == ["phone"], answer


def test_generic_policy_wording_is_supported():
    answer = "Report it through the bank's official customer support channel."
    assert find_unsupported_channels(answer, EVIDENCE) == []


def test_channel_present_in_support_text_is_allowed():
    support = EVIDENCE + " Customers may also email the support team."
    assert find_unsupported_channels("You can email the team.", support) == []


def test_no_false_positive_on_everyday_words():
    for answer in (
        "Policies call this a duplicate transaction.",
        "The refund was recalled and the application approved.",
        "C101 paid INR 499 to the Apple App Store.",
        "Apply the policy to this case.",
    ):
        assert find_unsupported_channels(answer, EVIDENCE) == [], answer


def test_remove_strips_parenthetical_example_list():
    cleaned = remove_unsupported_examples(INVENTED, ["chat", "branch"])
    assert cleaned == (
        "1. Report the transaction immediately through the bank's official "
        "customer support channel.\n"
        "2. Provide the transaction details."
    )


def test_remove_keeps_supported_instruction_in_same_sentence():
    answer = "Never share your OTP with anyone, including over the phone."
    assert remove_unsupported_examples(answer, ["phone"]) == (
        "Never share your OTP with anyone."
    )


def test_remove_handles_eg_without_leaving_fragment():
    answer = "Report via the official channel, e.g. call the bank or use chat."
    assert remove_unsupported_examples(answer, ["phone", "chat"]) == (
        "Report via the official channel."
    )


def test_remove_never_deletes_whole_sentences():
    answer = (
        "Report it through the official channel. Call the bank as well.\n"
        "- Visit your nearest branch.\n"
        "- Provide the transaction ID."
    )
    assert remove_unsupported_examples(answer, ["phone", "branch"]) == answer


def test_remove_keeps_parentheticals_without_unsupported_channels():
    answer = "Provide the transaction ID (from your statement)."
    assert remove_unsupported_examples(answer, ["chat"]) == answer


def test_collect_evidence_text_handles_bad_input():
    texts = []
    collect_evidence_text(texts, json.dumps({
        "status": "SUCCESS",
        "evidence": [{"content": "abc"}, {"no": "content"}, "bad"],
    }))
    collect_evidence_text(texts, json.dumps({"status": "NO_RELEVANT_POLICY_FOUND"}))
    collect_evidence_text(texts, "not json")
    collect_evidence_text(texts, None)
    assert texts == ["abc"]
