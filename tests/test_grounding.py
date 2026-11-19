import json

import pytest

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


# ---------- Procedural details (POL-002 regression) ----------

from pathlib import Path  # noqa: E402

from app.document_loader import split_into_sections
from app.grounding import (
    find_unsupported_details,
    not_specified_statement,
    replace_unsupported_detail_sentences,
)

REFUND_FILE = Path(__file__).resolve().parent.parent / "knowledge" / "policies" / "refund_policy.txt"
POL_002 = dict(split_into_sections(REFUND_FILE.read_text(encoding="utf-8")))
# The passages retrieval returns for "What happens when a refund is approved?"
POL_002_EVIDENCE = "\n".join(POL_002[s] for s in ("PROCESSING", "PURPOSE", "TRACKING"))


def test_pol_002_does_not_state_refund_destination_or_timeframe():
    # Guard: if the fictional policy ever gains these details, the
    # regression expectations below must be revisited.
    assert find_unsupported_details("credited back to the original payment method", POL_002_EVIDENCE)
    assert "original payment method" not in REFUND_FILE.read_text().lower()


@pytest.mark.parametrize("claim", [
    "The amount is credited back to the customer's original payment method.",
    "The refund is completed and credited back to the customer.",
    "The money is refunded to the customer's account.",
    "It is returned to your card.",
    # Leaked in a live answer during the final audit (no "to <destination>"):
    "Its status is updated to Processing (and later to Completed once the funds are returned).",
    "The amount is reimbursed after approval.",
])
def test_detects_unsupported_refund_destination(claim):
    assert find_unsupported_details(claim, POL_002_EVIDENCE) == ["refund destination"]


@pytest.mark.parametrize("claim, label", [
    ("Refunds are processed within 5-7 business days.", "timeframe"),
    ("It takes 3 working days.", "timeframe"),
    ("A processing fee applies.", "fees"),
])
def test_detects_unsupported_timeframe_and_fees(claim, label):
    assert label in find_unsupported_details(claim, POL_002_EVIDENCE)


@pytest.mark.parametrize("supported", [
    "Once approved, the refund is processed through the appropriate payment processing workflow.",
    "Each refund request has a unique reference ID; status is tracked as Pending, Processing, Completed or Rejected.",
    "The policy does not specify where the refunded amount is credited.",
    "C101 has 1 credit transaction.",
])
def test_policy_wording_is_not_flagged(supported):
    assert find_unsupported_details(supported, POL_002_EVIDENCE) == []


def test_detail_stated_in_evidence_is_allowed():
    evidence = POL_002_EVIDENCE + "\nRefunds are credited back to the original payment method within 5 days."
    answer = "The refund is credited back to the original payment method within 5 days."
    assert find_unsupported_details(answer, evidence) == []


def test_replacement_states_policy_does_not_specify_and_keeps_other_sentences():
    answer = (
        "When a refund is approved, it is processed through the payment processing workflow. "
        "It is then credited back to the customer's original payment method.\n"
        "- Status is tracked as Pending, Processing, Completed or Rejected."
    )

    assert replace_unsupported_detail_sentences(answer, ["refund destination"]) == (
        "When a refund is approved, it is processed through the payment processing workflow. "
        + not_specified_statement("refund destination") + "\n"
        "- Status is tracked as Pending, Processing, Completed or Rejected."
    )


def test_replacement_announces_each_category_once():
    answer = "It is credited back to your card. Funds are returned to the account in 3 days."
    result = replace_unsupported_detail_sentences(answer, ["refund destination", "timeframe"])

    assert result.count("does not specify where the refunded amount is credited") == 1
    assert result.count("does not specify a timeframe") == 1
    assert "card" not in result and "3 days" not in result


# ---------- "Policy does not specify" statements are not claims ----------

from app.grounding import find_unsupported_channels as _channels  # noqa: E402


@pytest.mark.parametrize("statement", [
    not_specified_statement("fees"),
    not_specified_statement("timeframe"),
    not_specified_statement("refund destination"),
    "The policy does not mention any fees or charges.",
    "The policy does not state whether refunds take 5 days.",
    "The policy doesn't say whether the amount is credited back to the card.",
    "No information is given about fees.",
])
def test_not_specified_statements_are_not_flagged_as_details(statement):
    assert find_unsupported_details(statement, POL_002_EVIDENCE) == []


def test_not_specified_channel_statement_is_not_flagged():
    assert _channels("The policy does not specify a phone number or email address.", POL_002_EVIDENCE) == []


@pytest.mark.parametrize("mixed, label", [
    ("The policy does not specify fees, but refunds are credited back to your card.", "refund destination"),
    ("The policy does not specify a timeframe; refunds take 5 days.", "timeframe"),
])
def test_claim_next_to_a_not_specified_clause_is_still_flagged(mixed, label):
    assert find_unsupported_details(mixed, POL_002_EVIDENCE) == [label]
