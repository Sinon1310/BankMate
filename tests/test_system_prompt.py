"""
System prompt for customer intelligence (Stage 1C, gate 6).

Deterministic checks only: the prompt's rules are present and consistent
with the code (tool names, filters, income categories, response fields),
existing rules are preserved, and the prompt stays within the token budget.
They do not, and cannot, show how a live model follows the rules.
"""

import json
import re

import pytest
from langchain_core.messages import AIMessage

import app.data_loader as dl
import app.tools as tools
from app import assistant, ledger, llm, tool_args
from scripts import generate_data as gen

PROMPT = assistant.system_message.content


def rule(number):
    """The text of rule `number` (up to the next rule or section heading)."""
    match = re.search(rf"(?ms)^{number}\. (.*?)(?=^\d+\. |^[A-Z][A-Z ]+:$|\Z)", PROMPT)
    assert match, number
    return " ".join(match.group(1).split())


# ---------- Structure and preservation ----------

def test_sections_in_order():
    headings = re.findall(r"(?m)^([A-Z][A-Z ]+):$", PROMPT)
    assert headings == ["CITATION RULES", "CONVERSATION MEMORY RULES", "POLICY GROUNDING RULES",
                        "SECURITY RULES", "CUSTOMER TOOL RULES", "FINANCIAL RULES",
                        "PAGINATION AND FILTER RULES", "CUSTOMER DATA RULES"]


def test_existing_rule_numbers_and_sections_are_preserved():
    # CLAUDE.md refers to rules 37-40 (grounding) and 41-44 (security) by number.
    sections = {n: heading for heading, block in re.findall(r"(?ms)^([A-Z][A-Z ]+):$(.*?)(?=^[A-Z][A-Z ]+:$|\Z)", PROMPT)
                for n in map(int, re.findall(r"(?m)^(\d+)\.", block))}
    assert all(sections[n] == "POLICY GROUNDING RULES" for n in range(37, 41))
    assert all(sections[n] == "SECURITY RULES" for n in range(41, 45))
    assert all(sections[n] == "CITATION RULES" for n in range(27, 31))
    assert rule(5) == "Never describe transaction flow as account balance."
    assert rule(42).startswith("Only look up customers whose IDs the user has given in this conversation.")
    assert rule(27).startswith("If you called search_banking_policies while answering, you must end your final answer")


def test_rule_13_points_customer_dispute_records_to_the_disputes_tool():
    assert "search_banking_policies" in rule(13) and "get_customer_disputes" in rule(13)


# ---------- Consistency with the code ----------

def test_every_registered_tool_is_named_in_the_tool_rules():
    assert {t.name for t in llm.tools} == set(assistant.tool_map)
    for name in assistant.tool_map:
        assert name in rule(45), name


def test_listed_filters_match_the_transaction_tool():
    filters = set(tool_args.TOOL_SPECS["get_customer_transactions"]) - {"limit", "offset"}
    listed = set(re.findall(r"[a-z]+_[a-z_]+|\bcategory\b|\bmerchant\b", rule(46)))
    assert "merchant" in filters and filters <= listed
    assert "never page through history" in rule(46)                       # gate 7: no page scanning


def test_income_and_transfer_definitions_match_the_ledger():
    text = rule(47)
    for category in ledger.INCOME_CATEGORIES:
        assert category in text
    assert "Transfer In is not income" in text and "is not spending" in text
    assert "Cash withdrawals are spending" in text
    assert "directions, not categories" in text


def test_financial_rules_cover_refunds_balances_totals_and_disputes():
    assert "Refunds are credits, never income" in rule(48) and "original purchase's category" in rule(48)
    assert "period of the refund date" in rule(48)                         # cash basis, as in app.ledger
    assert "spending_gross" in rule(48) and "spending_net" in rule(48)
    assert "balance" in rule(49) and "net transaction flow" in rule(49) and "net spending" in rule(49)
    assert "Available balance is not tracked" in rule(49)
    assert "never by adding displayed rows" in rule(50)
    # A disputed debit is a real transaction, counted once; the dispute itself adds nothing.
    assert "already counted once in totals" in rule(51) and "never add or subtract dispute amounts" in rule(51)
    assert "Resolved does not mean refunded" in rule(51)


def test_pagination_filter_and_data_rules():
    assert "has_more" in rule(52) and "never claim the full history was reviewed" in rule(52)
    assert "next_offset" in rule(52)
    for marker in ("ignored_arguments_notice", "NOT_AVAILABLE", "INVALID_ARGUMENT"):
        assert marker in rule(53)
    assert "never present unfiltered results as filtered" in rule(53)
    assert "State only customer facts returned by tools" in rule(54)
    assert "last four digits" in rule(55)
    assert "₹" in rule(56)


# ---------- Every field the prompt relies on exists in real tool output ----------

@pytest.fixture(scope="module")
def approved(tmp_path_factory):
    folder = tmp_path_factory.mktemp("approved")
    assert gen.main(["--output-dir", str(folder)]) == 0
    return dl.load_dataset(folder)


def test_fields_named_in_the_prompt_exist_in_tool_output(monkeypatch, approved):
    monkeypatch.setattr(dl, "_dataset", approved)
    page = json.loads(tools.get_customer_transactions.invoke({"customer_id": "C101"}))
    metrics = json.loads(tools.get_customer_metrics.invoke({"customer_id": "C101", "start_date": "2026-01-01"}))
    resolved_no_refund = json.loads(tools.get_customer_disputes.invoke({"customer_id": "C111"}))["disputes"][0]
    resolved_refund = json.loads(tools.get_customer_disputes.invoke({"customer_id": "C101"}))["disputes"][0]
    unsupported = json.loads(tools.get_customer_transactions.invoke({"customer_id": "C101", "limit": 500}))

    for field in ("matching_totals", "has_more", "next_offset", "filters"):
        assert field in page and field in PROMPT, field
    for field in ("spending_gross", "spending_net"):
        assert field in metrics and field in PROMPT, field
    assert metrics["period"]["clamped"] is True and "period.clamped" in PROMPT
    assert "as_of_date" in page and ("as-of date" in PROMPT)
    assert "refund" in resolved_refund and "refund" not in resolved_no_refund        # rule 51's distinction
    assert unsupported["status"] == "INVALID_ARGUMENT" and "INVALID_ARGUMENT" in PROMPT


def test_ignored_and_not_available_markers_exist_in_assistant_and_tools():
    notice = json.loads(assistant._with_ignored_notice("{}", "get_customer_transactions", 1))
    assert "ignored_arguments_notice" in notice and "ignored_arguments_notice" in PROMPT
    assert "NOT_AVAILABLE" in tools._legacy_unsupported() and "NOT_AVAILABLE" in PROMPT


# ---------- The model receives it ----------

def test_the_model_receives_the_updated_prompt_first():
    class Model:
        seen = []

        def invoke(self, messages):
            self.seen.append(messages)
            return AIMessage(content="Hello.")

    model = Model()
    sid = "prompt-test"
    try:
        assistant.ask_bankmate_with_sources("Hello", session_id=sid, model=model, tools={})
        first = model.seen[0][0]
        assert first is assistant.system_message and "CUSTOMER TOOL RULES" in first.content
    finally:
        for store in (assistant.conversation_sessions, assistant.grounding_context, assistant.session_last_active,
                      assistant.conversation_versions, assistant.request_results, assistant.requests_in_flight):
            store.pop(sid, None)


# ---------- Budget ----------

def test_prompt_stays_within_its_budget():
    # The prompt (plus ~1,000 tokens of tool schemas) is resent on every model
    # call; Groq's free tier allows 8,000 tokens per minute.
    assert len(PROMPT) <= 7600
    assert PROMPT.count("recommend financial products") == 1
