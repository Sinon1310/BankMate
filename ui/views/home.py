"""
Home: brand introduction, conversational entry point, suggested prompts
and a compact overview of what the assistant can do today.

Every control here leads somewhere real: prompts and cards open the AI
Assistant and send the selected question.
"""

from datetime import date

import streamlit as st

from app.data_loader import customers, transactions
from app.document_loader import POLICY_DIR
from app.security import MAX_QUESTION_CHARS
from ui import components as ui


SUGGESTED_PROMPTS = [
    ("history", "Show C102's transaction history"),
    ("compare", "Compare total debits for C101 and C103"),
    ("refund", "How are refunds handled for duplicate transactions?"),
    ("dispute", "What information is needed to raise a card dispute?"),
]

CAPABILITIES = [
    {
        "key": "customers",
        "glyph": "person_search",
        "tone": "",
        "title": "Customer lookup",
        "body": (
            "Retrieve customer profiles by ID and keep context across "
            "follow-up questions in the same conversation."
        ),
        "action": "Show C101's profile",
        "prompt": "Show me the profile of customer C101.",
    },
    {
        "key": "metrics",
        "glyph": "monitoring",
        "tone": "credit",
        "title": "Transaction metrics",
        "body": (
            "Debit, credit and net transaction flow calculated in Python "
            "from the transaction records, never estimated by the model."
        ),
        "action": "Get C101's metrics",
        "prompt": "What are C101's transaction metrics?",
    },
    {
        "key": "policies",
        "glyph": "policy",
        "tone": "violet",
        "title": "Policy answers",
        "body": (
            "Procedures from the fictional training policies, with "
            "citations validated against the documents actually retrieved."
        ),
        "action": "Ask about fraud reporting",
        "prompt": "What should a customer do about an unauthorized transaction?",
    },
]


def _activity_period():
    dates = sorted(
        date.fromisoformat(row["transaction_date"])
        for row in transactions
        if row.get("transaction_date")
    )

    if not dates:
        return "—"

    first, last = dates[0], dates[-1]

    # Day numbers are formatted manually: "%-d" is not portable.
    if first == last:
        return f"{first.day} {first.strftime('%b %Y')}"

    if (first.year, first.month) == (last.year, last.month):
        return f"{first.day}–{last.day} {last.strftime('%b %Y')}"

    return (
        f"{first.day} {first.strftime('%b')} – "
        f"{last.day} {last.strftime('%b %Y')}"
    )


# ---------- Hero ----------

ui.eyebrow("AI banking copilot")

ui.html(
    '<h1 class="bm-display">Banking answers,<br>'
    "<em>grounded in your data.</em></h1>"
    '<p class="bm-lede">Ask BankMate about customers, transaction activity '
    "or bank procedures. It retrieves records with live tool calls and cites "
    "the policy documents behind every policy answer.</p>"
)

# ---------- Conversational entry point ----------

with st.container(key="bm-composer"):
    typed = st.chat_input(
        "Ask about a customer, a transaction or a policy…",
        key="home-composer-input",
        max_chars=MAX_QUESTION_CHARS,
    )

    ui.html('<div class="bm-composer-hint">Try asking</div>')

    with st.container(horizontal=True, gap="small", key="bm-chiprow"):
        chosen = None

        for key, prompt in SUGGESTED_PROMPTS:
            if ui.prompt_chip(prompt, key):
                chosen = prompt

if typed:
    ui.go_to_assistant(typed)

if chosen:
    ui.go_to_assistant(chosen)

# ---------- Capabilities ----------

ui.section_label("What BankMate can do")

columns = st.columns(len(CAPABILITIES), gap="medium")

for column, capability in zip(columns, CAPABILITIES):
    with column:
        clicked = ui.capability_card(
            capability["key"],
            capability["glyph"],
            capability["title"],
            capability["body"],
            capability["action"],
            tone=capability["tone"],
        )

        if clicked:
            ui.go_to_assistant(capability["prompt"])

# ---------- Workspace snapshot (live from the data files) ----------

ui.section_label("Workspace")

ui.stat_strip([
    (len(customers), "Customers"),
    (len(transactions), "Transactions"),
    (len(list(POLICY_DIR.glob("*.txt"))), "Policy documents"),
    (_activity_period(), "Activity period"),
])

ui.footnote(
    "Demonstration workspace: customer and transaction records are synthetic, "
    "and policy documents are fictional training material."
)
