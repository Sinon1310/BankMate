"""
Overview: portfolio KPIs, monthly activity, segment mix, recent activity,
dispute overview and a conversational entry point to the assistant.

Every figure is computed from the loaded dataset by app.ledger (through
ui.insights). Every control leads somewhere real: chips and the composer
open the AI Assistant and send the question; shortcuts open pages.
"""

import streamlit as st

from app.security import MAX_QUESTION_CHARS
from ui import charts, insights, navigation
from ui import components as ui
from ui.data import format_day, format_inr

# Customer IDs are typed into each prompt: the assistant only looks up
# customers the user names (see app/security.py).
SUGGESTED_PROMPTS = [
    ("balances", "Show C101's account balances"),
    ("monthly", "Analyse C104's monthly spending"),
    ("refund", "How are refunds handled for duplicate transactions?"),
    ("dispute", "What is the status of C111's dispute?"),
]


def _period_text():
    start, end = insights.window()
    return f"{format_day(start)} – {format_day(end)}"


ui.page_header(
    "Customer intelligence overview",
    f"Synthetic portfolio activity, {_period_text()}. Balances are as of {format_day(insights.as_of())}.",
    kicker="Overview",
)

if not insights.available():
    ui.empty_state("database", "Expanded dataset required",
                   "This view needs the v1 dataset with accounts, cards and disputes.")
    st.stop()

data = insights.overview()
f = data["figures"]

# ---------- Portfolio KPIs ----------

ui.kpi_tiles([
    {"label": "Customers", "value": f"{data['customers']:,}", "note": f"{len(insights.segment_summary())} segments"},
    {"label": "Accounts", "value": f"{data['accounts']:,}", "note": "Savings and current"},
    {"label": "Debit cards", "value": f"{data['cards']:,}", "note": f"{data['active_cards']} active"},
    {"label": "Transactions", "value": f"{data['transactions']:,}", "note": "Posted in the period"},
    {"label": "Disputes", "value": f"{data['disputes']:,}", "note": f"{data['open_disputes']} open or under investigation"},
])

ui.kpi_tiles([
    {"label": "Total balance", "value": format_inr(data["total_balance"]),
     "note": f"All accounts, as of {format_day(insights.as_of())}", "swatch": "primary"},
    {"label": "Credits", "value": format_inr(f["total_credits"]), "note": "Includes refunds once",
     "swatch": "credit"},
    {"label": "Debits", "value": format_inr(f["total_debits"]), "note": "Includes transfers out", "swatch": "debit"},
    {"label": "Net transaction flow", "value": format_inr(f["net_cash_flow"], signed=True),
     "note": "Credits minus debits; not a balance"},
])

# ---------- Activity and segments ----------

left, right = st.columns([3, 2], gap="medium")

with left:
    with st.container(key="bm-panel-monthly"):
        ui.panel_heading("Monthly credits and debits", "Posted transactions per calendar month (November to date)")
        months = insights.monthly()
        if months:
            st.altair_chart(charts.monthly_flow(months), use_container_width=True, theme=None)
        else:
            ui.empty_state("bar_chart", "No activity", "There are no transactions in the dataset.")

with right:
    with st.container(key="bm-panel-segments"):
        ui.panel_heading("Balance by segment", f"Total account balances as of {format_day(insights.as_of())}")
        segments = insights.segment_summary()
        st.altair_chart(charts.ranked_bars([(f"{s} ({c})", bal, "Balance") for s, c, _a, bal in segments],
                                           height_per_row=40), use_container_width=True, theme=None)
        ui.html('<div class="bm-mini-legend">Segment (number of customers). Exact amounts on hover.</div>')

# ---------- Recent activity and disputes ----------

left, right = st.columns([3, 2], gap="medium")

with left:
    with st.container(key="bm-panel-recent"):
        ui.panel_heading("Recent transactions", "The latest posted activity across all customers")
        recent = insights.transactions_page(limit=6)
        names = {cid: name for cid, name, _seg in insights.customer_options()}
        ui.ledger_table(recent["rows"], customer_names=names, card_last4=insights.card_last4, compact=True)
        st.page_link(navigation.TRANSACTIONS["path"], label="Open the transaction explorer",
                     icon=":material/arrow_forward:")

with right:
    with st.container(key="bm-panel-disputes"):
        ui.panel_heading("Disputes", "Card disputes by status (policy POL-001 statuses)")
        counts = insights.dispute_counts()
        ui.html('<div class="bm-status-grid">' + "".join(
            f'<div class="bm-status-cell"><div class="bm-status-count">{n}</div>{ui.badge(status)}</div>'
            for status, n in counts.items()) + "</div>")
        open_cases = [d for d in insights.disputes() if d["status"] in insights.OPEN_STATUSES][:3]
        if open_cases:
            ui.html('<div class="bm-mini-list">' + "".join(
                f'<div class="bm-mini-row"><span><strong>{d["dispute_id"]}</strong> · {d["customer_name"]} '
                f'<span class="bm-muted">{d["customer_id"]}</span><br><span class="bm-muted">{d["reason"]} · '
                f'{d["merchant"]}</span></span><span class="bm-num">{format_inr(d["amount"])}</span></div>'
                for d in open_cases) + "</div>")
        st.page_link(navigation.DISPUTES["path"], label="Review all disputes", icon=":material/arrow_forward:")

# ---------- Ask BankMate ----------

with st.container(key="bm-panel-ask"):
    ui.panel_heading("Ask BankMate", "Questions about a customer (by ID), transactions or bank procedures")
    with st.container(key="bm-composer"):
        typed = st.chat_input(
            "Ask about a customer, a transaction or a policy…",
            key="home-composer-input",
            max_chars=MAX_QUESTION_CHARS,
        )
        with st.container(horizontal=True, gap="small", key="bm-chiprow"):
            chosen = None
            for key, prompt in SUGGESTED_PROMPTS:
                if ui.prompt_chip(prompt, key):
                    chosen = prompt

if typed:
    ui.go_to_assistant(typed)

if chosen:
    ui.go_to_assistant(chosen)

ui.footnote(
    "Demonstration workspace: all customers, accounts, cards and transactions are synthetic, "
    "and policy documents are fictional training material. No real bank is represented."
)
