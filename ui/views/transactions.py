"""
Transaction explorer: filter every transaction by date range, customer,
account, direction, category, merchant and minimum amount, with paginated
results (never the whole dataset at once) and app.ledger totals over all
matches. Filtering uses ledger.select_transactions, the same rule as the
assistant's transaction tool.
"""

from decimal import Decimal, InvalidOperation

import streamlit as st

from app.tool_args import CATEGORIES
from ui import insights
from ui import components as ui
from ui.data import format_day, format_inr

PAGE_SIZES = [25, 50, 100]

ui.page_header("Transactions", "Search and filter posted transactions across every customer and account.",
               kicker="Intelligence")

if not insights.available():
    ui.empty_state("database", "Expanded dataset required", "This view needs the v1 dataset.")
    st.stop()

start, end = insights.window()
customers = insights.customer_options()
labels = {cid: f"{name} · {cid}" for cid, name, _seg in customers}

with st.container(key="bm-panel-txn-filters"):
    c1, c2, c3 = st.columns([1.6, 1.4, 1.2])
    merchant = c1.text_input("Merchant", placeholder="Search merchant, e.g. QuickBite", icon=":material/search:",
                             key="txn-merchant", max_chars=40)
    dates = c2.date_input("Date range", value=(start, end), min_value=start, max_value=end, key="txn-dates",
                          format="DD/MM/YYYY")
    customer = c3.selectbox("Customer", ["All customers"] + [cid for cid, _n, _s in customers],
                            format_func=lambda v: labels.get(v, v), key="txn-customer")
    c4, c5, c6, c7 = st.columns([1.2, 1, 1.3, 1])
    account_choices = ["All accounts"] + [aid for aid, _t, _c in insights.account_options(
        None if customer == "All customers" else customer)]
    account = c4.selectbox("Account", account_choices, key="txn-account")
    kind = c5.selectbox("Type", ["All", "Debit", "Credit"], key="txn-type")
    category = c6.selectbox("Category", ["All"] + list(CATEGORIES), key="txn-category")
    min_text = c7.text_input("Minimum amount (₹)", placeholder="0", key="txn-min", max_chars=12)

# A half-selected date range (one date) is treated as that single day.
if isinstance(dates, (tuple, list)):
    first_day, last_day = (dates[0], dates[-1]) if dates else (start, end)
else:
    first_day = last_day = dates

min_amount, invalid_min = None, False
if min_text.strip():
    try:
        min_amount = Decimal(min_text.replace(",", "").strip())
        invalid_min = min_amount < 0
    except InvalidOperation:
        invalid_min = True

if invalid_min:
    ui.callout("Minimum amount must be a positive number, e.g. 5000.", glyph="error")
    st.stop()

filters = {
    "customer_ids": None if customer == "All customers" else [customer],
    "account_id": None if account == "All accounts" else account,
    "start": first_day, "end": last_day,
    "transaction_type": None if kind == "All" else kind,
    "category": None if category == "All" else category,
    "merchant": merchant.strip() or None,
    "min_amount": min_amount,
}
signature = tuple(sorted((k, str(v)) for k, v in filters.items()))
size_key = "txn-page-size"
page_size = st.session_state.get(size_key, PAGE_SIZES[0])

summary = insights.transactions_page(limit=0, **filters)
totals = summary["totals"]

ui.kpi_tiles([
    {"label": "Matching transactions", "value": f"{summary['total']:,}",
     "note": f"{format_day(first_day)} – {format_day(last_day)}"},
    {"label": "Debits", "value": format_inr(totals["total_debits"]), "swatch": "debit",
     "note": f"{totals['debit_count']:,} transactions"},
    {"label": "Credits", "value": format_inr(totals["total_credits"]), "swatch": "credit",
     "note": f"{totals['credit_count']:,} transactions, refunds {format_inr(totals['refund_total'])}"},
    {"label": "Net transaction flow", "value": format_inr(totals["net_cash_flow"], signed=True),
     "note": "Credits minus debits; not a balance"},
])

with st.container(key="bm-panel-txn-results"):
    if not summary["total"]:
        ui.empty_state("search_off", "No matching transactions",
                       "No transactions in the dataset match these filters. Try widening the date range or "
                       "clearing a filter.")
    else:
        offset = ui.pager("transactions", summary["total"], page_size, signature)
        page = insights.transactions_page(limit=page_size, offset=offset, **filters)
        names = {cid: name for cid, name, _s in customers}
        ui.ledger_table(page["rows"], customer_names=names, card_last4=insights.card_last4)
        st.selectbox("Rows per page", PAGE_SIZES, key=size_key)
