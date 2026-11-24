"""
Analytics: portfolio or customer-level figures for a date range.
Aggregates first (app.ledger via ui.insights): credits and debits, income
and spending, gross vs net spending, refunds, transfers, monthly cash flow,
month-end balances, category spending and account summaries. Transaction
details are paginated; the whole dataset is never rendered at once.

Balance, net transaction flow and net spending are always labelled as
different figures.
"""

import streamlit as st

from ui import charts, insights, navigation
from ui import components as ui
from ui.data import format_day, format_inr

PAGE_SIZE = 25


def month_name(month):
    """'2026-05' -> 'May 2026'."""
    from datetime import date
    return date(int(month[:4]), int(month[5:]), 1).strftime("%b %Y")

ui.page_header("Analytics", "Cash flow, spending and balances for any period, across all customers or a selection.",
               kicker="Intelligence")

if not insights.available():
    ui.empty_state("database", "Expanded dataset required", "This view needs the v1 dataset.")
    st.stop()

start, end = insights.window()
customers = insights.customer_options()
labels = {cid: f"{name} · {cid}" for cid, name, _s in customers}

with st.container(key="bm-panel-filters"):
    left, right = st.columns([1.2, 2])
    dates = left.date_input("Period", value=(start, end), min_value=start, max_value=end, key="analytics-dates",
                            format="DD/MM/YYYY")
    selected = right.multiselect("Customers", [cid for cid, _n, _s in customers], format_func=labels.get,
                                 key="analytics-customers", placeholder="All customers")

if isinstance(dates, (tuple, list)):
    first_day, last_day = (dates[0], dates[-1]) if dates else (start, end)
else:
    first_day = last_day = dates
scope = selected or None
scope_text = "all customers" if not selected else f"{len(selected)} selected customer(s)"

f = insights.figures(first_day, last_day, scope)
opening, closing = insights.balances_for(first_day, last_day, scope)

if not f["count"]:
    ui.empty_state("query_stats", "No transactions in this selection",
                   "Choose a wider period or different customers.")
    st.stop()

# ---------- Headline figures ----------

ui.section_header("Cash flow", f"{format_day(first_day)} – {format_day(last_day)} · {scope_text}")
ui.kpi_tiles([
    {"label": "Credits", "value": format_inr(f["total_credits"]), "swatch": "credit",
     "note": f"{f['credit_count']:,} transactions"},
    {"label": "Debits", "value": format_inr(f["total_debits"]), "swatch": "debit",
     "note": f"{f['debit_count']:,} transactions"},
    {"label": "Net transaction flow", "value": format_inr(f["net_cash_flow"], signed=True),
     "note": "Credits minus debits; not a balance"},
    {"label": "Closing balance", "value": format_inr(closing), "swatch": "primary",
     "note": f"Opening {format_inr(opening)} · end of {format_day(last_day)}"},
])

ui.section_header("Income and spending", "Refunds are credits but never income; transfers are shown separately")
ui.kpi_tiles([
    {"label": "Income", "value": format_inr(f["income"]), "note": "Salary, pension, business income, interest"},
    {"label": "Spending (gross)", "value": format_inr(f["spending_gross"]),
     "note": "Debits excluding transfers out; cash included"},
    {"label": "Refunds", "value": format_inr(f["refund_total"]), "note": "Reduce spending in the original category"},
    {"label": "Spending (net)", "value": format_inr(f["spending_net"]), "note": "Gross spending minus refunds"},
])
ui.kpi_tiles([
    {"label": "Transfers in", "value": format_inr(f["other_inflows"]), "note": "Not income"},
    {"label": "Transfers out", "value": format_inr(f["transfer_out_total"]),
     "note": "SIPs, EMIs, insurance, family, suppliers; not spending"},
])

# ---------- Trends ----------

months = insights.monthly(first_day, last_day, scope)
trend, balance = st.columns([3, 2], gap="medium")
with trend:
    with st.container(key="bm-panel-cashflow"):
        ui.panel_heading("Monthly cash flow", "Credits and debits per month (partial months are clipped to the period)")
        st.altair_chart(charts.monthly_flow(months), use_container_width=True, theme=None)
with balance:
    with st.container(key="bm-panel-balance-trend"):
        ui.panel_heading("Month-end balance", "Total of the selection's accounts")
        st.altair_chart(charts.balance_trend(months), use_container_width=True, theme=None)

with st.container(key="bm-panel-monthly-table"):
    ui.panel_heading("Monthly summary", "Net flow is credits minus debits; closing balance is at month end")
    ui.html('<div class="bm-table-wrap"><table class="bm-table bm-table-compact bm-table-sticky"><thead><tr>'
            '<th>Month</th><th class="bm-right">Credits</th><th class="bm-right">Debits</th>'
            '<th class="bm-right">Net flow</th><th class="bm-right">Refunds</th><th class="bm-right">Closing balance</th>'
            '</tr></thead><tbody>' + "".join(
                f'<tr><td class="bm-nowrap">{month_name(m["month"])}</td>'
                f'<td class="bm-right bm-num">{format_inr(m["figures"]["total_credits"])}</td>'
                f'<td class="bm-right bm-num">{format_inr(m["figures"]["total_debits"])}</td>'
                f'<td class="bm-right bm-num">{format_inr(m["figures"]["net_cash_flow"], signed=True)}</td>'
                f'<td class="bm-right bm-num">{format_inr(m["figures"]["refund_total"])}</td>'
                f'<td class="bm-right bm-num">{format_inr(m["closing_balance"])}</td></tr>' for m in months)
            + f"</tbody></table></div>{ui.SCROLL_HINT}")

# ---------- Categories ----------

cats = insights.categories(f)
category_col, refund_col = st.columns([1, 1], gap="medium")
with category_col:
    with st.container(key="bm-panel-categories"):
        ui.panel_heading("Spending by category", "Net of refunds · transfers out excluded")
        shown = [c for c in cats if c[3] > 0][:10]
        if shown:
            st.altair_chart(charts.ranked_bars([(c, net, "Net") for c, _g, _r, net in shown],
                                               color=charts.TOKENS["debit"]), use_container_width=True, theme=None)
with refund_col:
    with st.container(key="bm-panel-refunds"):
        ui.panel_heading("Gross vs net by category", "Where refunds reduced spending")
        ui.html('<div class="bm-table-wrap"><table class="bm-table bm-table-wide bm-table-sticky"><thead><tr><th>Category</th>'
                '<th class="bm-right">Gross</th><th class="bm-right">Refunds</th><th class="bm-right">Net</th>'
                '</tr></thead><tbody>' + "".join(
                    f'<tr><td>{c}</td><td class="bm-right bm-num">{format_inr(g)}</td>'
                    f'<td class="bm-right bm-num">{format_inr(r) if r else "—"}</td>'
                    f'<td class="bm-right bm-num">{format_inr(n)}</td></tr>' for c, g, r, n in cats)
                + f"</tbody></table></div>{ui.SCROLL_HINT}")

# ---------- Accounts ----------

with st.container(key="bm-panel-accounts-summary"):
    accounts = sorted(insights.account_summaries(first_day, last_day, scope), key=lambda a: -a["closing"])
    ui.panel_heading("Accounts", f"{len(accounts)} accounts · flows for the period and balances at its end")
    shown = accounts if len(accounts) <= 12 else accounts[:12]
    ui.html('<div class="bm-table-wrap"><table class="bm-table bm-table-compact bm-table-sticky"><thead><tr><th>Account</th>'
            '<th>Customer</th><th>Type</th><th class="bm-right">Credits</th><th class="bm-right">Debits</th>'
            '<th class="bm-right">Net flow</th><th class="bm-right">Closing balance</th></tr></thead><tbody>' + "".join(
                f'<tr><td class="bm-num bm-nowrap">{a["account_id"]}</td><td>{a["customer_name"]} '
                f'<span class="bm-muted bm-nowrap">{a["customer_id"]}</span></td><td>{a["account_type"]}</td>'
                f'<td class="bm-right bm-num">{format_inr(a["credits"])}</td>'
                f'<td class="bm-right bm-num">{format_inr(a["debits"])}</td>'
                f'<td class="bm-right bm-num">{format_inr(a["net"], signed=True)}</td>'
                f'<td class="bm-right bm-num">{format_inr(a["closing"])}</td></tr>' for a in shown)
            + f"</tbody></table></div>{ui.SCROLL_HINT}")
    if len(accounts) > len(shown):
        ui.caption(f"Showing the {len(shown)} largest accounts by closing balance of {len(accounts)}.")

# ---------- Transaction details (paginated) ----------

with st.container(key="bm-panel-analytics-transactions"):
    ui.panel_heading("Transaction details", "Newest first, 25 per page")
    first = insights.transactions_page(customer_ids=scope, start=first_day, end=last_day, limit=PAGE_SIZE)
    offset = ui.pager("analytics-transactions", first["total"], PAGE_SIZE, (first_day, last_day, tuple(selected)))
    page = first if offset == 0 else insights.transactions_page(customer_ids=scope, start=first_day, end=last_day,
                                                                limit=PAGE_SIZE, offset=offset)
    names = {cid: name for cid, name, _s in customers}
    ui.ledger_table(page["rows"], customer_names=names, card_last4=insights.card_last4)
    st.page_link(navigation.TRANSACTIONS["path"], label="More filters in the transaction explorer",
                 icon=":material/tune:")
