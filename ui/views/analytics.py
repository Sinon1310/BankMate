"""
Transaction Analytics over data/transactions.csv.

One filter row scopes everything below it: KPIs, charts and the
transaction table (which doubles as the table view for the charts).
"""

import streamlit as st

from ui import charts
from ui import components as ui
from ui import data


ALL = "All"
TYPE_OPTIONS = [ALL, data.DEBIT, data.CREDIT]

NOT_A_BALANCE = (
    "Net transaction flow is credits minus debits for the transactions "
    "shown. It is not an account balance — the data has no opening balances."
)

customers = data.all_customers()
names = {c["customer_id"]: c["customer_name"] for c in customers}
first_day, last_day = data.date_bounds()


ui.page_header(
    "Transaction Analytics",
    "Spending, inflows and net transaction flow across the recorded "
    "transactions. Adjust the filters to focus on customers, types or dates.",
)

if first_day is None:
    ui.empty_state("receipt_long", "No transactions", "data/transactions.csv has no records.")
    st.stop()

# ---------- Filters (one row, scopes the whole page) ----------

with st.container(key="bm-panel-filters"):
    customer_col, type_col, date_col = st.columns([2.2, 1.4, 1.6], gap="medium", vertical_alignment="top")

    with customer_col:
        selected_customers = st.multiselect(
            "Customers",
            options=list(names),
            format_func=lambda cid: f"{names[cid]} · {cid}",
            placeholder="All customers",
            key="analytics-customers",
        )

    with type_col:
        selected_type = st.segmented_control(
            "Type",
            TYPE_OPTIONS,
            default=ALL,
            key="analytics-type",
        )

    with date_col:
        picked = st.date_input(
            "Dates",
            value=(first_day, last_day),
            min_value=first_day,
            max_value=last_day,
            format="DD/MM/YYYY",
            key="analytics-dates",
        )

# A range picker returns one date while the user is mid-selection.
if isinstance(picked, (tuple, list)):
    start = picked[0] if len(picked) > 0 else first_day
    end = picked[1] if len(picked) > 1 else start
else:
    start = end = picked

rows = data.filter_transactions(
    customer_ids=selected_customers or None,
    types=None if selected_type in (None, ALL) else [selected_type],
    start=start,
    end=end,
)

if not rows:
    ui.empty_state(
        "filter_alt_off",
        "No transactions match these filters",
        "Widen the date range or include more customers or types.",
    )
    st.stop()

# ---------- KPIs ----------

summary = data.flow_summary(rows)

ui.kpi_tiles([
    {
        "label": "Total debits",
        "value": data.format_inr(summary["total_debit"]),
        "note": f"{summary['debit_count']} transactions",
        "swatch": "debit",
    },
    {
        "label": "Total credits",
        "value": data.format_inr(summary["total_credit"]),
        "note": f"{summary['credit_count']} transactions",
        "swatch": "credit",
    },
    {
        "label": "Net transaction flow",
        "value": data.format_inr(summary["net_flow"], signed=True),
        "note": "Credits minus debits",
    },
    {
        "label": "Transactions",
        "value": summary["count"],
        "note": f"{data.format_day(start)} – {data.format_day(end)}",
    },
])
ui.callout(NOT_A_BALANCE)

# ---------- Charts ----------

left, right = st.columns(2, gap="medium")

with left:
    with st.container(key="bm-panel-merchants"):
        ui.panel_heading("Debit spend by merchant", "Total debits per merchant, largest first")
        merchants = data.debit_by_merchant(rows)

        if merchants:
            st.altair_chart(charts.merchant_bars(merchants), use_container_width=True, theme=None)
        else:
            ui.empty_state("shopping_bag", "No debits", "There are no debit transactions in this selection.")

with right:
    with st.container(key="bm-panel-daily"):
        ui.panel_heading("Daily inflow and outflow", "Credits above zero, debits below")
        st.altair_chart(charts.daily_flow_columns(data.daily_flow(rows)), use_container_width=True, theme=None)

with st.container(key="bm-panel-net"):
    ui.panel_heading(
        "Net transaction flow by customer",
        "Credits minus debits for each customer in this selection — not a balance",
    )
    st.altair_chart(charts.net_flow_bars(data.net_flow_by_customer(rows)), use_container_width=True, theme=None)

# ---------- Table view ----------

ui.section_label("Transactions")
ui.transactions_table(
    data.sort_newest_first(rows),
    customer_names=names,
    caption="Transactions matching the filters, newest first",
)
