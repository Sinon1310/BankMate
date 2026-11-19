"""
Customer Explorer: search and select customers from data/customers.csv,
then see their profile, metrics (from the backend get_customer_metrics
tool) and transaction history.

The selected customer is mirrored to the ?customer= URL parameter, so a
customer view can be bookmarked or linked to.
"""

from html import escape

import streamlit as st

from ui import components as ui
from ui import data


SELECTED = "explorer_customer_id"
NOT_A_BALANCE = (
    "Net transaction flow is total credits minus total debits for the "
    "recorded transactions. It is not an account balance."
)


def _select(customer_id):
    st.session_state[SELECTED] = customer_id
    st.query_params["customer"] = customer_id


# ---------- Resolve the selected customer ----------

customers = data.all_customers()
known_ids = [c["customer_id"] for c in customers]

requested = st.query_params.get("customer")
if requested in known_ids and st.session_state.get(SELECTED) != requested:
    st.session_state[SELECTED] = requested

if st.session_state.get(SELECTED) not in known_ids and known_ids:
    st.session_state[SELECTED] = known_ids[0]

selected_id = st.session_state.get(SELECTED)


# ---------- Page ----------

ui.page_header(
    "Customer Explorer",
    "Search the customer records and review each customer's profile, "
    "transaction metrics and history.",
)

if not customers:
    ui.empty_state("person_off", "No customers", "data/customers.csv has no records.")
    st.stop()

list_column, detail_column = st.columns([1, 2.2], gap="large")

# ---------- Customer list ----------

with list_column:
    with st.container(key="bm-panel-customer-list"):
        query = st.text_input(
            "Search customers",
            placeholder="Search name, ID or city",
            icon=":material/search:",
            key="explorer-search",
            max_chars=100,
            label_visibility="collapsed",
        )

        matches = data.search_customers(query)

        if query and not matches:
            ui.html(
                f'<div class="bm-list-meta">No customers match “{escape(query)}”.</div>'
            )
        else:
            noun = "customer" if len(matches) == 1 else "customers"
            ui.html(f'<div class="bm-list-meta">{len(matches)} {noun}</div>')

        for customer in matches:
            cid = customer["customer_id"]
            is_selected = cid == selected_id
            key = "bm-cust-active" if is_selected else f"bm-cust-{cid}"

            with st.container(key=key):
                if st.button(
                    f"{customer['customer_name']}  ·  {cid}",
                    key=f"cust-btn-{cid}",
                    icon=":material/person:",
                    type="tertiary",
                    use_container_width=True,
                    disabled=is_selected,
                ):
                    _select(cid)
                    st.rerun()

# ---------- Customer detail ----------

customer = data.get_customer(selected_id)

with detail_column:
    with st.container(key="bm-panel-profile"):
        ui.html(
            '<div class="bm-profile">'
            f'<span class="bm-avatar" aria-hidden="true">{escape(data.initials(customer["customer_name"]))}</span>'
            '<div class="bm-profile-text">'
            f'<div class="bm-profile-name">{escape(customer["customer_name"])}</div>'
            '<div class="bm-profile-meta">'
            f'<span>{ui.icon("badge")}{escape(customer["customer_id"])}</span>'
            f'<span>{ui.icon("location_on")}{escape(customer.get("city") or "—")}</span>'
            f'<span>{ui.icon("account_balance")}{escape(customer.get("account_type") or "—")} account</span>'
            "</div></div></div>"
        )

        if st.button(
            "Ask BankMate about this customer",
            icon=":material/forum:",
            key="explorer-ask",
        ):
            ui.go_to_assistant(
                f"Give me an overview of customer {customer['customer_id']}: "
                "their profile and transaction metrics."
            )

    metrics = data.customer_metrics(selected_id)

    if metrics is None:
        ui.empty_state(
            "receipt_long",
            "No transactions yet",
            "This customer has no recorded transactions, so there are no metrics to show.",
        )
    else:
        ui.kpi_tiles([
            {"label": "Total debits", "value": data.format_inr(metrics["total_debit"]), "swatch": "debit"},
            {"label": "Total credits", "value": data.format_inr(metrics["total_credit"]), "swatch": "credit"},
            {"label": "Net transaction flow", "value": data.format_inr(metrics["net_flow"], signed=True)},
            {"label": "Transactions", "value": metrics["count"]},
        ])
        ui.callout(NOT_A_BALANCE)

        ui.section_label("Transaction history")
        ui.transactions_table(
            data.customer_transactions(selected_id),
            caption=f"Transactions for {customer['customer_name']}, newest first",
        )
