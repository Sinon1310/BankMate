"""
Customer 360: search and select a customer, then review their profile,
accounts with balances, debit cards (last four digits only), disputes,
activity and a paginated, filterable transaction history.

All figures come from the assistant's own customer tools (profile,
metrics, disputes, transactions), so this page and the assistant always
agree and nothing is recalculated in the UI. The selected customer is
mirrored to the ?customer= URL parameter.
"""

import json
from decimal import Decimal
from html import escape

import streamlit as st

from app import tools
from app.tool_args import CATEGORIES
from ui import charts, insights
from ui import components as ui
from ui import data
from ui.data import format_day, format_inr

SELECTED = "explorer_customer_id"
PAGE_SIZE = 10
NOT_A_BALANCE = ("Net transaction flow is credits minus debits for the period. "
                 "It is not an account balance.")


COMPACT_PICKER = "explorer-compact-select"


def _select(customer_id):
    st.session_state[SELECTED] = customer_id
    st.query_params["customer"] = customer_id


def _picked_compact():
    """Compact (mobile) selector changed: select that customer."""
    choice = st.session_state.get(COMPACT_PICKER)
    if choice:
        _select(choice)


def _tool(tool, **args):
    raw = tool.invoke(args)
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {"message": raw}


# ---------- Resolve the selected customer ----------

customers = data.all_customers()
known_ids = [c["customer_id"] for c in customers]

requested = st.query_params.get("customer")
if requested in known_ids and st.session_state.get(SELECTED) != requested:
    st.session_state[SELECTED] = requested
if st.session_state.get(SELECTED) not in known_ids and known_ids:
    st.session_state[SELECTED] = known_ids[0]
selected_id = st.session_state.get(SELECTED)

ui.page_header("Customer 360", "Profile, accounts, cards, disputes and activity for one customer at a time.",
               kicker="Intelligence")

if not customers:
    ui.empty_state("person_off", "No customers", "The dataset has no customer records.")
    st.stop()

list_column, detail_column = st.columns([1, 2.6], gap="medium")

# ---------- Customer list ----------

with list_column:
    with st.container(key="bm-panel-customer-list"):
        query = st.text_input("Search customers", placeholder="Search name, ID or city", icon=":material/search:",
                              key="explorer-search", max_chars=100, label_visibility="collapsed")
        matches = data.search_customers(query)
        if query and not matches:
            ui.html(f'<div class="bm-list-meta">No customers match “{escape(query)}”.</div>')
        else:
            ui.html(f'<div class="bm-list-meta">{len(matches)} customer{"" if len(matches) == 1 else "s"}</div>')

        # A fixed, scrolling height only for long lists (height=None is not accepted).
        scroll = {"height": 620} if len(matches) > 12 else {}
        with st.container(key="bm-customer-scroll", **scroll):
            for customer in matches:
                cid = customer["customer_id"]
                is_selected = cid == selected_id
                with st.container(key="bm-cust-active" if is_selected else f"bm-cust-{cid}"):
                    if st.button(f"{customer['customer_name']}  ·  {cid}", key=f"cust-btn-{cid}",
                                 icon=":material/person:", type="tertiary", use_container_width=True,
                                 disabled=is_selected):
                        _select(cid)
                        st.rerun()

# ---------- Customer detail ----------

with detail_column:
    # Narrow screens: a compact selector replaces the long list (CSS shows this
    # container and hides the list below the mobile breakpoint; desktop keeps
    # the list). Both drive the same selection.
    names = {c["customer_id"]: f'{c["customer_name"]} · {c["customer_id"]}' for c in customers}
    if st.session_state.get(COMPACT_PICKER) != selected_id:
        st.session_state[COMPACT_PICKER] = selected_id
    with st.container(key="bm-customer-picker-compact"):
        st.selectbox("Customer", known_ids, format_func=names.get, key=COMPACT_PICKER, on_change=_picked_compact)

    if not insights.available():
        customer = data.get_customer(selected_id)
        ui.html(f'<div class="bm-profile-name">{escape(customer["customer_name"])}</div>')
        ui.empty_state("database", "Expanded dataset required",
                       "Accounts, cards and disputes need the v1 dataset.")
        st.stop()

    profile = _tool(tools.get_customer_profile, customer_id=selected_id)
    metrics = _tool(tools.get_customer_metrics, customer_id=selected_id)
    disputes = _tool(tools.get_customer_disputes, customer_id=selected_id)
    has_activity = "total_debit" in metrics

    # Identity
    with st.container(key="bm-panel-profile"):
        head, action = st.columns([4, 1.4], vertical_alignment="center")
        with head:
            ui.html(
                '<div class="bm-profile">'
                f'<span class="bm-avatar" aria-hidden="true">{escape(data.initials(profile["customer_name"]))}</span>'
                '<div class="bm-profile-text">'
                f'<div class="bm-profile-name">{escape(profile["customer_name"])}</div>'
                '<div class="bm-profile-meta">'
                f'<span>{ui.icon("badge")}{escape(profile["customer_id"])}</span>'
                f'<span>{ui.icon("location_on")}{escape(profile["city"])}</span>'
                f'<span>{ui.icon("workspace_premium")}{escape(profile["segment"])}</span>'
                f'<span>{ui.icon("event")}Customer since {escape(format_day(profile["customer_since"]))}</span>'
                "</div></div></div>"
            )
        with action:
            if st.button("Ask BankMate", icon=":material/forum:", key="explorer-ask", use_container_width=True):
                ui.go_to_assistant(f"Give me an overview of customer {selected_id}: "
                                   "their accounts, balances and transaction metrics.")

    # KPIs
    tiles = [{"label": "Total balance", "value": format_inr(Decimal(profile["total_balance"])),
              "note": f"{len(profile['accounts'])} account(s), as of {format_day(profile['as_of_date'])}",
              "swatch": "primary"}]
    if has_activity:
        tiles += [
            {"label": "Credits", "value": format_inr(Decimal(metrics["total_credit"])), "swatch": "credit",
             "note": f"Income {format_inr(Decimal(metrics['income']))}"},
            {"label": "Debits", "value": format_inr(Decimal(metrics["total_debit"])), "swatch": "debit",
             "note": f"Spending {format_inr(Decimal(metrics['spending_gross']))}"},
            {"label": "Net transaction flow", "value": format_inr(Decimal(metrics["net_transaction_flow"]), signed=True),
             "note": f"{metrics['transaction_count']} transactions"},
        ]
    ui.kpi_tiles(tiles)
    if has_activity:
        ui.caption(f"{NOT_A_BALANCE} Period: {format_day(metrics['period']['start'])} – "
                    f"{format_day(metrics['period']['end'])}.")

    # Accounts and cards
    accounts_col, cards_col = st.columns([3, 2], gap="medium")
    with accounts_col:
        with st.container(key="bm-panel-accounts"):
            ui.panel_heading("Accounts", f"Balances as of {format_day(profile['as_of_date'])}")
            ui.html('<div class="bm-account-grid">' + "".join(
                '<div class="bm-account">'
                f'<div class="bm-account-top"><span class="bm-account-type">{escape(a["account_type"])}</span>'
                f'{ui.badge(a["status"], "success")}</div>'
                f'<div class="bm-account-balance bm-num">{escape(format_inr(Decimal(a["balance"])))}</div>'
                f'<div class="bm-account-meta"><span class="bm-num">{escape(a["account_id"])}</span>'
                f'<span>Opened {escape(format_day(a["opened_on"]))}</span></div></div>'
                for a in profile["accounts"]) + "</div>")
    with cards_col:
        with st.container(key="bm-panel-cards"):
            ui.panel_heading("Debit cards", "Last four digits only")
            if profile["cards"]:
                ui.html('<div class="bm-mini-list">' + "".join(
                    f'<div class="bm-mini-row"><span>{ui.icon("credit_card")} <strong>{escape(c["network"])} '
                    f'{ui.card_ref(c["card_last4"])}</strong><br><span class="bm-muted">{escape(c["account_id"])} · '
                    f'valid until {escape(format_day(c["valid_until"]))}'
                    + (f' · blocked {escape(format_day(c["blocked_on"]))}' if c.get("blocked_on") else "")
                    + (f' · {ui.card_ref(c["replaces_card_last4"], "replaces ")}' if c.get("replaces_card_last4") else "")
                    + f"</span></span>{ui.badge(c['status'])}</div>"
                    for c in profile["cards"]) + "</div>")
            else:
                ui.html('<div class="bm-list-meta">No debit cards on file.</div>')

    # Disputes
    with st.container(key="bm-panel-customer-disputes"):
        ui.panel_heading("Disputes", "Case records; a refund is shown only when one was actually issued")
        cases = disputes.get("disputes", [])
        if not cases:
            ui.html('<div class="bm-list-meta">No disputes for this customer.</div>')
        else:
            ui.html('<div class="bm-mini-list">' + "".join(
                f'<div class="bm-mini-row"><span><strong>{escape(d["dispute_id"])}</strong> · {escape(d["reason"])} · '
                f'{escape(d["merchant"])} · {escape(format_inr(Decimal(d["amount"])))}<br><span class="bm-muted">Filed '
                f'{escape(format_day(d["filed_on"]))}'
                + (f' · resolved {escape(format_day(d["resolved_on"]))}' if d.get("resolved_on") else "")
                + (f' · {escape(d["resolution_outcome"])}' if d.get("resolution_outcome") else "")
                + (f' · refund {escape(format_inr(Decimal(d["refund"]["amount"])))} on {escape(format_day(d["refund"]["date"]))}'
                   if d.get("refund") else "")
                + f"</span></span>{ui.badge(d['status'])}</div>"
                for d in cases) + "</div>")

    # Activity
    if has_activity:
        trend_col, category_col = st.columns([3, 2], gap="medium")
        with trend_col:
            with st.container(key="bm-panel-customer-monthly"):
                ui.panel_heading("Monthly activity", "Credits and debits per month")
                months = [{"month": m["month"], "figures": {"total_credits": Decimal(m["credits"]),
                                                            "total_debits": Decimal(m["debits"]),
                                                            "net_cash_flow": Decimal(m["net_transaction_flow"])}}
                          for m in metrics.get("monthly", [])]
                if months:
                    st.altair_chart(charts.monthly_flow(months, height=220), use_container_width=True, theme=None)
        with category_col:
            with st.container(key="bm-panel-customer-categories"):
                ui.panel_heading("Spending by category", "Net of refunds; transfers out excluded")
                top = [c for c in metrics.get("by_category", []) if Decimal(c["net"]) > 0][:6]
                if top:
                    st.altair_chart(charts.ranked_bars([(c["category"], Decimal(c["net"]), "Net") for c in top],
                                                       color=charts.TOKENS["debit"], height_per_row=30),
                                    use_container_width=True, theme=None)
                else:
                    ui.html('<div class="bm-list-meta">No spending in this period.</div>')

    # Transactions
    with st.container(key="bm-panel-customer-transactions"):
        ui.panel_heading("Transactions", "Newest first; filters run on the full history")
        c1, c2, c3 = st.columns([1, 1.3, 1.6])
        kind = c1.selectbox("Type", ["All", "Debit", "Credit"], key="explorer-type")
        category = c2.selectbox("Category", ["All"] + list(CATEGORIES), key="explorer-category")
        merchant = c3.text_input("Merchant", placeholder="Part of a merchant name", key="explorer-merchant",
                                 max_chars=40)
        args = {"customer_id": selected_id, "limit": PAGE_SIZE}
        if kind != "All":
            args["transaction_type"] = kind
        if category != "All":
            args["category"] = category
        if merchant.strip():
            args["merchant"] = merchant.strip()
        signature = (selected_id, kind, category, merchant.strip())
        first = _tool(tools.get_customer_transactions, **args)
        if first.get("status") == "INVALID_ARGUMENT":
            ui.callout(first["message"], glyph="error")
        elif not first.get("total_matching"):
            ui.html('<div class="bm-list-meta">No transactions match these filters.</div>')
        else:
            offset = ui.pager("customer-transactions", first["total_matching"], PAGE_SIZE, signature)
            page = first if offset == 0 else _tool(tools.get_customer_transactions, offset=offset, **args)
            index = {t["transaction_id"]: t for t in insights.dataset().tables["transactions"]}
            rows = [index[r[0]] for r in page["rows"]]
            ui.ledger_table(rows, card_last4=insights.card_last4)
            totals = first["matching_totals"]
            ui.caption(f"Totals for all {first['total_matching']} matching transactions: debits "
                        f"{format_inr(Decimal(totals['debits']))}, credits {format_inr(Decimal(totals['credits']))}.")
