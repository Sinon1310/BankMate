"""
Disputes: card dispute cases by status (the four statuses defined in the
fictional POL-001 policy), with the disputed transaction, customer, reason,
dates and outcome. A refund is shown only where the data links an actual
refund transaction: a Resolved status is never treated as proof of one.
Dispute amounts are the disputed transactions' amounts, already counted
once in transaction totals.
"""

from html import escape

import streamlit as st

from ui import insights, navigation
from ui import components as ui
from ui.data import format_day, format_inr

ui.page_header("Disputes", "Card dispute cases, their status, outcome and any refund actually issued.",
               kicker="Intelligence")

if not insights.available():
    ui.empty_state("database", "Expanded dataset required", "This view needs the v1 dataset.")
    st.stop()

def closed_word(status):
    """Date label for a decided case: a Rejected dispute was closed, not resolved."""
    return "resolved" if status == "Resolved" else "closed"


counts = insights.dispute_counts()
all_cases = insights.disputes()
refunded = sum(1 for d in all_cases if d["refund"])

ui.kpi_tiles(
    [{"label": status, "value": str(n), "note": "Awaiting review" if status == "Open" else
      "Being investigated" if status == "Under Investigation" else
      f"{refunded} with a refund issued" if status == "Resolved" else "Not upheld"}
     for status, n in counts.items()]
)

with st.container(key="bm-panel-dispute-table"):
    status = st.radio("Status", ["All"] + list(insights.DISPUTE_STATUSES), key="dispute-status",
                      horizontal=True, label_visibility="collapsed")
    cases = insights.disputes(None if status in (None, "All") else status)
    if not cases:
        ui.empty_state("task_alt", "No disputes", "No disputes have this status.")
    else:
        body = "".join(
            "<tr>"
            f'<td><strong>{escape(d["dispute_id"])}</strong><div class="bm-muted bm-col-secondary">'
            f'filed {escape(format_day(d["raised_on"]))}</div>'
            f'<div class="bm-mobile-only bm-dispute-glance" aria-hidden="true">{ui.badge(d["status"])} '
            f'<span class="bm-num bm-nowrap">{escape(format_inr(d["amount"]))}</span></div></td>'
            f'<td>{escape(d["customer_name"])}<div class="bm-muted bm-col-secondary">{escape(d["customer_id"])}</div></td>'
            f'<td>{escape(d["merchant"])}<div class="bm-muted bm-col-secondary bm-num">{escape(d["transaction_id"])} · '
            f'{escape(format_day(d["transaction_date"]))}'
            + (f' · {ui.card_ref(d["card_last4"])}' if d["card_last4"] else "") + "</div></td>"
            f'<td class="bm-right bm-num">{escape(format_inr(d["amount"]))}</td>'
            f'<td>{escape(d["reason"])}</td>'
            f'<td>{ui.badge(d["status"])}'
            + (f'<div class="bm-muted bm-col-secondary">{closed_word(d["status"])} '
               f'{escape(format_day(d["resolved_on"]))}</div>' if d["resolved_on"] else "") + "</td>"
            f'<td>{escape(d["resolution_outcome"] or "—")}</td>'
            + (f'<td class="bm-num">{escape(format_inr(d["refund"]["amount"]))}<div class="bm-muted '
               f'bm-col-secondary">{escape(d["refund"]["transaction_id"])} · {escape(format_day(d["refund"]["date"]))}'
               "</div></td>" if d["refund"] else '<td class="bm-muted">None</td>')
            + "</tr>" for d in cases)
        ui.html('<div class="bm-table-wrap"><table class="bm-table bm-table-wide bm-table-sticky"><thead><tr>'
                '<th>Dispute</th><th>Customer</th><th>Disputed transaction</th><th class="bm-right">Amount</th>'
                '<th>Reason</th><th>Status</th><th>Outcome</th><th>Refund issued</th></tr></thead>'
                f"<tbody>{body}</tbody></table></div>{ui.SCROLL_HINT}")

    options = {d["customer_id"]: f'{d["customer_name"]} · {d["customer_id"]}' for d in cases}
    if options:
        pick, go = st.columns([3, 1.4], vertical_alignment="bottom")
        chosen = pick.selectbox("Open a customer", list(options), format_func=options.get, key="dispute-customer")
        if go.button("Open in Customer 360", icon=":material/person_search:", key="dispute-open-customer",
                     use_container_width=True):
            st.session_state["explorer_customer_id"] = chosen
            st.query_params["customer"] = chosen
            st.switch_page(navigation.CUSTOMERS["path"])

ui.caption("A Resolved dispute does not imply a refund: a refund is listed only when a refund transaction exists.")
ui.footnote("Dispute statuses follow the fictional BankMate card dispute policy (POL-001). Amounts are the "
            "disputed transactions' amounts, already counted once in transaction totals.")
