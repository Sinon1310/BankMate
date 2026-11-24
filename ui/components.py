"""
Reusable BankMate UI components.

Every page builds from these so spacing, typography and interaction
patterns stay consistent. Dynamic text is HTML-escaped before it is
placed inside custom markup.
"""

from html import escape

import streamlit as st

from app.citations import display_case
from ui import navigation, state


# ---------- Primitives ----------

def icon(name, extra_class=""):
    """Material Symbols glyph (font bundled with Streamlit)."""

    classes = ("bm-icon " + extra_class).strip()
    return f'<span class="{classes}" aria-hidden="true">{escape(name)}</span>'


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


# ---------- Page structure ----------

def eyebrow(text):
    html(
        f'<span class="bm-eyebrow"><span class="bm-eyebrow-dot"></span>'
        f"{escape(text)}</span>"
    )


def page_header(title, subtitle=None, kicker=None):
    parts = ['<div class="bm-page-header">']

    if kicker:
        parts.append(
            f'<span class="bm-eyebrow"><span class="bm-eyebrow-dot"></span>'
            f"{escape(kicker)}</span>"
        )

    parts.append(f'<h1 class="bm-page-title">{escape(title)}</h1>')

    if subtitle:
        parts.append(f'<p class="bm-page-subtitle">{escape(subtitle)}</p>')

    parts.append("</div>")
    html("".join(parts))


def section_label(text):
    html(f'<div class="bm-section-label">{escape(text)}</div>')


def footnote(text):
    html(f'<div class="bm-footnote">{escape(text)}</div>')


def caption(text):
    """A small muted note inside a panel or under a KPI row (no divider)."""
    html(f'<div class="bm-caption">{escape(text)}</div>')


def stat_strip(items):
    """items: iterable of (value, label)."""

    cells = "".join(
        f'<div class="bm-stat"><div class="bm-stat-value">{escape(str(value))}</div>'
        f'<div class="bm-stat-label">{escape(label)}</div></div>'
        for value, label in items
    )
    html(f'<div class="bm-stats">{cells}</div>')


# ---------- Navigation ----------

def go_to_assistant(prompt=None):
    """
    Open the assistant. With a prompt, it is sent on arrival in a new
    conversation, so entry points never append to an unrelated thread.
    """

    if prompt:
        state.queue_prompt(prompt, new_conversation=True)

    st.switch_page(navigation.ASSISTANT["path"])


def go_to_policy(policy_id):
    """
    Open a policy's detail view. Uses session state + switch_page (not a
    URL link) so the current browser session and conversations survive.
    """

    state.select_policy(policy_id)
    st.switch_page(navigation.POLICIES["path"])


def sidebar_note(title, body):
    with st.sidebar:
        html(
            f'<div class="bm-sidebar-note"><strong>{escape(title)}</strong>'
            f"{escape(body)}</div>"
        )


def mobile_nav(current_title):
    """
    Horizontal page links, shown only below the mobile breakpoint.
    The current page is rendered highlighted and non-clickable.
    """

    with st.container(horizontal=True, key="bm-mobile-nav"):
        for page in navigation.PAGES:
            is_current = page["title"] == current_title
            key = "bm-navlink-active" if is_current else f"bm-navlink-{page['url_path']}"

            with st.container(key=key):
                st.page_link(
                    page["path"],
                    label=page.get("short", page["title"]),
                    icon=page["icon"],
                    disabled=is_current,
                )


# ---------- Interactive building blocks ----------

def prompt_chip(label, key):
    """Pill-shaped suggestion. Returns True when clicked."""

    with st.container(key=f"bm-chip-{key}"):
        return st.button(label, key=f"chip-btn-{key}")


def capability_card(key, glyph, title, body, action_label, tone=""):
    """Card with icon tile, copy and one real action. Returns clicked."""

    with st.container(key=f"bm-card-{key}"):
        html(
            f'<span class="bm-icon-tile {escape(tone)}">{icon(glyph)}</span>'
            f'<div class="bm-card-title">{escape(title)}</div>'
            f'<p class="bm-card-body">{escape(body)}</p>'
        )

        with st.container(key=f"bm-link-{key}"):
            return st.button(
                f"{action_label} :material/arrow_forward:",
                key=f"card-btn-{key}",
                type="tertiary",
            )


# ---------- Assistant ----------

SOURCES_LABEL = "Sources · fictional BankMate training policies"


def sources_block(citations):
    """
    Policy-level citations returned by ask_bankmate_with_sources.
    Fields: policy_id, policy_name, version, source_files (any may be
    empty). Renders nothing when there are no citations.
    """

    if not citations:
        return

    rows = []

    for citation in citations:
        policy_id = escape(citation.get("policy_id", ""))
        name = escape(display_case(citation.get("policy_name") or "")) or policy_id

        meta = []
        if citation.get("version"):
            meta.append("Version " + escape(citation["version"]))
        meta.extend(escape(f) for f in citation.get("source_files") or [])

        rows.append(
            '<div class="bm-source">'
            f'<span class="bm-source-id">{policy_id}</span>'
            '<div class="bm-source-text">'
            f'<div class="bm-source-title">{name}</div>'
            + (f'<div class="bm-source-meta">{" · ".join(meta)}</div>' if meta else "")
            + "</div></div>"
        )

    html(
        '<div class="bm-sources" role="note" aria-label="Sources">'
        f'<div class="bm-sources-label">{icon("menu_book")}{escape(SOURCES_LABEL)}</div>'
        + "".join(rows)
        + "</div>"
    )


def thinking_indicator():
    return (
        '<div class="bm-thinking" role="status" aria-live="polite">'
        '<span class="bm-thinking-dots" aria-hidden="true">'
        "<span></span><span></span><span></span></span>"
        "Working on your request…"
        "</div>"
    )


# ---------- Data display ----------

def kpi_tiles(items):
    """
    items: dicts with label, value, optional note and optional swatch
    ("credit" | "debit" | "primary"). The swatch carries identity next
    to the label; values stay in text colour.
    """

    cells = []

    for item in items:
        swatch = item.get("swatch")
        dot = f'<span class="bm-swatch bm-swatch-{escape(swatch)}"></span>' if swatch else ""
        note = f'<div class="bm-kpi-note">{escape(item["note"])}</div>' if item.get("note") else ""

        cells.append(
            '<div class="bm-kpi">'
            f'<div class="bm-kpi-label">{dot}{escape(item["label"])}</div>'
            f'<div class="bm-kpi-value">{escape(str(item["value"]))}</div>'
            f"{note}</div>"
        )

    html(f'<div class="bm-kpis">{"".join(cells)}</div>')


def callout(text, glyph="info"):
    html(f'<div class="bm-callout">{icon(glyph)}<span>{escape(text)}</span></div>')


def empty_state(glyph, title, body):
    html(
        '<div class="bm-empty">'
        f'<span class="bm-icon-tile">{icon(glyph)}</span>'
        f'<div class="bm-empty-title">{escape(title)}</div>'
        f'<p class="bm-empty-body">{escape(body)}</p></div>'
    )


def panel_heading(title, subtitle=None):
    sub = f'<div class="bm-panel-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    html(f'<div class="bm-panel-title">{escape(title)}</div>{sub}')


def transactions_table(rows, customer_names=None, caption=None):
    """
    Accessible HTML table of transactions (newest first expected).
    customer_names: optional {customer_id: name} to add a Customer column.
    Amounts are signed: credits +, debits -. Type is shown as text with
    a direction icon, never colour alone.
    """

    # Imported here to keep components importable without app data.
    from ui.data import format_day, format_inr

    show_customer = customer_names is not None
    head = ["Date", "Transaction"]
    if show_customer:
        head.append("Customer")
    head += ["Merchant", "Type", "Amount"]

    body = []

    for row in rows:
        kind = (row.get("transaction_type") or "").strip().lower()
        is_credit = kind == "credit"
        if kind == "credit":
            amount = format_inr(row["amount"], signed=True)
        elif kind == "debit":
            amount = "−" + format_inr(row["amount"])
        else:
            amount = format_inr(row["amount"])
        badge = (
            f'<span class="bm-type bm-type-{"credit" if is_credit else "debit"}">'
            f'{icon("south_west" if is_credit else "north_east")}'
            f'{escape(row.get("transaction_type") or "")}</span>'
        )

        cells = [
            f'<td class="bm-nowrap">{escape(format_day(row["transaction_date"]))}</td>',
            f'<td class="bm-muted bm-num bm-col-secondary">{escape(row["transaction_id"])}</td>',
        ]
        if show_customer:
            name = customer_names.get(row["customer_id"], row["customer_id"])
            cells.append(
                f'<td>{escape(name)} <span class="bm-muted bm-col-secondary">{escape(row["customer_id"])}</span></td>'
            )
        cells += [
            f'<td>{escape(row.get("merchant") or "—")}</td>',
            f'<td class="bm-col-type">{badge}</td>',
            f'<td class="bm-num bm-right">{escape(amount)}</td>',
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")

    classes = {
        "Amount": ' class="bm-right"',
        "Transaction": ' class="bm-col-secondary"',
        "Type": ' class="bm-col-type"',
    }
    header = "".join(
        f'<th scope="col"{classes.get(label, "")}>{label}</th>'
        for label in head
    )
    caption_html = f'<caption>{escape(caption)}</caption>' if caption else ""

    html(
        '<div class="bm-table-wrap">'
        f'<table class="bm-table{" bm-table-wide" if show_customer else ""}">'
        f"{caption_html}<thead><tr>{header}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


# ---------- Stage UI: shared building blocks for the redesigned pages ----------

SCROLL_HINT = '<div class="bm-scroll-hint">Swipe the table sideways to see more columns.</div>'

STATUS_TONES = {"Open": "warning", "Under Investigation": "info", "Resolved": "success", "Rejected": "neutral",
                "Active": "success", "Blocked": "danger", "Expired": "neutral"}


def card_ref(last4, prefix=""):
    """'•••• 1234' (last four digits only) that never wraps between the dots and digits."""
    return f'<span class="bm-nowrap">{escape(prefix)}•••• {escape(last4)}</span>'


def badge(text, tone=None):
    tone = tone or STATUS_TONES.get(text, "neutral")
    return f'<span class="bm-badge bm-badge-{escape(tone)}">{escape(text)}</span>'


def section_header(title, subtitle=None):
    sub = f'<span class="bm-section-sub">{escape(subtitle)}</span>' if subtitle else ""
    html(f'<div class="bm-section-head"><span class="bm-section-title">{escape(title)}</span>{sub}</div>')


def ledger_table(rows, customer_names=None, card_last4=None, caption=None, compact=False):
    """
    Transactions in the full (v1) shape: date, customer (optional), account,
    merchant, category, channel and card (last four digits only), type and a
    signed amount. Direction is shown as text with an icon, never colour alone.
    """

    from ui.data import format_day, format_inr

    show_customer = customer_names is not None
    head = ["Date"] + (["Customer"] if show_customer else []) + ["Merchant", "Category"]
    if not compact:
        head += ["Account", "Channel", "Type"]
    head += ["Amount"]
    body = []
    for row in rows:
        is_credit = row["transaction_type"] == "Credit"
        amount = format_inr(row["amount"], signed=True) if is_credit else "−" + format_inr(row["amount"])
        badge_html = (f'<span class="bm-type bm-type-{"credit" if is_credit else "debit"}">'
                      f'{icon("south_west" if is_credit else "north_east")}{escape(row["transaction_type"])}</span>')
        last4 = card_last4(row.get("card_id")) if card_last4 else ""
        channel = escape(row.get("channel", "")) + (f' <span class="bm-muted">{card_ref(last4)}</span>' if last4 else "")
        refund = (f' <span class="bm-muted bm-col-secondary">refund of {escape(row["related_transaction_id"])}</span>'
                  if row.get("related_transaction_id") else "")
        cells = [f'<td class="bm-nowrap">{escape(format_day(row["transaction_date"]))}'
                 f'<div class="bm-muted bm-col-secondary bm-num">{escape(row["transaction_id"])}</div></td>']
        if show_customer:
            cells.append(f'<td>{escape(customer_names.get(row["customer_id"], row["customer_id"]))}'
                         f'<div class="bm-muted bm-col-secondary">{escape(row["customer_id"])}</div></td>')
        cells += [f'<td>{escape(row.get("merchant") or "—")}{refund}</td>',
                  f'<td>{escape(row.get("category", ""))}</td>']
        if not compact:
            cells += [f'<td class="bm-num bm-nowrap">{escape(row.get("account_id", ""))}</td>',
                      f'<td class="bm-nowrap">{channel}</td>',
                      f'<td class="bm-col-type">{badge_html}</td>']
        cells.append(f'<td class="bm-num bm-right bm-nowrap">{escape(amount)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    right = ' class="bm-right"'
    header = "".join(f'<th scope="col"{right if h == "Amount" else ""}>{h}</th>' for h in head)
    caption_html = f"<caption>{escape(caption)}</caption>" if caption else ""
    html('<div class="bm-table-wrap"><table class="bm-table bm-table-wide">'
         f'{caption_html}<thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
         f'{SCROLL_HINT}')


def pager(key, total, page_size, signature=None):
    """
    Page controls for a filtered list. Returns the offset to show. The page
    resets to the first one whenever `signature` (the filters) changes.
    """

    state_key, sig_key = f"bm-page-{key}", f"bm-page-sig-{key}"
    if st.session_state.get(sig_key) != signature:
        st.session_state[sig_key] = signature
        st.session_state[state_key] = 0
    pages = max(1, -(-total // page_size))
    page = min(st.session_state.get(state_key, 0), pages - 1)

    with st.container(horizontal=True, key=f"bm-pager-{key}", vertical_alignment="center"):
        first = page * page_size + 1 if total else 0
        last = min(total, (page + 1) * page_size)
        html(f'<span class="bm-pager-text">Showing <strong>{first:,}–{last:,}</strong> of '
             f'<strong>{total:,}</strong></span>')
        if st.button("Previous", key=f"{key}-prev", icon=":material/chevron_left:", disabled=page == 0,
                     type="tertiary"):
            st.session_state[state_key] = page - 1
            st.rerun()
        if st.button("Next", key=f"{key}-next", icon=":material/chevron_right:", disabled=page >= pages - 1,
                     type="tertiary"):
            st.session_state[state_key] = page + 1
            st.rerun()
    return page * page_size


def detail_list(items):
    """items: (label, value_html) pairs rendered as a compact definition list."""

    rows = "".join(f"<div><dt>{escape(label)}</dt><dd>{value}</dd></div>" for label, value in items)
    html(f'<dl class="bm-dl">{rows}</dl>')
