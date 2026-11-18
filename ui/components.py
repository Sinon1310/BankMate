"""
Reusable BankMate UI components.

Every page builds from these so spacing, typography and interaction
patterns stay consistent. Dynamic text is HTML-escaped before it is
placed inside custom markup.
"""

from html import escape

import streamlit as st

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
                    label=page["title"],
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


def _display_case(text):
    # Policy names are stored upper-case in the documents; mirror the
    # backend's display rule (title-case only all-caps text).
    return text.title() if text.isupper() else text


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
        name = escape(_display_case(citation.get("policy_name") or "")) or policy_id

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
