"""
BankMate design system: tokens and a small, scoped CSS layer.

Most styling comes from .streamlit/config.toml. This CSS only adds what
the theme cannot express: card surfaces, chips, typographic helpers and
the mobile navigation row. Selectors target keyed containers
(`.st-key-<key>`) and stable `data-testid` hooks wherever possible.
"""

import streamlit as st


TOKENS = {
    # Surfaces
    "bg": "#F7F5F0",
    "surface": "#FFFFFF",
    "surface_subtle": "#F4F1EB",
    "border": "#E3DED4",
    "border_strong": "#D6D0C4",
    # Text
    "text": "#0B1B3A",
    "text_muted": "#5B6478",
    "text_faint": "#8A93A6",
    # Accents
    "primary": "#2E5BFF",
    "primary_hover": "#2149E0",
    "primary_soft": "#EAF0FF",
    "violet": "#7C5CFF",
    "violet_soft": "#F1EDFF",
    # Semantic (always paired with a text label)
    "credit": "#0F8A5F",
    "credit_soft": "#E7F5EF",
    "debit": "#C2410C",
    "debit_soft": "#FCEEE6",
    "danger": "#D92D20",
    # Shape & depth
    "radius_sm": "8px",
    "radius": "12px",
    "radius_lg": "16px",
    "shadow_sm": "0 1px 2px rgba(11, 27, 58, 0.04)",
    "shadow": "0 1px 2px rgba(11, 27, 58, 0.04), 0 6px 20px -8px rgba(11, 27, 58, 0.10)",
    "shadow_lg": "0 2px 4px rgba(11, 27, 58, 0.04), 0 16px 40px -16px rgba(11, 27, 58, 0.18)",
}

MOBILE_BREAKPOINT = "768px"


def _css_variables():
    return "\n".join(
        f"  --bm-{name.replace('_', '-')}: {value};"
        for name, value in TOKENS.items()
    )


_CSS = """
:root {
__VARIABLES__
}

/* ---------- App frame ---------- */

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 1120px;
  padding-top: 2.75rem;
  padding-bottom: 4rem;
}

[data-testid="stMain"] p,
[data-testid="stMain"] li {
  line-height: 1.6;
}

.bm-num {
  font-variant-numeric: tabular-nums;
}

/* ---------- Sidebar ---------- */

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
  border-radius: var(--bm-radius-sm);
  padding-top: 0.4rem;
  padding-bottom: 0.4rem;
  transition: background-color 150ms ease;
}

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {
  font-weight: 500;
  color: var(--bm-text-muted);
}

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
  background: var(--bm-primary-soft);
}

[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span {
  color: var(--bm-primary);
  font-weight: 600;
}

[data-testid="stNavSectionHeader"] {
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--bm-text-faint);
}

.bm-sidebar-note {
  margin-top: 0.5rem;
  padding: 0.75rem 0.875rem;
  border: 1px solid var(--bm-border);
  border-radius: var(--bm-radius);
  background: var(--bm-surface-subtle);
  font-size: 0.78rem;
  line-height: 1.5;
  color: var(--bm-text-muted);
}

.bm-sidebar-note strong {
  display: block;
  color: var(--bm-text);
  font-weight: 600;
  margin-bottom: 0.125rem;
}

/* ---------- Typography helpers ---------- */

.bm-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.3rem 0.7rem 0.3rem 0.55rem;
  border: 1px solid var(--bm-border);
  border-radius: 999px;
  background: var(--bm-surface);
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  color: var(--bm-text-muted);
}

.bm-eyebrow-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--bm-violet);
  box-shadow: 0 0 0 3px var(--bm-violet-soft);
}

.bm-display {
  margin: 1rem 0 0.75rem;
  font-family: "Inter Tight", Inter, sans-serif;
  font-size: clamp(1.875rem, 1.2rem + 2.2vw, 2.75rem);
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: -0.025em;
  color: var(--bm-text);
}

.bm-display em {
  font-style: normal;
  color: var(--bm-primary);
}

.bm-lede {
  max-width: 40rem;
  margin: 0;
  font-size: 1.0625rem;
  line-height: 1.6;
  color: var(--bm-text-muted);
}

.bm-section-label {
  margin: 2.75rem 0 0.875rem;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--bm-text-faint);
}

.bm-page-header {
  margin-bottom: 1.75rem;
}

[data-testid="stMarkdownContainer"] h1.bm-page-title,
.bm-page-title {
  padding: 0;
  margin: 0.25rem 0 0.375rem;
  font-family: "Inter Tight", Inter, sans-serif;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--bm-text);
}

.bm-page-subtitle {
  margin: 0;
  max-width: 44rem;
  color: var(--bm-text-muted);
}

.bm-footnote {
  margin-top: 3rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--bm-border);
  font-size: 0.78rem;
  color: var(--bm-text-faint);
}

/* ---------- Icons (Material Symbols, bundled with Streamlit) ---------- */

.bm-icon {
  font-family: "Material Symbols Rounded";
  font-weight: normal;
  font-style: normal;
  font-size: 20px;
  line-height: 1;
  letter-spacing: normal;
  text-transform: none;
  white-space: nowrap;
  direction: ltr;
  -webkit-font-smoothing: antialiased;
  font-feature-settings: "liga";
  user-select: none;
}

.bm-icon-tile {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 10px;
  background: var(--bm-primary-soft);
  color: var(--bm-primary);
}

.bm-icon-tile.violet {
  background: var(--bm-violet-soft);
  color: var(--bm-violet);
}

.bm-icon-tile.credit {
  background: var(--bm-credit-soft);
  color: var(--bm-credit);
}

/* ---------- Cards (keyed containers: st-key-bm-card-*) ---------- */

[data-testid="stColumn"]:has([class*="st-key-bm-card"]) > [data-testid="stVerticalBlock"],
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-bm-card"]) {
  height: 100%;
}

[class*="st-key-bm-card"] {
  height: 100%;
  background: var(--bm-surface);
  border: 1px solid var(--bm-border) !important;
  border-radius: var(--bm-radius-lg) !important;
  box-shadow: var(--bm-shadow-sm);
  padding: 1.25rem 1.25rem 1rem !important;
  transition: box-shadow 150ms ease, border-color 150ms ease;
}

[class*="st-key-bm-card"] > [data-testid="stLayoutWrapper"]:last-child {
  margin-top: auto;
}

[class*="st-key-bm-card"]:hover {
  border-color: var(--bm-border-strong) !important;
  box-shadow: var(--bm-shadow);
}

.bm-card-title {
  margin: 0.875rem 0 0.25rem;
  font-weight: 600;
  font-size: 1rem;
  color: var(--bm-text);
}

[data-testid="stMarkdownContainer"] p.bm-card-body,
.bm-card-body {
  margin: 0 0 0.5rem;
  font-size: 0.875rem;
  line-height: 1.55;
  color: var(--bm-text-muted);
}

/* ---------- Composer (home conversational entry) ---------- */

.st-key-bm-composer {
  background: var(--bm-surface);
  border: 1px solid var(--bm-border) !important;
  border-radius: 20px !important;
  box-shadow: var(--bm-shadow-lg);
  padding: 0.875rem 0.875rem 1rem !important;
  margin-top: 2rem;
}

.st-key-bm-composer [data-testid="stChatInput"] div {
  background: transparent;
  border-color: transparent;
  box-shadow: none;
}

.st-key-bm-composer [data-testid="stChatInputTextArea"] {
  font-size: 1rem;
  min-height: 3rem;
}

.st-key-bm-composer [data-testid="stChatInputSubmitButton"] {
  background: var(--bm-primary);
  color: #FFFFFF;
  border-radius: 10px;
}

.st-key-bm-composer [data-testid="stChatInputSubmitButton"]:disabled {
  background: var(--bm-surface-subtle);
  color: var(--bm-text-faint);
}

.bm-composer-hint {
  padding: 0.5rem 0.5rem 0.25rem;
  border-top: 1px solid var(--bm-border);
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--bm-text-faint);
}

/* ---------- Chips (keyed buttons: st-key-bm-chip-*) ---------- */

.st-key-bm-chiprow {
  flex-wrap: wrap !important;
  padding: 0 0.5rem;
}

.st-key-bm-chiprow > [data-testid="stLayoutWrapper"] {
  flex: 0 1 auto !important;
  width: auto !important;
  min-width: 0;
}

[class*="st-key-bm-chip-"] button {
  white-space: nowrap;
}

[class*="st-key-bm-chip"] button {
  min-height: 2rem;
  padding: 0.3rem 0.8rem;
  border: 1px solid var(--bm-border);
  border-radius: 999px;
  background: var(--bm-surface-subtle);
  color: var(--bm-text);
  font-size: 0.8125rem;
  font-weight: 500;
  transition: background-color 150ms ease, border-color 150ms ease, color 150ms ease;
}

[class*="st-key-bm-chip"] button:hover {
  border-color: var(--bm-primary);
  background: var(--bm-primary-soft);
  color: var(--bm-primary);
}

[class*="st-key-bm-chip"] button p {
  font-size: 0.8125rem;
}

/* ---------- Text links styled as actions (st-key-bm-link-*) ---------- */

[class*="st-key-bm-link"] button {
  padding: 0;
  min-height: 1.75rem;
  color: var(--bm-primary);
  font-weight: 600;
}

[class*="st-key-bm-link"] button p {
  font-size: 0.8125rem;
  font-weight: 600;
}

[class*="st-key-bm-link"] button:hover {
  color: var(--bm-primary-hover);
}

/* ---------- Stat strip ---------- */

/* 1px gaps over a border-coloured background draw the dividers, so the
   grid stays correct at any column count. */
.bm-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
  gap: 1px;
  border: 1px solid var(--bm-border);
  border-radius: var(--bm-radius-lg);
  background: var(--bm-border);
  overflow: hidden;
}

.bm-stat {
  padding: 1rem 1.25rem;
  background: var(--bm-surface);
}

.bm-stat-value {
  font-family: "Inter Tight", Inter, sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  color: var(--bm-text);
}

.bm-stat-label {
  margin-top: 0.125rem;
  font-size: 0.8125rem;
  color: var(--bm-text-muted);
}

/* ---------- Assistant: thread layout ---------- */

.st-key-bm-thread {
  max-width: 46rem;
  margin: 0 auto;
  gap: 0.25rem !important;
}

[data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
  max-width: 48rem;
  padding-bottom: 1.25rem;
}

[data-testid="stBottom"] [data-testid="stChatInput"] > div {
  background: var(--bm-surface);
  border: 1px solid var(--bm-border);
  border-radius: 16px;
  box-shadow: var(--bm-shadow);
}

[data-testid="stBottom"] [data-testid="stChatInput"] > div:focus-within {
  border-color: var(--bm-primary);
  box-shadow: 0 0 0 3px var(--bm-primary-soft), var(--bm-shadow);
}

[data-testid="stBottom"] [data-testid="stChatInput"] div {
  background: transparent;
}

[data-testid="stBottom"] [data-testid="stChatInputSubmitButton"] {
  background: var(--bm-primary);
  color: #FFFFFF;
  border-radius: 10px;
}

[data-testid="stBottom"] [data-testid="stChatInputSubmitButton"]:disabled {
  background: var(--bm-surface-subtle);
  color: var(--bm-text-faint);
}

.bm-thread-header {
  padding: 0 0 1rem;
  margin-bottom: 0.75rem;
  border-bottom: 1px solid var(--bm-border);
}

.bm-thread-title {
  font-family: "Inter Tight", Inter, sans-serif;
  font-size: clamp(1.0625rem, 0.95rem + 0.5vw, 1.25rem);
  font-weight: 700;
  letter-spacing: -0.015em;
  color: var(--bm-text);
}

.bm-thread-meta {
  margin-top: 0.125rem;
  font-size: 0.8125rem;
  color: var(--bm-text-faint);
}

/* ---------- Assistant: messages ---------- */

[data-testid="stChatMessage"] {
  padding: 0.5rem 0;
  gap: 0.875rem;
  background: transparent;
}

/* User: right-aligned bubble, no avatar */
[class*="st-key-bm-msg-user"] [data-testid="stChatMessage"] {
  flex-direction: row-reverse;
}

[class*="st-key-bm-msg-user"] [data-testid="stChatMessage"] > :first-child {
  display: none;
}

[class*="st-key-bm-msg-user"] [data-testid="stChatMessageContent"] {
  flex: 0 1 auto;
  max-width: min(85%, 34rem);
  margin-left: auto !important;
  margin-right: 0 !important;
  padding: 0.625rem 1rem;
  border-radius: 18px 18px 6px 18px;
  background: var(--bm-primary-soft);
  color: var(--bm-text);
}

[class*="st-key-bm-msg-user"] [data-testid="stChatMessageContent"] p {
  margin: 0;
}

/* Streamlit pulls markdown up by -1rem; undo it inside bubbles so the
   bubble wraps its text. */
[class*="st-key-bm-msg-user"] [data-testid="stMarkdownContainer"],
[class*="st-key-bm-msg-error"] [data-testid="stMarkdownContainer"] {
  margin-bottom: 0 !important;
}

/* Assistant: violet AI mark, open layout for readable answers */
[class*="st-key-bm-msg-assistant"] [data-testid="stChatMessage"] > :first-child,
[class*="st-key-bm-msg-error"] [data-testid="stChatMessage"] > :first-child {
  width: 2rem;
  height: 2rem;
  border-radius: 10px;
  background: var(--bm-violet-soft);
  color: var(--bm-violet);
}

[class*="st-key-bm-msg-error"] [data-testid="stChatMessage"] > :first-child {
  background: #FDECEA;
  color: var(--bm-danger);
}

[class*="st-key-bm-msg-error"] [data-testid="stChatMessageContent"] {
  padding: 0.625rem 0.875rem;
  border: 1px solid #F6CFCB;
  border-radius: var(--bm-radius);
  background: #FFF8F7;
}

[class*="st-key-bm-msg-error"] [data-testid="stChatMessageContent"] p {
  margin: 0;
  color: #8A1F17;
}

[class*="st-key-bm-msg-assistant"] [data-testid="stChatMessageContent"] {
  padding-top: 0.25rem;
}

[class*="st-key-bm-msg-assistant"] [data-testid="stChatMessageContent"] table {
  font-size: 0.875rem;
}

.st-key-bm-retry {
  padding-left: 2.875rem;
}

/* ---------- Assistant: sources ---------- */

.bm-sources {
  margin-top: 0.875rem;
  padding: 0.75rem;
  border: 1px solid var(--bm-border);
  border-radius: var(--bm-radius);
  background: var(--bm-surface);
}

.bm-sources-label {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  margin: 0 0 0.5rem 0.125rem;
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--bm-text-faint);
}

.bm-sources-label .bm-icon {
  font-size: 15px;
}

.bm-source {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.5rem 0.375rem;
  border-top: 1px solid var(--bm-border);
}

.bm-sources-label + .bm-source {
  border-top: none;
}

.bm-source-id {
  flex: none;
  padding: 0.125rem 0.5rem;
  border-radius: 6px;
  background: var(--bm-violet-soft);
  color: var(--bm-violet);
  font-size: 0.75rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.01em;
}

.bm-source-title {
  font-size: 0.875rem;
  font-weight: 600;
  line-height: 1.35;
  color: var(--bm-text);
}

.bm-source-meta {
  margin-top: 0.125rem;
  font-size: 0.78rem;
  color: var(--bm-text-faint);
}

/* ---------- Assistant: working indicator ---------- */

.bm-thinking {
  display: inline-flex;
  align-items: center;
  gap: 0.625rem;
  padding: 0.375rem 0;
  font-size: 0.875rem;
  color: var(--bm-text-muted);
}

.bm-thinking-dots {
  display: inline-flex;
  gap: 4px;
}

.bm-thinking-dots span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--bm-violet);
  opacity: 0.35;
  animation: bm-pulse 1.2s ease-in-out infinite;
}

.bm-thinking-dots span:nth-child(2) { animation-delay: 0.15s; }
.bm-thinking-dots span:nth-child(3) { animation-delay: 0.3s; }

@keyframes bm-pulse {
  0%, 80%, 100% { opacity: 0.35; transform: translateY(0); }
  40% { opacity: 1; transform: translateY(-2px); }
}

@media (prefers-reduced-motion: reduce) {
  .bm-thinking-dots span { animation: none; opacity: 0.7; }
}

/* ---------- Assistant: empty state ---------- */

.bm-chat-empty {
  padding: 3.5rem 0 1.25rem;
  text-align: center;
}

.bm-chat-empty-icon {
  width: 3rem;
  height: 3rem;
  border-radius: 14px;
}

.bm-icon-tile.bm-tile-danger {
  background: #FDECEA;
  color: var(--bm-danger);
}

.bm-chat-empty-icon .bm-icon {
  font-size: 24px;
}

[data-testid="stMarkdownContainer"] h2.bm-chat-empty-title,
.bm-chat-empty-title {
  margin: 1rem 0 0.375rem;
  padding: 0;
  font-family: "Inter Tight", Inter, sans-serif;
  font-size: 1.625rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--bm-text);
}

.bm-chat-empty-body {
  max-width: 32rem;
  margin: 0 auto 1.25rem !important;
  text-wrap: balance;
  color: var(--bm-text-muted);
}

.bm-chat-empty-detail {
  margin: 0 auto !important;
  font-size: 0.8125rem;
  color: var(--bm-text-faint);
}

.st-key-bm-chiprow-empty {
  flex-wrap: wrap !important;
  justify-content: center !important;
}

.st-key-bm-chiprow-empty > [data-testid="stLayoutWrapper"] {
  flex: 0 1 auto !important;
  width: auto !important;
  min-width: 0;
}

/* ---------- Assistant: sidebar conversation list ---------- */

.bm-side-label {
  margin: 1.25rem 0 0.375rem 0.5rem;
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--bm-text-faint);
}

[class*="st-key-bm-conv"] button {
  justify-content: flex-start;
  min-height: 2.25rem;
  padding: 0.375rem 0.625rem;
  border-radius: var(--bm-radius-sm);
  color: var(--bm-text-muted);
  transition: background-color 150ms ease;
}

[class*="st-key-bm-conv"] button:hover {
  background: var(--bm-surface-subtle);
  color: var(--bm-text);
}

[class*="st-key-bm-conv"] button p {
  overflow: hidden;
  font-size: 0.8125rem;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
}

[class*="st-key-bm-conv"] [data-testid="stElementContainer"] {
  margin-bottom: -0.5rem;
}

.st-key-bm-conv-active button,
.st-key-bm-conv-active button:disabled {
  background: var(--bm-surface-subtle);
  color: var(--bm-text) !important;
  opacity: 1;
  cursor: default;
}

.st-key-bm-conv-active button p {
  font-weight: 600;
}

/* ---------- Mobile navigation (hidden on desktop) ---------- */

.st-key-bm-mobile-nav {
  display: none !important;
}

@media (max-width: __BREAKPOINT__) {
  [data-testid="stMainBlockContainer"],
  .block-container {
    padding-top: 4.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
  }

  .st-key-bm-mobile-nav {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto;
    gap: 0.375rem !important;
    margin: 0 -1rem 1.5rem;
    padding: 4px 1rem 4px !important;
    scrollbar-width: none;
  }

  .st-key-bm-mobile-nav [data-testid="stElementContainer"],
  .st-key-bm-mobile-nav [data-testid="stPageLink"] {
    margin: 0 !important;
  }

  .st-key-bm-mobile-nav > [data-testid="stLayoutWrapper"] {
    flex: 0 0 auto !important;
    width: auto !important;
  }

  .st-key-bm-mobile-nav::-webkit-scrollbar {
    display: none;
  }

  .st-key-bm-mobile-nav [data-testid="stPageLink-NavLink"] {
    white-space: nowrap;
    padding: 0.3rem 0.8rem;
    border: 1px solid var(--bm-border);
    border-radius: 999px;
    background: var(--bm-surface);
  }

  .st-key-bm-mobile-nav [data-testid="stPageLink-NavLink"] p,
  .st-key-bm-mobile-nav [data-testid="stPageLink-NavLink"] span {
    font-size: 0.875rem;
  }

  .st-key-bm-navlink-active [data-testid="stPageLink-NavLink"] {
    border-color: var(--bm-primary);
    background: var(--bm-primary-soft);
    opacity: 1;
  }

  .st-key-bm-navlink-active [data-testid="stPageLink-NavLink"] p,
  .st-key-bm-navlink-active [data-testid="stPageLink-NavLink"] span {
    color: var(--bm-primary) !important;
    font-weight: 600;
  }

  .bm-stats {
    grid-template-columns: 1fr 1fr;
  }

  .st-key-bm-chiprow > [data-testid="stLayoutWrapper"] {
    max-width: 100%;
  }

  [class*="st-key-bm-chip-"] button {
    white-space: normal;
    text-align: left;
  }

  .st-key-bm-composer {
    margin-top: 1.5rem;
  }
}

/* ---------- Chrome cleanup ---------- */

#MainMenu,
footer {
  visibility: hidden;
}
"""


def apply_theme():
    """Inject the scoped CSS layer. Call once per run, before content."""

    css = (
        _CSS
        .replace("__VARIABLES__", _css_variables())
        .replace("__BREAKPOINT__", MOBILE_BREAKPOINT)
    )

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
