"""
BankMate web application entry point.

Run from the repository root:
    streamlit run streamlit_app.py

Sets up the page, theme and navigation, then runs the selected page from
ui/views/. Pages are registered in ui/navigation.py.
"""

import streamlit as st

from ui import components as ui
from ui import navigation, state
from ui.theme import apply_theme


st.set_page_config(
    page_title="BankMate",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="auto",
)

apply_theme()
state.init_state()

st.logo(
    "ui/assets/logo.svg",
    size="large",
    icon_image="ui/assets/logo_mark.svg",
)

pages = {
    section: [
        st.Page(
            page["path"],
            title=page["title"],
            icon=page["icon"],
            url_path=page["url_path"],
            default=page["default"],
        )
        for page in section_pages
    ]
    for section, section_pages in navigation.sections().items()
}

current_page = st.navigation(pages, position="sidebar")

ui.mobile_nav(current_page.title)

current_page.run()

# Rendered after the page so page-specific sidebar actions sit above it.
ui.sidebar_note(
    "Demo workspace",
    "Synthetic customer data and fictional training policies.",
)
