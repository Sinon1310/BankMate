"""
Single source of truth for BankMate pages.

Only pages that are implemented are registered here, so the sidebar,
the mobile navigation row and in-app links never point at unfinished
screens. Paths are relative to the entry script (streamlit_app.py).
"""

HOME = {
    "path": "ui/views/home.py",
    "title": "Home",
    "icon": ":material/home:",
    "url_path": "home",
    "section": "Overview",
    "default": True,
}

ASSISTANT = {
    "path": "ui/views/assistant.py",
    "title": "AI Assistant",
    "icon": ":material/forum:",
    "url_path": "assistant",
    "section": "Workspace",
    "default": False,
}

PAGES = [HOME, ASSISTANT]


def sections():
    """Pages grouped by section, preserving registration order."""

    grouped = {}

    for page in PAGES:
        grouped.setdefault(page["section"], []).append(page)

    return grouped
