"""
Single source of truth for BankMate pages.

Only pages that are implemented are registered here, so the sidebar,
the mobile navigation row and in-app links never point at unfinished
screens. Paths are relative to the entry script (streamlit_app.py).
"""

HOME = {
    "path": "ui/views/home.py",
    "title": "Home",
    "short": "Home",
    "icon": ":material/home:",
    "url_path": "home",
    "section": "Overview",
    "default": True,
}

ASSISTANT = {
    "path": "ui/views/assistant.py",
    "title": "AI Assistant",
    "short": "Assistant",
    "icon": ":material/forum:",
    "url_path": "assistant",
    "section": "Workspace",
    "default": False,
}

CUSTOMERS = {
    "path": "ui/views/customers.py",
    "title": "Customer Explorer",
    "short": "Customers",
    "icon": ":material/group:",
    "url_path": "customers",
    "section": "Workspace",
    "default": False,
}

ANALYTICS = {
    "path": "ui/views/analytics.py",
    "title": "Transaction Analytics",
    "short": "Analytics",
    "icon": ":material/monitoring:",
    "url_path": "analytics",
    "section": "Workspace",
    "default": False,
}

POLICIES = {
    "path": "ui/views/policies.py",
    "title": "Policy Intelligence",
    "short": "Policies",
    "icon": ":material/menu_book:",
    "url_path": "policies",
    "section": "Workspace",
    "default": False,
}

PAGES = [HOME, ASSISTANT, CUSTOMERS, ANALYTICS, POLICIES]


def sections():
    """Pages grouped by section, preserving registration order."""

    grouped = {}

    for page in PAGES:
        grouped.setdefault(page["section"], []).append(page)

    return grouped
