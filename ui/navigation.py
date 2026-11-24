"""
Single source of truth for BankMate pages.

Only pages that are implemented are registered here, so the sidebar,
the mobile navigation row and in-app links never point at unfinished
screens. Paths are relative to the entry script (streamlit_app.py).
"""

HOME = {
    "path": "ui/views/home.py",
    "title": "Overview",
    "short": "Overview",
    "icon": ":material/space_dashboard:",
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
    "section": "Overview",
    "default": False,
}

CUSTOMERS = {
    "path": "ui/views/customers.py",
    "title": "Customer 360",
    "short": "Customers",
    "icon": ":material/person_search:",
    "url_path": "customers",
    "section": "Intelligence",
    "default": False,
}

ANALYTICS = {
    "path": "ui/views/analytics.py",
    "title": "Analytics",
    "short": "Analytics",
    "icon": ":material/monitoring:",
    "url_path": "analytics",
    "section": "Intelligence",
    "default": False,
}

POLICIES = {
    "path": "ui/views/policies.py",
    "title": "Policy Intelligence",
    "short": "Policies",
    "icon": ":material/menu_book:",
    "url_path": "policies",
    "section": "Knowledge",
    "default": False,
}

TRANSACTIONS = {
    "path": "ui/views/transactions.py",
    "title": "Transactions",
    "short": "Txns",
    "icon": ":material/receipt_long:",
    "url_path": "transactions",
    "section": "Intelligence",
    "default": False,
}

DISPUTES = {
    "path": "ui/views/disputes.py",
    "title": "Disputes",
    "short": "Disputes",
    "icon": ":material/gavel:",
    "url_path": "disputes",
    "section": "Intelligence",
    "default": False,
}

PAGES = [HOME, ASSISTANT, CUSTOMERS, TRANSACTIONS, ANALYTICS, DISPUTES, POLICIES]


def sections():
    """Pages grouped by section, preserving registration order."""

    grouped = {}

    for page in PAGES:
        grouped.setdefault(page["section"], []).append(page)

    return grouped
