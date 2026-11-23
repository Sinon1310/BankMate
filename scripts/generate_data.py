"""
Deterministic synthetic banking dataset for BankMate (Stage 1A).

    python -m scripts.generate_data --output-dir <folder> [--seed N]

Writes customers/accounts/cards/transactions/disputes CSVs and
dataset.json into <folder>, after validating everything with
app.data_validation. The repository's data/ folder is refused: the
application datasets are only replaced in a later, reviewed stage.

Determinism: one random.Random(seed) drives everything, and only its
random() method is used (through the helpers below). Python guarantees
random() sequences for a given integer seed across versions, unlike
randrange/choice/shuffle, so the output is byte-identical on Python 3.9
and 3.12. No set iteration or hash() influences ordering.

All people, merchants and employers are fictional. Customer and account
structure and every edge case are specified explicitly below; the
generator only randomises amounts, merchants and day-to-day activity.
"""

import argparse
import calendar
import csv
import hashlib
import json
import random
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import date, timedelta
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import ledger                                   # noqa: E402
from app.data_validation import TABLES, validate_dataset  # noqa: E402


GENERATOR_VERSION = "1.1.0"
DEFAULT_SEED = 20261121
WINDOW_START = date(2026, 5, 1)
AS_OF_DATE = date(2026, 11, 21)
TRANSACTION_RANGE = (1500, 2500)

CENT = Decimal("0.01")
ZERO = Decimal("0")


# ---------- Deterministic random helpers (random() only) ----------

class Rng:
    def __init__(self, seed):
        self._random = random.Random(seed)

    def random(self):
        return self._random.random()

    def between(self, low, high):
        """Integer in [low, high]."""
        return low + min(int(self.random() * (high - low + 1)), high - low)

    def pick(self, items):
        return items[min(int(self.random() * len(items)), len(items) - 1)]

    def weighted(self, pairs):
        total = sum(weight for _, weight in pairs)
        target = self.random() * total
        for item, weight in pairs:
            target -= weight
            if target < 0:
                return item
        return pairs[-1][0]

    def chance(self, probability):
        return self.random() < probability

    def poisson(self, lam):
        if lam <= 0:
            return 0
        threshold, count, product = pow(2.718281828459045, -lam), 0, 1.0
        while True:
            product *= self.random()
            if product <= threshold:
                return count
            count += 1

    def amount(self, low, high, paise_chance=0.35):
        """A rupee amount in [low, high], sometimes with paise."""
        rupees = self.between(int(low), int(high))
        paise = self.between(1, 99) if self.chance(paise_chance) else 0
        return (Decimal(rupees) + Decimal(paise) / 100).quantize(CENT)

    def round_amount(self, low, high, step):
        """A multiple of `step` in [low, high] (salaries, rent, ATM cash)."""
        return Decimal(self.between(int(low) // step, int(high) // step) * step).quantize(CENT)


def fmt(amount):
    return str(Decimal(amount).quantize(CENT, rounding=ROUND_HALF_UP))


def d(value):
    return date.fromisoformat(value)


# ---------- Explicit structure ----------

# id, name, city, segment, profile. C101-C103 keep their original names
# and cities. Profiles drive activity and are not stored.
CUSTOMERS = [
    ("C101", "Arjun Mehta", "Mumbai", "Retail", "salaried"),
    ("C102", "Priya Sharma", "Pune", "Business", "business"),
    ("C103", "Rahul Nair", "Bengaluru", "Retail", "salaried"),
    ("C104", "Ananya Iyer", "Chennai", "Premier", "salaried"),
    ("C105", "Vikrant Sethi", "Delhi", "Business", "business"),
    ("C106", "Neha Kulkarni", "Pune", "Retail", "salaried"),
    ("C107", "Rohan Das", "Kolkata", "Student", "student"),
    ("C108", "Kavya Reddy", "Hyderabad", "Retail", "salaried"),
    ("C109", "Aditya Joshi", "Ahmedabad", "Business", "business"),
    ("C110", "Meera Pillai", "Kochi", "Senior", "senior"),
    ("C111", "Siddharth Rao", "Bengaluru", "Retail", "salaried"),
    ("C112", "Ishita Banerjee", "Kolkata", "Retail", "low"),
    ("C113", "Kunal Bhatia", "Delhi", "Premier", "salaried"),
    ("C114", "Pooja Desai", "Surat", "Business", "business"),
    ("C115", "Nikhil Verma", "Lucknow", "Student", "student"),
    ("C116", "Sneha Patil", "Mumbai", "Retail", "salaried"),
    ("C117", "Arnav Gupta", "Jaipur", "Student", "student"),
    ("C118", "Divya Menon", "Kochi", "Retail", "salaried"),
    ("C119", "Harsh Agarwal", "Jaipur", "Business", "business"),
    ("C120", "Tanvi Shah", "Ahmedabad", "Retail", "salaried"),
    ("C121", "Ramesh Iyengar", "Chennai", "Senior", "senior"),
    ("C122", "Lakshmi Narayanan", "Chennai", "Senior", "senior"),
    ("C123", "Suresh Bhat", "Bengaluru", "Senior", "senior"),
    ("C124", "Gopal Krishnan", "Hyderabad", "Senior", "senior"),
    ("C125", "Zoya Qureshi", "Lucknow", "Student", "student"),
    ("C126", "Aarav Kapoor", "Delhi", "Premier", "salaried"),
    ("C127", "Riya Chatterjee", "Kolkata", "Retail", "low"),
    ("C128", "Farhan Siddiqui", "Mumbai", "Retail", "low"),
    ("C129", "Isha Trivedi", "Surat", "Student", "student"),
    ("C130", "Dev Malhotra", "Pune", "Retail", "low"),
]

# customer, account_type, opened_on, role, has_card. Roles: primary
# (main personal account), secondary (extra personal account), business
# (business current account), idle (rarely used savings).
ACCOUNTS = [
    ("C101", "Savings", "2019-04-12", "primary", True),
    ("C101", "Current", "2022-09-05", "secondary", True),
    ("C102", "Current", "2018-02-20", "business", True),
    ("C102", "Savings", "2018-02-20", "primary", True),
    ("C102", "Current", "2023-06-11", "business", False),
    ("C103", "Savings", "2020-11-03", "primary", True),
    ("C104", "Savings", "2017-08-21", "primary", True),
    ("C104", "Savings", "2021-01-15", "idle", True),
    ("C105", "Current", "2019-10-01", "business", True),
    ("C105", "Savings", "2026-07-15", "primary", True),      # opened inside the window
    ("C106", "Savings", "2021-05-09", "primary", True),
    ("C107", "Savings", "2024-07-22", "primary", True),
    ("C108", "Savings", "2020-03-14", "primary", True),
    ("C109", "Current", "2016-12-05", "business", True),
    ("C109", "Savings", "2017-01-10", "primary", False),
    ("C110", "Savings", "2012-06-18", "primary", True),
    ("C110", "Savings", "2015-03-02", "idle", True),
    ("C111", "Savings", "2022-02-28", "primary", True),
    ("C112", "Savings", "2023-09-30", "primary", True),
    ("C113", "Savings", "2016-04-11", "primary", True),
    ("C113", "Savings", "2019-12-01", "idle", True),
    ("C113", "Current", "2021-07-19", "secondary", False),
    ("C114", "Current", "2020-08-25", "business", True),
    ("C114", "Savings", "2020-08-25", "primary", True),
    ("C115", "Savings", "2025-01-08", "primary", True),
    ("C116", "Savings", "2019-07-30", "primary", True),
    ("C117", "Savings", "2024-06-03", "primary", True),
    ("C118", "Savings", "2021-10-17", "primary", True),
    ("C119", "Current", "2022-04-04", "business", True),
    ("C119", "Savings", "2022-04-04", "primary", False),
    ("C120", "Savings", "2020-01-27", "primary", True),
    ("C121", "Savings", "2010-09-09", "primary", True),
    ("C122", "Savings", "2011-02-14", "primary", True),
    ("C123", "Savings", "2014-05-23", "primary", True),
    ("C124", "Savings", "2013-11-11", "primary", True),
    ("C125", "Savings", "2025-07-01", "primary", True),
    ("C126", "Savings", "2018-06-06", "primary", True),
    ("C126", "Savings", "2026-09-01", "secondary", True),    # opened inside the window
    ("C127", "Savings", "2020-07-01", "primary", True),
    ("C128", "Savings", "2023-03-15", "primary", True),
    ("C129", "Savings", "2026-06-12", "primary", True),      # opened inside the window
    ("C130", "Savings", "2026-11-18", "primary", False),     # opened inside the window, no activity
]

# Initial deposits (Transfer In) for accounts opened inside the window.
# C130 deliberately has none: the zero-transaction customer.
INITIAL_DEPOSITS = {("C105", 1): "50000.00", ("C126", 1): "25000.00", ("C129", 0): "6000.00"}

# Primary cards with a fixed issue date (edge cases); others are derived.
FIXED_CARD_ISSUE = {
    ("C116", 0): ("2021-08-10", "2026-08"),   # expires in the window, renewed below
    ("C127", 0): ("2021-07-05", "2026-07"),   # expires in the window, not renewed
}
# Replacement cards: (customer, account index), issued_on, expires_on, reason.
REPLACEMENT_CARDS = [
    (("C111", 0), "2026-09-08", "2031-09", "block"),     # after D0003 (blocked 2026-09-04)
    (("C116", 0), "2026-08-12", "2031-08", "renewal"),
]
BLOCKED_CARDS = {("C111", 0): "2026-09-04"}

LOW_BALANCE_ACCOUNT = ("C117", 0)
LOW_BALANCE_TARGET = Decimal("1000.00")


# ---------- Fictional merchants ----------

MERCHANTS = {
    "Groceries": ["FreshCart", "DailyBasket", "GreenLeaf Mart", "Annapurna Stores"],
    "Dining": ["QuickBite", "Masala Junction", "Urban Tiffin", "Cafe Monsoon"],
    "Shopping": ["ShopKart", "StyleLane", "GadgetHub", "HomeNest"],
    "Travel": ["SkyRoute Air", "TrainEase", "StayInn Hotels", "TripWise"],
    "Transport": ["CityCab", "MetroLink Recharge", "FuelStop", "AutoGo"],
    "Utilities": ["UrbanPower Electricity", "AquaCity Water", "SwiftTel Mobile", "NetSpeed Broadband"],
    "Rent": ["HomeLease Rentals"],
    "Health": ["CareWell Pharmacy", "MediPlus Clinic", "LifeLine Diagnostics"],
    "Education": ["BrightPath Academy", "LearnHub Online", "Scholars Book Depot"],
    "Entertainment": ["StarScreen Cinemas", "StreamFlix", "PlayZone Arcade"],
    "Cash Withdrawal": ["ATM Withdrawal"],
    "Transfer Out": ["Family transfer", "Kiran Textiles", "Metro Supplies Co", "Lotus Packaging"],
}
EMPLOYERS = ["Nimbus Technologies", "Orion Finserv", "Kestrel Logistics", "Banyan Health",
             "Saffron Retail", "Tidewater Analytics", "Peacock Media", "Indus Engineering"]
CLIENTS = ["Client payment - Arcadia Exports", "Client payment - Bluebell Foods",
           "Client payment - Cedar Interiors", "Client payment - Delta Prints"]

AMOUNTS = {   # category -> (low, high) rupees for discretionary debits
    "Groceries": (250, 3200), "Dining": (180, 2600), "Shopping": (450, 8500),
    "Travel": (1200, 16000), "Transport": (60, 1100), "Utilities": (300, 2500),
    "Health": (150, 3800), "Education": (800, 12000), "Entertainment": (149, 1400),
    "Transfer Out": (1000, 15000),
}
# ATM withdrawals (low, high, step) by profile: multiples of the step.
CASH = {"student": (500, 3000, 500), "low": (500, 4000, 500), "senior": (1000, 10000, 500)}
DEFAULT_CASH = (500, 8000, 500)

# Budget-aware spending: one discretionary purchase uses at most this share
# of the balance left after the rest of the month's committed payments.
BUDGET_SHARE = Decimal("0.6")

CATEGORY_WEIGHTS = {
    "salaried": [("Groceries", 18), ("Dining", 16), ("Shopping", 12), ("Transport", 16),
                 ("Entertainment", 7), ("Health", 5), ("Travel", 3), ("Cash Withdrawal", 5),
                 ("Education", 1), ("Utilities", 2), ("Transfer Out", 3)],
    # Students' education costs are scheduled payments (see STUDENT_EDUCATION).
    "student": [("Dining", 25), ("Transport", 20), ("Groceries", 10), ("Shopping", 10),
                ("Entertainment", 15), ("Cash Withdrawal", 5)],
    "senior": [("Groceries", 25), ("Health", 25), ("Cash Withdrawal", 15), ("Dining", 5),
               ("Transport", 8), ("Shopping", 7), ("Utilities", 5), ("Travel", 2), ("Transfer Out", 8)],
    "business": [("Shopping", 20), ("Travel", 15), ("Transport", 15), ("Dining", 15), ("Utilities", 10)],
    "low": [("Groceries", 30), ("Transport", 20), ("Dining", 20), ("Shopping", 15), ("Cash Withdrawal", 15)],
}

# Discretionary transactions per month (per customer, before seasonality).
MONTHLY_RATE = {"salaried": (9, 17), "student": (5, 11), "senior": (2, 6), "low": (1, 3)}
BUSINESS_RATE = {"business": (2, 5), "personal": (2, 4)}
# Share of a customer's discretionary activity on each personal account role.
ROLE_SHARE = {"primary": 0.8, "secondary": 0.2, "idle": 0.04}
CARD_SHARE = {"salaried": 0.55, "student": 0.3, "senior": 0.4, "business": 0.5, "low": 0.35}
AMOUNT_SCALE = {"Premier": Decimal("1.6"), "Student": Decimal("0.6"), "Senior": Decimal("0.9")}
SEASON = {10: 1.15, 11: 1.20}          # festive season (Diwali 8 Nov 2026)
INTEREST_DATES = (date(2026, 6, 30), date(2026, 9, 30))
QUARTERLY_INTEREST = Decimal("0.0075")

# Students: two occasional education payments (books, course fees) on the
# 6th of two different months, sized to what the account can afford.
STUDENT_EDUCATION = {"payments": 2, "day": 6, "range": (1000, 4500), "minimum": Decimal("500.00")}

# Recurring commitments (Transfer Out unless stated) for the primary
# account. All counterparties are fictional.
SIP_FUND = "SIP - Northstar Mutual Fund"
HOME_LOAN = "Home loan EMI - Pinnacle Home Finance"
LIFE_INSURANCE = "Insurance premium - SecureLife Insurance"
HEALTH_INSURANCE = "Health insurance premium - SecureLife Insurance"
HOUSEHOLD_STAFF = "Household staff salary"
FAMILY_SUPPORT = "Family support sent"
PHARMACY = "CareWell Pharmacy"


# ---------- Scripted events (refunds, disputes, edge cases) ----------
# Each refund/dispute target is a scripted purchase, so the case is
# guaranteed rather than left to chance. Keys: (customer, account index).

SCRIPTED_PURCHASES = [
    # tag, account, date, category, merchant, amount, duplicate
    ("R1", ("C104", 0), "2026-05-18", "Shopping", "StyleLane", "3499.00", False),
    ("R2", ("C106", 0), "2026-06-27", "Shopping", "HomeNest", "5890.00", False),
    ("R3", ("C108", 0), "2026-07-20", "Travel", "SkyRoute Air", "8640.00", False),
    ("R4", ("C113", 0), "2026-08-09", "Dining", "Masala Junction", "1860.00", False),
    ("R5", ("C118", 0), "2026-09-12", "Entertainment", "StarScreen Cinemas", "1200.00", False),
    ("R6", ("C120", 0), "2026-10-05", "Shopping", "GadgetHub", "12999.00", False),
    ("R7", ("C110", 0), "2026-10-19", "Groceries", "DailyBasket", "2345.50", False),
    ("R8", ("C126", 0), "2026-10-28", "Shopping", "StyleLane", "7200.00", False),
    ("R9", ("C103", 0), "2026-06-08", "Travel", "StayInn Hotels", "9800.00", False),
    ("R10", ("C122", 0), "2026-07-07", "Health", "LifeLine Diagnostics", "4500.00", False),
    ("D1", ("C101", 0), "2026-06-14", "Shopping", "ShopKart", "2750.00", True),
    ("D2", ("C116", 0), "2026-08-28", "Dining", "Urban Tiffin", "3480.00", False),
    ("D3", ("C111", 0), "2026-09-03", "Shopping", "GadgetHub", "7999.00", False),
    ("D4", ("C124", 0), "2026-07-14", "Shopping", "HomeNest", "4250.00", False),
    ("D5", ("C105", 0), "2026-10-22", "Dining", "Urban Tiffin", "2180.00", False),
    ("D6", ("C118", 0), "2026-11-02", "Shopping", "StyleLane", "5499.00", False),
    ("D7", ("C106", 0), "2026-11-12", "Transport", "CityCab", "640.00", True),
    ("D8", ("C113", 0), "2026-11-01", "Shopping", "ShopKart", "3150.00", False),
    ("EDGE-LAST-DAY", ("C101", 0), "2026-11-21", "Groceries", "FreshCart", "1186.40", False),
]

# tag of purchase, refund date, amount ("full" = the whole amount).
SCRIPTED_REFUNDS = [
    ("R1", "2026-05-26", "full"), ("R2", "2026-07-06", "full"), ("R3", "2026-08-04", "full"),
    ("R4", "2026-08-11", "full"), ("R5", "2026-09-13", "full"), ("R6", "2026-10-16", "full"),
    ("R7", "2026-10-21", "full"), ("R8", "2026-11-06", "2400.00"), ("R9", "2026-06-19", "3200.00"),
    ("R10", "2026-07-15", "1500.00"),
    ("D1", "2026-06-24", "full"),        # D0001, refunds the duplicate charge
    ("D2", "2026-09-12", "1200.00"),     # D0002, partial, the month after the purchase
]

# dispute_id, purchase tag, raised, reason, status, resolved, outcome.
# The disputed transaction of a duplicate pair is the second charge.
SCRIPTED_DISPUTES = [
    ("D0001", "D1", "2026-06-16", "Duplicate charge", "Resolved", "2026-06-24", "Full refund"),
    ("D0002", "D2", "2026-09-02", "Incorrect amount", "Resolved", "2026-09-12", "Partial refund"),
    ("D0003", "D3", "2026-09-04", "Unauthorized", "Resolved", "2026-09-25", "No financial adjustment"),
    ("D0004", "D4", "2026-07-25", "Goods not received", "Rejected", "2026-08-10", ""),
    ("D0005", "D5", "2026-10-28", "Incorrect amount", "Under Investigation", "", ""),
    ("D0006", "D6", "2026-11-04", "Unauthorized", "Under Investigation", "", ""),
    ("D0007", "D7", "2026-11-15", "Duplicate charge", "Open", "", ""),
    ("D0008", "D8", "2026-11-17", "Goods not received", "Open", "", ""),
]


# ---------- Generation ----------

class GenerationError(RuntimeError):
    pass


def build_customers_accounts_cards(rng):
    customers, accounts, cards = [], [], []
    account_ids = {}
    profiles = {}
    first_opened = {}

    per_customer = defaultdict(int)
    for number, (cid, acc_type, opened, role, has_card) in enumerate(ACCOUNTS):
        position = per_customer[cid]
        per_customer[cid] += 1
        account_ids[(cid, position)] = f"ACC{1001 + number}"
        first_opened[cid] = min(first_opened.get(cid, opened), opened)

    for cid, name, city, segment, profile in CUSTOMERS:
        profiles[cid] = (profile, segment)
        customers.append({"customer_id": cid, "customer_name": name, "city": city,
                          "segment": segment, "customer_since": first_opened[cid]})

    per_customer = defaultdict(int)
    meta = {}
    for cid, acc_type, opened, role, has_card in ACCOUNTS:
        position = per_customer[cid]
        per_customer[cid] += 1
        aid = account_ids[(cid, position)]
        profile, segment = profiles[cid]
        opened_date = d(opened)
        inside = opened_date >= WINDOW_START

        if inside:
            opening = ZERO
        elif role == "business":
            opening = rng.amount(90000, 420000, 0.5)
        elif role == "idle":
            opening = rng.amount(60000, 380000, 0.5)
        elif profile == "student":
            opening = rng.amount(2500, 9000, 0.5)
        elif profile == "senior":
            opening = rng.amount(80000, 420000, 0.5)
        elif profile == "low":
            opening = rng.amount(4000, 30000, 0.5)
        elif segment == "Premier":
            opening = rng.amount(180000, 600000, 0.5)
        else:
            opening = rng.amount(25000, 140000, 0.5)

        accounts.append({
            "account_id": aid, "customer_id": cid, "account_type": acc_type, "opened_on": opened,
            "opening_balance_date": max(WINDOW_START, opened_date).isoformat(),
            "opening_balance": fmt(opening),
        })
        meta[aid] = {"customer_id": cid, "position": position, "role": role,
                     "profile": profile, "segment": segment, "opened": opened_date}

    used_last4 = set()

    def new_last4():
        while True:
            value = f"{rng.between(0, 9999):04d}"
            if value not in used_last4:
                used_last4.add(value)
                return value

    card_number = 5001
    card_of = {}
    for (cid, acc_type, opened, role, has_card), account in zip(ACCOUNTS, accounts):
        if not has_card:
            continue
        aid = account["account_id"]
        key = (cid, meta[aid]["position"])
        opened_date = d(opened)

        if key in FIXED_CARD_ISSUE:
            issued, expires = FIXED_CARD_ISSUE[key]
        elif opened_date >= WINDOW_START:
            issued_date = opened_date + timedelta(days=rng.between(2, 5))
            issued, expires = issued_date.isoformat(), f"{issued_date.year + 5}-{issued_date.month:02d}"
        else:
            # Issued 2022-01..2026-03 (or after opening): expires 2027+.
            low = max(opened_date, date(2022, 1, 1))
            span = (date(2026, 3, 31) - low).days
            issued_date = low + timedelta(days=rng.between(0, max(span, 0)))
            issued, expires = issued_date.isoformat(), f"{issued_date.year + 5}-{issued_date.month:02d}"

        card = {"card_id": f"CRD{card_number}", "account_id": aid,
                "network": rng.pick(["RuPay", "Visa", "Mastercard"]), "card_last4": new_last4(),
                "issued_on": issued, "expires_on": expires,
                "blocked_on": BLOCKED_CARDS.get(key, ""), "replaces_card_id": "", "status": ""}
        cards.append(card)
        card_of[key] = card
        card_number += 1

    for key, issued, expires, _reason in REPLACEMENT_CARDS:
        old = card_of[key]
        cards.append({"card_id": f"CRD{card_number}", "account_id": old["account_id"],
                      "network": old["network"], "card_last4": new_last4(),
                      "issued_on": issued, "expires_on": expires, "blocked_on": "",
                      "replaces_card_id": old["card_id"], "status": ""})
        card_number += 1

    for card in cards:
        card["status"] = ledger.card_status_on(card, AS_OF_DATE)

    return customers, accounts, cards, account_ids, meta


def recurring_commitments(account, m, p):
    """
    Scheduled debits for one account: bills, rent/EMI, SIPs, premiums,
    household staff and medicines. Each item: day of month, months (None =
    every month), category, merchant, channel, a fixed amount or a draw
    function, and an upper bound ("max") used to reserve funds.
    """

    profile, role = m["profile"], m["role"]
    rng = p["rng"]
    items = []

    def add(day, category, merchant, channel="Bank Transfer", fixed=None, draw=None, maximum=None, months=None):
        items.append({"day": day, "category": category, "merchant": merchant, "channel": channel,
                      "fixed": fixed, "draw": draw, "max": fixed if fixed is not None else maximum,
                      "months": months})

    scale = p["scale"]
    if role == "primary" and profile == "salaried":
        if p["pays_rent"]:
            add(5, "Rent", "HomeLease Rentals", fixed=p["rent"])
        if "emi" in p:
            add(5, "Transfer Out", HOME_LOAN, fixed=p["emi"])
            add(7, "Transfer Out", SIP_FUND, fixed=p["sip"])
            add(20, "Transfer Out", LIFE_INSURANCE, fixed=p["insurance"], months=(p["insurance_month"],))
        add(10, "Utilities", "UrbanPower Electricity", "UPI",
            draw=lambda: (rng.amount(700, 3200) * scale).quantize(CENT), maximum=(Decimal(3200) * scale).quantize(CENT))
        add(15, "Utilities", "SwiftTel Mobile", "UPI",
            draw=lambda: rng.pick([Decimal(v) for v in ("299", "399", "499", "699", "999")]), maximum=Decimal("999"))
        if p["broadband"]:
            add(18, "Utilities", "NetSpeed Broadband", "UPI",
                draw=lambda: rng.pick([Decimal(v) for v in ("599", "799", "999", "1499")]), maximum=Decimal("1499"))
    if role == "primary" and profile == "student":
        add(12, "Utilities", "SwiftTel Mobile", "UPI",
            draw=lambda: (rng.amount(250, 2200) * scale).quantize(CENT), maximum=(Decimal(2200) * scale).quantize(CENT))
    if role == "primary" and profile == "senior":
        add(3, "Transfer Out", HOUSEHOLD_STAFF, fixed=p["household_staff"])
        add(8, "Health", PHARMACY, "UPI",
            draw=lambda: (rng.amount(*p["medicines"]) * scale).quantize(CENT),
            maximum=(Decimal(p["medicines"][1]) * scale).quantize(CENT))
        add(12, "Utilities", "UrbanPower Electricity", "UPI",
            draw=lambda: (rng.amount(250, 2200) * scale).quantize(CENT), maximum=(Decimal(2200) * scale).quantize(CENT))
        add(22, "Transfer Out", HEALTH_INSURANCE, fixed=p["health_insurance"], months=(p["insurance_month"],))
        if p["sip"]:
            add(7, "Transfer Out", SIP_FUND, fixed=p["sip"])
        if p["family_support"]:
            add(20, "Transfer Out", FAMILY_SUPPORT, fixed=p["family_support"])
    if role == "business":
        add(10, "Utilities", "UrbanPower Electricity",
            draw=lambda: rng.amount(1500, 9000), maximum=Decimal("9000"))

    # The account's day loop starts at its opening date, so nothing is due earlier.
    return items


class AccountLedger:
    """The generator's own running balance (independent of app.ledger)."""

    def __init__(self, account):
        self.account_id = account["account_id"]
        self.customer_id = account["customer_id"]
        self.balance = Decimal(account["opening_balance"])
        self.rows = []
        self.skipped = []          # guard refusals (details); expected to stay empty
        self.month_close = OrderedDict()

    def close_day(self, day):
        """Record the end-of-day balance; the last one in a month is its closing."""
        self.month_close[day.isoformat()[:7]] = self.balance

    def credit(self, day, amount, category, merchant, channel, card_id="", related_tag="", tag=""):
        amount = Decimal(amount).quantize(CENT, rounding=ROUND_HALF_UP)
        self.balance += amount
        self._add(day, "Credit", amount, category, merchant, channel, card_id, related_tag, tag)

    def debit(self, day, amount, category, merchant, channel, card_id="", tag="", required=False):
        amount = Decimal(amount).quantize(CENT, rounding=ROUND_HALF_UP)
        if amount > self.balance:
            if required:
                raise GenerationError(f"scripted debit {tag} of {amount} on {day} exceeds balance {self.balance}")
            self.skipped.append({"date": day.isoformat(), "customer_id": self.customer_id,
                                 "account_id": self.account_id, "category": category, "merchant": merchant,
                                 "channel": channel, "amount": fmt(amount), "balance": fmt(self.balance)})
            return False
        self.balance -= amount
        self._add(day, "Debit", amount, category, merchant, channel, card_id, "", tag)
        return True

    def _add(self, day, txn_type, amount, category, merchant, channel, card_id, related_tag, tag):
        self.rows.append({"date": day, "seq": len(self.rows), "transaction_type": txn_type,
                          "amount": amount, "category": category, "merchant": merchant,
                          "channel": channel, "card_id": card_id, "related_tag": related_tag, "tag": tag})


def generate(seed=DEFAULT_SEED):
    rng = Rng(seed)
    customers, accounts, cards, account_ids, meta = build_customers_accounts_cards(rng)

    cards_by_account = defaultdict(list)
    for card in cards:
        cards_by_account[card["account_id"]].append(card)

    def card_for(aid, day):
        """The newest card in service on `day`, or None."""
        valid = [c for c in cards_by_account[aid] if ledger.card_valid_on(c, day)]
        return max(valid, key=lambda c: c["issued_on"]) if valid else None

    # Per-customer parameters, drawn in a fixed order.
    params = {}
    for cid, _name, _city, segment, profile in CUSTOMERS:
        p = {"scale": AMOUNT_SCALE.get(segment, Decimal("1")), "rng": rng}
        if profile == "salaried":
            premier = segment == "Premier"
            p["salary"] = rng.round_amount(150000, 260000, 500) if premier else rng.round_amount(38000, 95000, 500)
            p["employer"] = rng.pick(EMPLOYERS)
            p["rent"] = rng.round_amount(18000, 48000, 500) if premier else rng.round_amount(11000, 32000, 500)
            p["pays_rent"] = rng.chance(0.75)
            p["broadband"] = rng.chance(0.6)
            p["rate"] = rng.between(*MONTHLY_RATE["salaried"])
            if premier:
                # Premier customers own their homes: a home-loan EMI replaces rent.
                p["pays_rent"] = False
                p["emi"] = rng.round_amount(35000, 70000, 500)
                p["sip"] = rng.round_amount(15000, 40000, 500)
                p["insurance"] = rng.round_amount(12000, 30000, 500)
                p["insurance_month"] = rng.between(6, 10)
        elif profile == "student":
            p["allowance"] = rng.round_amount(5000, 12000, 500)
            p["rate"] = rng.between(*MONTHLY_RATE["student"])
            first = rng.between(6, 8)
            p["education_months"] = (first, first + rng.between(2, 3))
        elif profile == "senior":
            p["pension"] = rng.round_amount(28000, 65000, 500)
            p["rate"] = rng.between(*MONTHLY_RATE["senior"])
            p["household_staff"] = rng.round_amount(4000, 9000, 500)
            p["medicines"] = (1500, 5000)
            p["health_insurance"] = rng.round_amount(18000, 40000, 500)
            p["insurance_month"] = rng.between(6, 10)
            p["sip"] = rng.round_amount(5000, 15000, 500) if rng.chance(0.6) else None
            p["family_support"] = rng.round_amount(5000, 15000, 500) if rng.chance(0.4) else None
        elif profile == "business":
            p["rate_business"] = rng.between(*BUSINESS_RATE["business"])
            p["drawings"] = rng.round_amount(30000, 90000, 500)
            p["rate"] = rng.between(*BUSINESS_RATE["personal"])
        else:
            p["rate"] = rng.between(*MONTHLY_RATE["low"])
        params[cid] = p

    scripted_by_account = defaultdict(lambda: defaultdict(list))
    for tag, key, day, category, merchant, amount, duplicate in SCRIPTED_PURCHASES:
        scripted_by_account[account_ids[key]][d(day)].append(
            ("purchase", tag, category, merchant, Decimal(amount), duplicate))
    purchase_account = {tag: account_ids[key] for tag, key, *_ in SCRIPTED_PURCHASES}
    purchase_amount = {tag: Decimal(amount) for tag, _k, _d, _c, _m, amount, _dup in SCRIPTED_PURCHASES}
    for tag, day, amount in SCRIPTED_REFUNDS:
        value = purchase_amount[tag] if amount == "full" else Decimal(amount)
        scripted_by_account[purchase_account[tag]][d(day)].append(("refund", tag, value))
    initial = {account_ids[key]: Decimal(v) for key, v in INITIAL_DEPOSITS.items()}
    low_balance_account = account_ids[LOW_BALANCE_ACCOUNT]

    ledgers = {}
    budget_limited = Counter()
    for account in accounts:
        aid = account["account_id"]
        m = meta[aid]
        cid, profile, role = m["customer_id"], m["profile"], m["role"]
        p = params[cid]
        book = AccountLedger(account)
        ledgers[aid] = book
        if cid == "C130":
            # Zero-transaction customer: the balance stays at its opening value.
            for month in ledger.months_between(account["opening_balance_date"], AS_OF_DATE):
                book.month_close[month] = book.balance
            continue

        purchase_cards = {}
        commitments = recurring_commitments(account, m, p)

        def reserve_after(day):
            """Committed payments still due later this month (upper bounds)."""
            return sum((c["max"] for c in commitments
                        if c["day"] > day.day and (c["months"] is None or day.month in c["months"])), ZERO)

        def spend(day, amount, category, merchant, channel, card_id="", minimum=Decimal("1"), step=None):
            """A discretionary debit sized to what the account can afford."""
            free = book.balance - reserve_after(day)
            budget = (free * BUDGET_SHARE).quantize(Decimal("1"), rounding=ROUND_DOWN) if free > 0 else ZERO
            if amount > budget:
                amount = (budget // step) * step if step else budget
            if amount < minimum:
                budget_limited[(profile, category)] += 1
                return
            book.debit(day, amount, category, merchant, channel, card_id)

        day = d(account["opening_balance_date"])
        while day <= AS_OF_DATE:
            month_factor = SEASON.get(day.month, 1.0)
            weekend = 1.25 if day.weekday() >= 5 else 0.9
            card = card_for(aid, day)

            # Credits first.
            if day == m["opened"] and aid in initial:
                book.credit(day, initial[aid], "Transfer In", "Initial deposit", "Bank Transfer")
            if role == "primary" and day.day == 1:
                if profile == "salaried":
                    book.credit(day, p["salary"], "Salary", f"Salary - {p['employer']}", "Bank Transfer")
                elif profile == "senior":
                    book.credit(day, p["pension"], "Pension", "Pension - Retirement Fund", "Bank Transfer")
            if role == "primary" and profile == "business" and day.day == 3:
                book.credit(day, p["drawings"], "Business Income", "Partner profit share", "Bank Transfer")
            if role == "primary" and profile == "student" and day.day == 2:
                book.credit(day, p["allowance"], "Transfer In", "Family transfer", "UPI")
            if profile == "low" and role == "primary" and rng.chance(0.6 / 30):
                book.credit(day, rng.round_amount(2000, 15000, 100), "Transfer In", "UPI transfer received", "UPI")
            if role == "business":
                for _ in range(rng.poisson(2.2 / 30)):
                    book.credit(day, rng.round_amount(12000, 85000, 100), "Business Income",
                                rng.pick(CLIENTS), "Bank Transfer")
            if day in INTEREST_DATES and account["account_type"] == "Savings" and book.balance > 0:
                interest = (book.balance * QUARTERLY_INTEREST).quantize(CENT, rounding=ROUND_HALF_UP)
                if interest >= Decimal("1.00"):
                    book.credit(day, interest, "Interest", "Savings interest", "Bank Transfer")

            # Scripted purchases and refunds.
            for event in scripted_by_account[aid].get(day, []):
                if event[0] == "purchase":
                    _kind, tag, category, merchant, amount, duplicate = event
                    if card is None:
                        raise GenerationError(f"scripted purchase {tag} on {day}: no card in service")
                    purchase_cards[tag] = card["card_id"]
                    book.debit(day, amount, category, merchant, "Card", card["card_id"], tag=tag, required=True)
                    if duplicate:
                        book.debit(day, amount, category, merchant, "Card", card["card_id"],
                                   tag=tag + "-DUP", required=True)
                else:
                    _kind, tag, amount = event
                    refund_card = purchase_cards.get(tag, "")
                    original_card = next(c for c in cards if c["card_id"] == refund_card) if refund_card else None
                    if original_card and ledger.card_valid_on(original_card, day):
                        channel, card_id = "Card", refund_card
                    else:
                        channel, card_id = "Bank Transfer", ""
                    # A duplicate's refund reverses the disputed (second) charge.
                    target = tag + "-DUP" if any(r["tag"] == tag + "-DUP" for r in book.rows) else tag
                    book.credit(day, amount, "Refund", "", channel, card_id, related_tag=target)

            # Recurring commitments (bills, EMIs, SIPs, premiums): never budget-capped;
            # the balance guard records any that cannot be paid.
            for c in commitments:
                if c["day"] == day.day and (c["months"] is None or day.month in c["months"]):
                    amount = c["fixed"] if c["fixed"] is not None else c["draw"]()
                    book.debit(day, amount, c["category"], c["merchant"], c["channel"])

            # Students: occasional, affordable education payments.
            if role == "primary" and profile == "student" and day.day == STUDENT_EDUCATION["day"] \
                    and day.month in p["education_months"]:
                spend(day, rng.amount(*STUDENT_EDUCATION["range"]), "Education",
                      rng.pick(MERCHANTS["Education"]), "UPI", minimum=STUDENT_EDUCATION["minimum"])

            if role == "business":
                for _ in range(rng.poisson(1.8 / 30)):
                    spend(day, rng.round_amount(8000, 70000, 100), "Transfer Out",
                          rng.pick(MERCHANTS["Transfer Out"][1:]), "Bank Transfer", minimum=Decimal("2000"))

            # Discretionary debits.
            if role == "business":
                lam = p["rate_business"] / 30.4 * month_factor * weekend
                weights = CATEGORY_WEIGHTS["business"]
            elif profile == "business":
                lam = p["rate"] / 30.4 * month_factor * weekend
                weights = CATEGORY_WEIGHTS["salaried"]
            else:
                lam = p["rate"] * ROLE_SHARE.get(role, 1.0) / 30.4 * month_factor * weekend
                weights = CATEGORY_WEIGHTS[profile]

            for _ in range(rng.poisson(lam)):
                category = rng.weighted(weights)
                if category == "Cash Withdrawal":
                    if card is None:
                        continue
                    low, high, step = CASH.get(profile, DEFAULT_CASH)
                    spend(day, rng.round_amount(low, high, step), category, "ATM Withdrawal", "ATM",
                          card["card_id"], minimum=Decimal(step), step=step)
                    continue
                low, high = AMOUNTS[category]
                amount = (rng.amount(low, high) * p["scale"]).quantize(CENT)
                merchant = rng.pick(MERCHANTS[category])
                share = CARD_SHARE["business" if role == "business" else profile]
                minimum = Decimal(low) * p["scale"]
                if card is not None and category != "Transfer Out" and rng.chance(share):
                    spend(day, amount, category, merchant, "Card", card["card_id"], minimum=minimum)
                else:
                    spend(day, amount, category, merchant,
                          "Bank Transfer" if category == "Transfer Out" else "UPI", minimum=minimum)

            # Low-balance edge case: a fee payment that leaves < 1,000.
            if aid == low_balance_account and day == AS_OF_DATE - timedelta(days=1):
                keep = Decimal(rng.between(200, 900))
                if book.balance > keep:
                    book.debit(day, book.balance - keep, "Education", "BrightPath Academy", "UPI",
                               tag="EDGE-LOW-BALANCE", required=True)

            book.close_day(day)
            day += timedelta(days=1)

    # Assign IDs in (date, account, generation order) and resolve links.
    rows = []
    for aid, book in ledgers.items():
        for row in book.rows:
            rows.append((row["date"], aid, row["seq"], row))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    transactions, id_of_tag = [], {}
    for number, (day, aid, _seq, row) in enumerate(rows, start=1):
        tid = f"T{number:05d}"
        if row["tag"]:
            id_of_tag[row["tag"]] = tid
        transactions.append({
            "transaction_id": tid, "account_id": aid, "customer_id": meta[aid]["customer_id"],
            "card_id": row["card_id"], "transaction_date": day.isoformat(),
            "transaction_type": row["transaction_type"], "amount": fmt(row["amount"]),
            "merchant": row["merchant"], "category": row["category"], "channel": row["channel"],
            "related_transaction_id": row["related_tag"],
        })

    by_id = {t["transaction_id"]: t for t in transactions}
    refund_of_original = {}
    for txn in transactions:
        if txn["related_transaction_id"]:
            original_id = id_of_tag[txn["related_transaction_id"]]
            txn["related_transaction_id"] = original_id
            txn["merchant"] = f"Refund - {by_id[original_id]['merchant']}"
            refund_of_original[original_id] = txn["transaction_id"]

    disputes = []
    for did, tag, raised, reason, status, resolved, outcome in SCRIPTED_DISPUTES:
        disputed = id_of_tag.get(tag + "-DUP", id_of_tag[tag])
        refund = refund_of_original.get(disputed, "") if outcome in ("Full refund", "Partial refund") else ""
        disputes.append({"dispute_id": did, "transaction_id": disputed, "raised_on": raised, "reason": reason,
                         "status": status, "resolved_on": resolved, "resolution_outcome": outcome,
                         "refund_transaction_id": refund})

    closing = {aid: fmt(book.balance) for aid, book in ledgers.items()}
    monthly = {aid: OrderedDict((month, fmt(value)) for month, value in book.month_close.items())
               for aid, book in ledgers.items()}
    diagnostics = {
        "guard_skips": [skip for book in ledgers.values() for skip in book.skipped],
        "budget_limited": dict(sorted((f"{profile}/{category}", count)
                                      for (profile, category), count in budget_limited.items())),
    }

    dataset = {"customers": customers, "accounts": accounts, "cards": cards,
               "transactions": transactions, "disputes": disputes}

    if not TRANSACTION_RANGE[0] <= len(transactions) <= TRANSACTION_RANGE[1]:
        raise GenerationError(f"{len(transactions)} transactions, outside the allowed range {TRANSACTION_RANGE}; "
                              "adjust the activity profiles (never pad).")

    meta_json = OrderedDict([
        ("dataset", "bankmate-synthetic"),
        ("description", "Fictional banking data for the BankMate portfolio project. Not real customers."),
        ("generator", "scripts/generate_data.py"),
        ("generator_version", GENERATOR_VERSION),
        ("seed", seed),
        ("window_start", WINDOW_START.isoformat()),
        ("as_of_date", AS_OF_DATE.isoformat()),
        ("currency", "INR"),
        ("row_counts", OrderedDict((t, len(dataset[t])) for t in TABLES)),
        ("closing_balances", OrderedDict(sorted(closing.items()))),
        ("monthly_closing_balances", OrderedDict(sorted(monthly.items()))),
    ])
    return dataset, meta_json, diagnostics


# ---------- Output ----------

def write_dataset(dataset, meta_json, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for table, columns in TABLES.items():
        with open(folder / f"{table}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(dataset[table])

    meta_json = OrderedDict(meta_json)
    meta_json["sha256"] = OrderedDict(
        (f"{t}.csv", hashlib.sha256((folder / f"{t}.csv").read_bytes()).hexdigest()) for t in TABLES
    )
    (folder / "dataset.json").write_text(json.dumps(meta_json, indent=2) + "\n", encoding="utf-8")


def as_validation_input(dataset, meta_json):
    data = dict(dataset)
    data["meta"] = dict(meta_json)
    data["headers"] = {t: list(cols) for t, cols in TABLES.items()}
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True, help="Folder to write the dataset into")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    output = Path(args.output_dir).resolve()
    if output == (PROJECT_ROOT / "data").resolve():
        parser.error("refusing to write into the application's data/ folder (replaced only in a reviewed stage)")

    dataset, meta_json, diagnostics = generate(args.seed)
    errors = validate_dataset(as_validation_input(dataset, meta_json))
    if errors:
        print(f"Validation failed with {len(errors)} error(s); nothing written:", file=sys.stderr)
        for error in errors[:50]:
            print("  " + error, file=sys.stderr)
        return 1

    write_dataset(dataset, meta_json, output)
    print(f"Wrote {sum(len(dataset[t]) for t in TABLES)} rows to {output} (seed {args.seed}).")
    skips = diagnostics["guard_skips"]
    print(f"Debits refused by the insufficient-balance guard: {len(skips)}")
    for skip in skips:
        print("  " + ", ".join(f"{k}={v}" for k, v in skip.items()))
    print(f"Discretionary debits not drawn because the budget did not allow them: "
          f"{sum(diagnostics['budget_limited'].values())} {diagnostics['budget_limited']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
