"""
Tests for ui/data.py: formatting, search, filtering and aggregation.

Totals are cross-checked against the backend's get_customer_metrics tool
so the Explorer, Analytics and the assistant can never disagree.
"""

import json
from datetime import date
from decimal import Decimal

import pytest

from app.data_loader import customers, transactions
from app.tools import get_customer_metrics
from ui import data


# ---------- Formatting ----------

@pytest.mark.parametrize("value, expected", [
    (0, "₹0"),
    (800, "₹800"),
    ("2500", "₹2,500"),
    (Decimal("95000"), "₹95,000"),
    (100000, "₹1,00,000"),
    (12345678, "₹1,23,45,678"),
    (Decimal("1234.50"), "₹1,234.50"),
    (-4500, "\u2212₹4,500"),            # Unicode minus, not a hyphen
])
def test_format_inr_uses_indian_grouping(value, expected):
    assert data.format_inr(value) == expected


def test_format_inr_signed():
    assert data.format_inr(16000, signed=True) == "+₹16,000"
    assert data.format_inr(0, signed=True) == "₹0"
    assert data.format_inr(-300, signed=True) == "\u2212₹300"


def test_format_day_and_initials():
    assert data.format_day("2026-11-03") == "3 Nov 2026"
    assert data.format_day(date(2026, 11, 15)) == "15 Nov 2026"
    assert data.initials("Arjun Mehta") == "AM"
    assert data.initials("Priya") == "P"
    assert data.initials("") == "?"


# ---------- Customers ----------

def test_search_matches_id_name_city_and_type_case_insensitively():
    assert [c["customer_id"] for c in data.search_customers("c102")] == ["C102"]
    assert [c["customer_id"] for c in data.search_customers("ARJUN")] == ["C101"]
    assert [c["customer_id"] for c in data.search_customers("bengaluru")] == ["C103"]
    assert {c["customer_id"] for c in data.search_customers("savings")} == {"C101", "C103"}
    assert data.search_customers("nobody") == []
    assert len(data.search_customers("  ")) == len(customers)


def test_customer_metrics_come_from_backend_tool():
    for customer in customers:
        cid = customer["customer_id"]
        tool = json.loads(get_customer_metrics.invoke({"customer_id": cid}))
        metrics = data.customer_metrics(cid)

        assert metrics["total_debit"] == Decimal(tool["total_debit"])
        assert metrics["total_credit"] == Decimal(tool["total_credit"])
        assert metrics["net_flow"] == Decimal(tool["net_transaction_flow"])
        assert metrics["count"] == tool["transaction_count"]


def test_customer_metrics_none_for_customer_without_transactions():
    assert data.customer_metrics("C999") is None


def test_customer_transactions_newest_first():
    rows = data.customer_transactions("C101")
    dates = [r["transaction_date"] for r in rows]

    assert dates == sorted(dates, reverse=True)
    assert {r["customer_id"] for r in rows} == {"C101"}


# ---------- Filtering & aggregation ----------

def test_flow_summary_matches_tool_for_every_customer():
    for customer in customers:
        cid = customer["customer_id"]
        tool = json.loads(get_customer_metrics.invoke({"customer_id": cid}))
        summary = data.flow_summary(data.filter_transactions(customer_ids=[cid]))

        assert summary["total_debit"] == Decimal(tool["total_debit"])
        assert summary["total_credit"] == Decimal(tool["total_credit"])
        assert summary["net_flow"] == Decimal(tool["net_transaction_flow"])


def test_portfolio_summary_is_sum_of_customers():
    summary = data.flow_summary(transactions)
    per_customer = [data.customer_metrics(c["customer_id"]) for c in customers]

    assert summary["total_debit"] == sum(m["total_debit"] for m in per_customer)
    assert summary["total_credit"] == sum(m["total_credit"] for m in per_customer)
    assert summary["count"] == len(transactions)
    assert summary["debit_count"] + summary["credit_count"] == len(transactions)


def test_filters_by_customer_type_and_inclusive_dates():
    debits = data.filter_transactions(types=["Debit"])
    assert debits and all(r["transaction_type"] == "Debit" for r in debits)

    window = data.filter_transactions(start=date(2026, 11, 2), end=date(2026, 11, 3))
    assert window and all("2026-11-02" <= r["transaction_date"] <= "2026-11-03" for r in window)

    combined = data.filter_transactions(customer_ids=["C103"], types=["credit"])
    assert [r["transaction_id"] for r in combined] == ["T009"]

    assert data.filter_transactions(customer_ids=[]) == []


def test_date_bounds_cover_dataset():
    first, last = data.date_bounds()
    assert first == min(date.fromisoformat(r["transaction_date"]) for r in transactions)
    assert last == max(date.fromisoformat(r["transaction_date"]) for r in transactions)


def test_debit_by_merchant_ranks_and_ignores_credits():
    items = data.debit_by_merchant(transactions)
    amounts = [amount for _, amount in items]

    assert amounts == sorted(amounts, reverse=True)
    assert "Salary" not in dict(items)  # credits only
    assert dict(items)["Amazon"] == Decimal("4000")  # 2,500 + 1,500 across customers


def test_daily_flow_totals_match_summary():
    days = data.daily_flow(transactions)
    summary = data.flow_summary(transactions)

    assert [d for d, _, _ in days] == sorted(d for d, _, _ in days)
    assert sum(c for _, c, _ in days) == summary["total_credit"]
    assert sum(d for _, _, d in days) == summary["total_debit"]


def test_net_flow_by_customer_matches_tool():
    for cid, name, net in data.net_flow_by_customer(transactions):
        tool = json.loads(get_customer_metrics.invoke({"customer_id": cid}))
        assert net == Decimal(tool["net_transaction_flow"])
        assert name == data.get_customer(cid)["customer_name"]
