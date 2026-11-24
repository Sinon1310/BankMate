"""
Customer intelligence tools (Stage 1C, gate 4) over the approved dataset.

The approved dataset is generated into a temp folder (byte-identical to the
reviewed candidate) and loaded through the real loader. Expected figures
are computed directly from the rows, independently of app.ledger.
"""

import json
import re
from collections import Counter
from decimal import Decimal as D

import pytest

import app.data_loader as dl
import app.tools as tools
from scripts import generate_data as gen

AMOUNT = re.compile(r"^-?\d+\.\d{2}$")


@pytest.fixture(scope="module")
def approved_dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("approved")
    assert gen.main(["--output-dir", str(folder)]) == 0
    return dl.load_dataset(folder)


@pytest.fixture
def data(monkeypatch, approved_dataset):
    monkeypatch.setattr(dl, "_dataset", approved_dataset)
    return approved_dataset


def call(tool, **args):
    raw = tool.invoke(args)
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def txns(data, customer_id):
    return [t for t in data.tables["transactions"] if t["customer_id"] == customer_id]


def total(rows, kind):
    return sum((D(t["amount"]) for t in rows if t["transaction_type"] == kind), D("0"))


def all_pages(customer_id, **filters):
    rows, offset, pages = [], 0, []
    while True:
        page = call(tools.get_customer_transactions, customer_id=customer_id, offset=offset, **filters)
        pages.append(page)
        rows += page["rows"]
        if not page["has_more"]:
            assert "next_offset" not in page
            return rows, pages
        offset = page["next_offset"]


# =====================================================================
# Transactions
# =====================================================================

def test_default_page(data):
    page = call(tools.get_customer_transactions, customer_id="C101")
    n = len(txns(data, "C101"))
    assert (page["total_matching"], page["returned_count"], page["limit"], page["offset"]) == (n, 20, 20, 0)
    assert page["has_more"] is True and page["next_offset"] == 20
    assert page["columns"] == tools.TRANSACTION_COLUMNS and len(page["rows"]) == 20
    assert "not the complete list" in page["note"]


def test_maximum_limit(data):
    assert call(tools.get_customer_transactions, customer_id="C113", limit=50)["returned_count"] == 50
    assert call(tools.get_customer_transactions, customer_id="C113", limit=51)["status"] == "INVALID_ARGUMENT"


def test_pagination_covers_every_row_once_in_order(data):
    rows, pages = all_pages("C102", limit=50)
    expected = sorted(txns(data, "C102"), key=lambda t: (t["transaction_date"], t["transaction_id"]), reverse=True)
    assert [r[0] for r in rows] == [t["transaction_id"] for t in expected]
    assert len(pages) == 4 and pages[-1]["returned_count"] == len(expected) - 150
    assert all(p["matching_totals"] == pages[0]["matching_totals"] for p in pages)


def test_offset_past_the_end(data):
    n = len(txns(data, "C103"))
    page = call(tools.get_customer_transactions, customer_id="C103", offset=n + 5)
    assert page["rows"] == [] and page["has_more"] is False and "next_offset" not in page
    assert page["total_matching"] == n and "past the last" in page["note"]


def test_totals_cover_all_matches_not_the_page(data):
    rows = txns(data, "C113")
    page = call(tools.get_customer_transactions, customer_id="C113", limit=5)
    totals = page["matching_totals"]
    assert D(totals["debits"]) == total(rows, "Debit") and D(totals["credits"]) == total(rows, "Credit")
    assert totals["debit_count"] + totals["credit_count"] == len(rows) and page["returned_count"] == 5
    assert D(totals["net"]) == total(rows, "Credit") - total(rows, "Debit")


def test_inclusive_date_boundaries(data):
    day = txns(data, "C101")[10]["transaction_date"]
    page = call(tools.get_customer_transactions, customer_id="C101", start_date=day, end_date=day, limit=50)
    expected = {t["transaction_id"] for t in txns(data, "C101") if t["transaction_date"] == day}
    assert {r[0] for r in page["rows"]} == expected and page["total_matching"] == len(expected)
    october = call(tools.get_customer_transactions, customer_id="C101", start_date="2026-10-01",
                   end_date="2026-10-31", limit=50)
    assert october["total_matching"] == sum(t["transaction_date"][:7] == "2026-10" for t in txns(data, "C101"))


def test_category_type_amount_and_account_filters(data):
    rows = txns(data, "C101")
    dining = call(tools.get_customer_transactions, customer_id="C101", category="dining", limit=50)
    assert dining["total_matching"] == sum(t["category"] == "Dining" for t in rows)
    assert {r[5] for r in dining["rows"]} == {"Dining"} and dining["filters"]["category"] == "Dining"
    credits = call(tools.get_customer_transactions, customer_id="C101", transaction_type="Credit", limit=50)
    assert credits["total_matching"] == sum(t["transaction_type"] == "Credit" for t in rows)
    big = call(tools.get_customer_transactions, customer_id="C101", min_amount=5000, limit=50)
    assert big["total_matching"] == sum(D(t["amount"]) >= 5000 for t in rows)
    current = call(tools.get_customer_transactions, customer_id="C101", account_id="ACC1002", limit=50)
    assert current["total_matching"] == sum(t["account_id"] == "ACC1002" for t in rows)
    combined = call(tools.get_customer_transactions, customer_id="C101", account_id="ACC1001",
                    transaction_type="Debit", start_date="2026-09-01", end_date="2026-09-30")
    assert combined["total_matching"] == sum(
        t["account_id"] == "ACC1001" and t["transaction_type"] == "Debit" and t["transaction_date"][:7] == "2026-09"
        for t in rows)


def test_multi_account_customer(data):
    accounts = [a["account_id"] for a in data.tables["accounts"] if a["customer_id"] == "C102"]
    assert len(accounts) == 3
    per_account = [call(tools.get_customer_transactions, customer_id="C102", account_id=a)["total_matching"]
                   for a in accounts]
    assert sum(per_account) == call(tools.get_customer_transactions, customer_id="C102")["total_matching"]


def test_another_customers_account_is_denied_without_an_existence_oracle(data):
    other = call(tools.get_customer_transactions, customer_id="C101", account_id="ACC1006")
    missing = call(tools.get_customer_transactions, customer_id="C101", account_id="ACC9999")
    assert other["status"] == "ACCESS_DENIED" and other == missing


def test_empty_results(data):
    page = call(tools.get_customer_transactions, customer_id="C101", category="Pension")
    assert page["total_matching"] == 0 and page["rows"] == [] and page["has_more"] is False
    assert page["matching_totals"]["debits"] == "0.00" and "No transactions match" in page["note"]
    c130 = call(tools.get_customer_transactions, customer_id="C130")
    assert c130["total_matching"] == 0


def test_refunds_appear_as_separate_linked_entries(data):
    page = call(tools.get_customer_transactions, customer_id="C104", category="Refund")
    refund = page["rows"][0]
    original = next(t for t in data.tables["transactions"] if t["transaction_id"] == refund[9])
    assert refund[3] == "Credit" and original["transaction_type"] == "Debit"
    shopping = call(tools.get_customer_transactions, customer_id="C104", start_date=original["transaction_date"],
                    end_date=original["transaction_date"], category=original["category"])
    assert original["transaction_id"] in {r[0] for r in shopping["rows"]}       # original still listed in full


def test_dates_outside_the_window_are_clamped_or_empty(data):
    clamped = call(tools.get_customer_transactions, customer_id="C103", start_date="2026-01-01")
    assert clamped["period"] == {"start": "2026-05-01", "end": "2026-11-21", "clamped": True}
    empty = call(tools.get_customer_transactions, customer_id="C103", start_date="2027-01-01")
    assert empty["total_matching"] == 0 and "outside the data" in empty["period"]["note"]


def test_amounts_and_card_references_are_safe(data):
    page = call(tools.get_customer_transactions, customer_id="C111", limit=50)
    assert all(AMOUNT.match(r[4]) for r in page["rows"])
    assert all(r[8] == "" or re.fullmatch(r"\d{4}", r[8]) for r in page["rows"])
    assert "CRD" not in json.dumps(page)                                           # no internal card IDs


# =====================================================================
# Profile
# =====================================================================

def test_single_account_profile(data):
    profile = call(tools.get_customer_profile, customer_id="C103")
    (account,) = profile["accounts"]
    assert account["balance"] == profile["total_balance"] == data.meta["closing_balances"][account["account_id"]]
    assert (profile["customer_name"], profile["city"], profile["as_of_date"]) == ("Rahul Nair", "Bengaluru", "2026-11-21")
    assert "Available balance is not tracked" in profile["notes"][0]


def test_multi_account_profile(data):
    profile = call(tools.get_customer_profile, customer_id="C101")
    balances = {a["account_id"]: D(a["balance"]) for a in profile["accounts"]}
    assert balances == {a: D(data.meta["closing_balances"][a]) for a in ("ACC1001", "ACC1002")}
    assert D(profile["total_balance"]) == sum(balances.values())
    assert profile["account_type"] == "Savings + Current" and profile["transaction_count"] == len(txns(data, "C101"))


def test_customer_without_transactions(data):
    profile = call(tools.get_customer_profile, customer_id="C130")
    assert profile["transaction_count"] == 0 and profile["total_balance"] == "0.00"
    assert profile["cards"] == [] and profile["disputes"]["total"] == 0


def test_card_status_and_historical_validity(data):
    blocked = call(tools.get_customer_profile, customer_id="C111")["cards"]
    old = next(c for c in blocked if c["status"] == "Blocked")
    new = next(c for c in blocked if c.get("replaces_card_last4"))
    assert (old["blocked_on"], old["valid_until"]) == ("2026-09-04", "2026-09-03")
    assert new["replaces_card_last4"] == old["card_last4"] and new["status"] == "Active"
    renewed = call(tools.get_customer_profile, customer_id="C116")["cards"]
    assert {c["status"] for c in renewed} == {"Expired", "Active"}
    expired = next(c for c in renewed if c["status"] == "Expired")
    assert expired["valid_until"] == "2026-08-31" and "blocked_on" not in expired


def test_profiles_never_expose_full_numbers(data):
    for customer in data.tables["customers"]:
        text = tools.get_customer_profile.invoke({"customer_id": customer["customer_id"]})
        assert not re.search(r"\d{9,}", text.replace("-", ""))                  # no long digit runs
        for card in json.loads(text)["cards"]:
            assert re.fullmatch(r"\d{4}", card["card_last4"]) and "card_id" not in card


def test_profile_dispute_summary(data):
    profile = call(tools.get_customer_profile, customer_id="C111")
    assert profile["disputes"] == {"total": 1, "open": 0, "under_investigation": 0, "resolved": 1, "rejected": 0}


# =====================================================================
# Metrics
# =====================================================================

def metrics(**args):
    return call(tools.get_customer_metrics, **args)


def test_metrics_totals_and_classification(data):
    for cid in ("C101", "C104", "C107", "C110", "C113"):
        rows, m = txns(data, cid), metrics(customer_id=cid)
        by_cat = Counter()
        for t in rows:
            by_cat[t["category"]] += D(t["amount"])
        assert D(m["total_debit"]) == total(rows, "Debit") and D(m["total_credit"]) == total(rows, "Credit")
        assert D(m["income"]) == sum(by_cat[c] for c in ("Salary", "Pension", "Business Income", "Interest"))
        assert D(m["transfer_in"]) == by_cat["Transfer In"] and D(m["transfer_out"]) == by_cat["Transfer Out"]
        assert D(m["refunds"]) == by_cat["Refund"]
        assert D(m["spending_gross"]) == total(rows, "Debit") - by_cat["Transfer Out"]          # cash included
        assert D(m["spending_net"]) == D(m["spending_gross"]) - by_cat["Refund"]
        assert D(m["total_credit"]) == D(m["income"]) + D(m["transfer_in"]) + D(m["refunds"])  # refunds once
        assert m["transaction_count"] == len(rows)


def test_category_net_spending(data):
    m = metrics(customer_id="C104")
    assert all(D(c["net"]) == D(c["gross"]) - D(c["refunds"]) for c in m["by_category"])
    assert sum(D(c["net"]) for c in m["by_category"]) == D(m["spending_net"])
    assert "Transfer Out" not in {c["category"] for c in m["by_category"]}
    assert "Refund" not in {c["category"] for c in m["by_category"]}


def test_cross_month_refund(data):
    # Scripted case R2: HomeNest 5,890.00 on 27 Jun (C106), refunded in full on 6 Jul.
    june = metrics(customer_id="C106", start_date="2026-06-01", end_date="2026-06-30")
    july = metrics(customer_id="C106", start_date="2026-07-01", end_date="2026-07-31")
    both = metrics(customer_id="C106", start_date="2026-06-01", end_date="2026-07-31")
    shop = lambda m: next(c for c in m["by_category"] if c["category"] == "Shopping")
    assert D(shop(june)["refunds"]) == 0 and D(shop(july)["refunds"]) == D("5890.00")
    july_rows = [t for t in txns(data, "C106") if t["transaction_date"][:7] == "2026-07"]
    july_income = sum((D(t["amount"]) for t in july_rows
                       if t["category"] in ("Salary", "Pension", "Business Income", "Interest")), D("0"))
    assert D(july["refunds"]) >= D("5890.00") and D(july["income"]) == july_income     # refund is not income
    assert D(shop(both)["net"]) == D(shop(june)["net"]) + D(shop(july)["net"])
    assert D(both["total_credit"]) == D(june["total_credit"]) + D(july["total_credit"])          # counted once


def test_monthly_summary_and_reconciliation(data):
    for cid in ("C101", "C102", "C126"):
        m = metrics(customer_id=cid)
        months = m["monthly"]
        assert [x["month"] for x in months] == ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]
        assert sum(D(x["net_transaction_flow"]) for x in months) == D(m["net_transaction_flow"])
        assert D(m["balance"]["closing"]) == D(m["balance"]["opening"]) + D(m["net_transaction_flow"])
        assert D(months[-1]["closing_balance"]) == D(m["balance"]["closing"])
        accounts = m["accounts"]
        assert sum(D(a["closing_balance"]) for a in accounts) == D(m["balance"]["closing"])
        for a in accounts:
            assert D(a["closing_balance"]) == D(data.meta["closing_balances"][a["account_id"]])
            assert D(a["closing_balance"]) == D(a["opening_balance"]) + D(a["net_transaction_flow"])


def test_partial_period_balances(data):
    m = metrics(customer_id="C101", start_date="2026-08-15", end_date="2026-09-15")
    assert [x["month"] for x in m["monthly"]] == ["2026-08", "2026-09"]
    assert D(m["balance"]["closing"]) == D(m["balance"]["opening"]) + D(m["net_transaction_flow"])


def test_metrics_decimal_format(data):
    m = metrics(customer_id="C113")
    values = [m[k] for k in ("total_debit", "total_credit", "net_transaction_flow", "income", "transfer_in",
                             "transfer_out", "refunds", "spending_gross", "spending_net")]
    values += [c[k] for c in m["by_category"] for k in ("gross", "refunds", "net")]
    assert all(AMOUNT.match(v) for v in values)


def test_metrics_empty_period_and_no_transactions(data):
    empty = metrics(customer_id="C101", start_date="2027-01-01")
    assert empty["transaction_count"] == 0 and empty["total_debit"] == "0.00" and "balance" not in empty
    assert metrics(customer_id="C130") == "No transactions found for customer C130."


def test_metrics_spell_out_what_is_not_a_balance(data):
    notes = " ".join(metrics(customer_id="C101")["notes"])
    assert "It is not a balance" in notes and "refunds are not income" in notes


# =====================================================================
# Disputes
# =====================================================================

def disputes(**args):
    return call(tools.get_customer_disputes, **args)


def test_all_statuses_and_filter(data):
    seen = Counter()
    for customer in data.tables["customers"]:
        for d in disputes(customer_id=customer["customer_id"])["disputes"]:
            seen[d["status"]] += 1
    assert seen == {"Open": 2, "Under Investigation": 2, "Resolved": 3, "Rejected": 1}
    assert disputes(customer_id="C111", status="open")["total"] == 0
    assert disputes(customer_id="C111", status="Resolved")["total"] == 1


def test_resolved_without_and_with_refund(data):
    (no_refund,) = disputes(customer_id="C111")["disputes"]
    assert no_refund["status"] == "Resolved" and no_refund["resolution_outcome"] == "No financial adjustment"
    assert "refund" not in no_refund
    (refunded,) = disputes(customer_id="C101")["disputes"]
    refund = next(t for t in data.tables["transactions"] if t["transaction_id"] == refunded["refund"]["transaction_id"])
    assert refunded["resolution_outcome"] == "Full refund" and refund["related_transaction_id"] == refunded["transaction_id"]
    assert refunded["refund"]["amount"] == refunded["amount"]
    (open_case,) = disputes(customer_id="C106")["disputes"]
    assert open_case["status"] == "Open" and "resolved_on" not in open_case and "resolution_outcome" not in open_case


def test_customer_isolation(data):
    for customer in data.tables["customers"]:
        cid = customer["customer_id"]
        own = {t["transaction_id"] for t in txns(data, cid)}
        assert all(d["transaction_id"] in own for d in disputes(customer_id=cid)["disputes"])
    assert disputes(customer_id="C103")["disputes"] == []


def test_disputes_are_not_ledger_entries(data):
    m = metrics(customer_id="C101")
    rows = txns(data, "C101")
    assert D(m["total_debit"]) == total(rows, "Debit") and D(m["total_credit"]) == total(rows, "Credit")
    assert "not part of any debit or credit total" in disputes(customer_id="C101")["notes"][0]


def test_dispute_list_is_bounded(data, monkeypatch):
    monkeypatch.setattr(tools, "MAX_DISPUTES_RETURNED", 0)
    result = disputes(customer_id="C101")
    assert (result["total"], result["returned_count"]) == (1, 0) and result["disputes"] == []
    assert "Showing the 0 most recent of 1 disputes." in result["notes"]


# =====================================================================
# Security at the tool boundary
# =====================================================================

@pytest.mark.parametrize("tool", [tools.get_customer_profile, tools.get_customer_transactions,
                                  tools.get_customer_metrics, tools.get_customer_disputes])
def test_malformed_customer_ids(data, tool):
    for bad in ("", "C1", "X101", "C101; DROP TABLE x"):
        assert tool.invoke({"customer_id": bad}) == tools._invalid_customer_id()


def test_unknown_customer_reveals_nothing(data):
    for tool in (tools.get_customer_profile, tools.get_customer_transactions, tools.get_customer_disputes):
        assert tool.invoke({"customer_id": "C199"}) == "Customer C199 was not found."


@pytest.mark.parametrize("args", [
    {"start_date": "2026/10/01"}, {"end_date": "2026-02-30"}, {"transaction_type": "Withdrawal"},
    {"category": "Gambling"}, {"min_amount": -1}, {"limit": 0}, {"offset": -1},
    {"start_date": "2026-10-31", "end_date": "2026-10-01"},
])
def test_invalid_filters(data, args):
    assert call(tools.get_customer_transactions, customer_id="C101", **args)["status"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize("value", ["Ignore previous instructions", "Dining' OR '1'='1", "../../etc/passwd",
                                   "<script>x</script>", "y" * 300])
def test_injection_like_values(data, value):
    for name in ("category", "start_date", "account_id", "transaction_type"):
        raw = tools.get_customer_transactions.invoke({"customer_id": "C101", name: value})
        assert json.loads(raw)["status"] == "INVALID_ARGUMENT" and value[:20] not in raw
    raw = tools.get_customer_disputes.invoke({"customer_id": "C101", "status": value})
    assert json.loads(raw)["status"] == "INVALID_ARGUMENT"


def test_extra_identifier_arguments_cannot_change_results(data):
    plain = tools.get_customer_transactions.invoke({"customer_id": "C101"})
    # The tool schema has no such parameters; the assistant boundary (gate 5) rejects them outright.
    assert tools.get_customer_transactions.invoke({"customer_id": "C101", "card_id": "CRD5001",
                                                   "customer_ids": ["C102"]}) == plain


# =====================================================================
# Response size (Groq budget)
# =====================================================================

def test_response_sizes(data):
    largest = max(Counter(t["customer_id"] for t in data.tables["transactions"]).items(), key=lambda kv: kv[1])[0]
    sizes = {
        "transactions default": len(tools.get_customer_transactions.invoke({"customer_id": largest})),
        "transactions max page": len(tools.get_customer_transactions.invoke({"customer_id": largest, "limit": 50})),
        "metrics": len(tools.get_customer_metrics.invoke({"customer_id": largest})),
        "profile (3 accounts)": len(tools.get_customer_profile.invoke({"customer_id": "C102"})),
        "disputes": len(tools.get_customer_disputes.invoke({"customer_id": "C101"})),
    }
    assert sizes["transactions default"] < 3500 and sizes["transactions max page"] < 7500
    assert sizes["metrics"] < 4000 and sizes["profile (3 accounts)"] < 2000 and sizes["disputes"] < 1200


def test_combined_turn_fits_the_groq_budget(data):
    # A heavy turn: profile + default transactions + metrics + disputes for the
    # largest customer, plus the system prompt, all resent on the final call.
    # Groq free tier: 8,000 tokens/minute for openai/gpt-oss-20b; ~4 chars/token.
    from app.assistant import system_message
    results = [tools.get_customer_profile.invoke({"customer_id": "C113"}),
               tools.get_customer_transactions.invoke({"customer_id": "C113"}),
               tools.get_customer_metrics.invoke({"customer_id": "C113"}),
               tools.get_customer_disputes.invoke({"customer_id": "C113"})]
    tool_chars = sum(len(r) for r in results)
    estimated_tokens = (tool_chars + len(system_message.content)) / 4
    assert estimated_tokens < 4000, (tool_chars, estimated_tokens)


# =====================================================================
# Legacy format (data/ until Stage 1D)
# =====================================================================

def test_legacy_format_keeps_original_behaviour(monkeypatch, tmp_path):
    (tmp_path / "customers.csv").write_text(
        "customer_id,customer_name,city,account_type\nC101,Arjun Mehta,Mumbai,Savings\n", encoding="utf-8")
    (tmp_path / "transactions.csv").write_text(
        "transaction_id,customer_id,transaction_date,transaction_type,amount,merchant\n"
        "T001,C101,2026-11-01,Debit,2500,Amazon\nT003,C101,2026-11-03,Credit,25000,Salary\n", encoding="utf-8")
    monkeypatch.setattr(dl, "_dataset", dl.load_dataset(tmp_path))
    assert call(tools.get_customer_profile, customer_id="C101") == {
        "customer_id": "C101", "customer_name": "Arjun Mehta", "city": "Mumbai", "account_type": "Savings"}
    assert len(call(tools.get_customer_transactions, customer_id="C101")) == 2
    assert call(tools.get_customer_metrics, customer_id="C101")["total_debit"] == "2500"
    assert call(tools.get_customer_disputes, customer_id="C101")["status"] == "NOT_AVAILABLE"
    assert call(tools.get_customer_transactions, customer_id="C101", account_id="ACC1001")["status"] == "ACCESS_DENIED"


def test_legacy_format_applies_supported_filters_and_refuses_the_rest(monkeypatch, tmp_path):
    (tmp_path / "customers.csv").write_text(
        "customer_id,customer_name,city,account_type\nC101,Arjun Mehta,Mumbai,Savings\n", encoding="utf-8")
    (tmp_path / "transactions.csv").write_text(
        "transaction_id,customer_id,transaction_date,transaction_type,amount,merchant\n"
        "T001,C101,2026-11-01,Debit,2500,Amazon\nT003,C101,2026-11-03,Credit,25000,Salary\n", encoding="utf-8")
    monkeypatch.setattr(dl, "_dataset", dl.load_dataset(tmp_path))
    debits = call(tools.get_customer_transactions, customer_id="C101", transaction_type="Debit")
    assert [r["transaction_id"] for r in debits] == ["T001"]
    assert [r["transaction_id"] for r in call(tools.get_customer_transactions, customer_id="C101",
                                               start_date="2026-11-02", end_date="2026-11-03")] == ["T003"]
    assert call(tools.get_customer_transactions, customer_id="C101", min_amount=30000) == \
        "No transactions match these filters for customer C101."
    assert call(tools.get_customer_transactions, customer_id="C101", category="Dining")["status"] == "NOT_AVAILABLE"
    november_3 = call(tools.get_customer_metrics, customer_id="C101", start_date="2026-11-03", end_date="2026-11-03")
    assert (november_3["total_debit"], november_3["total_credit"], november_3["transaction_count"]) == ("0", "25000", 1)
    assert call(tools.get_customer_metrics, customer_id="C101", start_date="2026-12-01") == \
        "No transactions match these dates for customer C101."


def test_fixture_metrics_keep_their_original_values():
    m = call(tools.get_customer_metrics, customer_id="C101")
    assert (D(m["total_debit"]), D(m["total_credit"]), D(m["net_transaction_flow"]), m["transaction_count"]) == (
        D("9000"), D("25000"), D("16000"), 5)


# =====================================================================
# Merchant filter (gate 7)
# =====================================================================

def merchant_rows(data, customer_id, needle):
    return [t for t in txns(data, customer_id) if needle.lower() in t["merchant"].lower()]


def test_merchant_filter_matches_case_insensitively_including_refunds(data):
    page = call(tools.get_customer_transactions, customer_id="C106", merchant="homenest")
    expected = merchant_rows(data, "C106", "HomeNest")
    assert page["filters"] == {"merchant": "homenest"} and page["total_matching"] == len(expected) == 5
    assert {r[0] for r in page["rows"]} == {t["transaction_id"] for t in expected}
    assert {r[6] for r in page["rows"]} == {"HomeNest", "Refund - HomeNest"}       # purchase and its refund
    upper = call(tools.get_customer_transactions, customer_id="C106", merchant="HOMENEST")
    assert upper["rows"] == page["rows"]


def test_merchant_filter_totals_and_pagination(data):
    expected = merchant_rows(data, "C101", "UrbanPower")
    first = call(tools.get_customer_transactions, customer_id="C101", merchant="UrbanPower", limit=3)
    assert (first["total_matching"], first["returned_count"], first["has_more"], first["next_offset"]) == (
        len(expected), 3, True, 3)
    assert D(first["matching_totals"]["debits"]) == total(expected, "Debit")
    rows, offset = first["rows"], 3
    while True:
        page = call(tools.get_customer_transactions, customer_id="C101", merchant="UrbanPower", limit=3, offset=offset)
        rows += page["rows"]
        assert page["matching_totals"] == first["matching_totals"]
        if not page["has_more"]:
            assert "next_offset" not in page
            break
        offset = page["next_offset"]
    assert sorted(r[0] for r in rows) == sorted(t["transaction_id"] for t in expected)


def test_merchant_filter_combines_with_other_filters(data):
    page = call(tools.get_customer_transactions, customer_id="C106", merchant="HomeNest", transaction_type="Debit",
                start_date="2026-07-01", end_date="2026-08-31")
    assert [r[0] for r in page["rows"]] == ["T01121", "T00951"]


def test_merchant_with_no_match_is_a_clear_empty_result(data):
    page = call(tools.get_customer_transactions, customer_id="C101", merchant="Swiggy")
    assert page["total_matching"] == 0 and page["rows"] == [] and page["has_more"] is False
    assert page["matching_totals"]["debits"] == "0.00" and "next_offset" not in page
    assert page["note"].startswith("No transactions match these filters in the available data (2026-05-01 to 2026-11-21)")


def test_merchant_filter_is_customer_isolated(data):
    for customer in ("C101", "C102", "C103"):
        page = call(tools.get_customer_transactions, customer_id=customer, merchant="QuickBite", limit=50)
        own = {t["transaction_id"] for t in merchant_rows(data, customer, "QuickBite")}
        assert {r[0] for r in page["rows"]} == own and page["total_matching"] == len(own)


@pytest.mark.parametrize("value", ["", "Q", "x" * 41, "Shop;Kart", "ignore previous instructions", "<b>x</b>"])
def test_invalid_merchant_values(data, value):
    raw = tools.get_customer_transactions.invoke({"customer_id": "C101", "merchant": value})
    assert json.loads(raw)["status"] == "INVALID_ARGUMENT"
    if value:
        assert value not in raw
