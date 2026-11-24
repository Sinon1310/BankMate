"""
Ledger calculations (Stage 1A): refunds counted once, cash-basis category
attribution, balances only over complete history, date-based card validity.
"""

from datetime import date
from decimal import Decimal

from app import ledger


def txn(tid, day, kind, amount, category, related=""):
    return {"transaction_id": tid, "transaction_date": day, "transaction_type": kind,
            "amount": amount, "category": category, "related_transaction_id": related}


ACCOUNT = {"account_id": "ACC9001", "opened_on": "2020-01-01",
           "opening_balance_date": "2026-09-01", "opening_balance": "20000.00"}

# Design example: original debit in September, partial refund in October.
CROSS_MONTH = [
    txn("T1", "2026-09-01", "Credit", "40000.00", "Salary"),
    txn("T2", "2026-09-10", "Debit", "6000.00", "Shopping"),
    txn("T3", "2026-09-15", "Debit", "1000.00", "Dining"),
    txn("T4", "2026-10-03", "Credit", "2500.00", "Refund", related="T2"),
    txn("T5", "2026-10-08", "Debit", "3000.00", "Groceries"),
]
BY_ID = {t["transaction_id"]: t for t in CROSS_MONTH}
D = Decimal


def test_cross_month_refund_cash_basis():
    sep = ledger.period_summary(CROSS_MONTH, BY_ID, "2026-09-01", "2026-09-30")
    octo = ledger.period_summary(CROSS_MONTH, BY_ID, "2026-10-01", "2026-10-31")
    whole = ledger.period_summary(CROSS_MONTH, BY_ID)

    # September is not restated by the October refund.
    assert sep == ledger.period_summary(CROSS_MONTH[:3], BY_ID, "2026-09-01", "2026-09-30")
    assert (sep["total_credits"], sep["total_debits"], sep["net_cash_flow"]) == (D("40000"), D("7000"), D("33000"))
    assert sep["net_spend"] == {"Dining": D("1000"), "Shopping": D("6000")}

    # October: the refund is a credit, not income, and reduces Shopping (its original category).
    assert (octo["total_credits"], octo["total_debits"]) == (D("2500"), D("3000"))
    assert (octo["income"], octo["refund_total"], octo["net_cash_flow"]) == (D("0"), D("2500"), D("-500"))
    assert octo["gross_spend"] == {"Groceries": D("3000")}
    assert octo["refunds_by_category"] == {"Shopping": D("2500")}
    assert octo["net_spend"] == {"Groceries": D("3000"), "Shopping": D("-2500")}
    assert "Refund" not in octo["gross_spend"] and "Refund" not in octo["net_spend"]

    # Whole window and the no-double-count identities.
    assert whole["net_spend"]["Shopping"] == D("3500")
    for f in (sep, octo, whole):
        assert f["total_credits"] == f["income"] + f["other_inflows"] + f["refund_total"]          # R1
        assert sum(f["net_spend"].values(), D(0)) == f["total_debits"] - f["refund_total"]         # R4
        assert f["net_cash_flow"] == f["income"] + f["other_inflows"] - sum(f["net_spend"].values(), D(0))  # R5

    # Balances reconcile month by month and day by day.
    statements = ledger.monthly_statements(ACCOUNT, CROSS_MONTH, BY_ID, "2026-10-31")
    assert [(s["month"], s["opening"], s["closing"]) for s in statements] == [
        ("2026-09", D("20000.00"), D("53000.00")), ("2026-10", D("53000.00"), D("52500.00"))]
    assert [ledger.balance_on(ACCOUNT, CROSS_MONTH, day, "2026-10-31")
            for day in ("2026-10-02", "2026-10-03", "2026-10-08")] == [D("53000.00"), D("55500.00"), D("52500.00")]


def test_refund_counted_once_in_cash_flow():
    whole = ledger.period_summary(CROSS_MONTH, BY_ID)
    assert whole["total_credits"] == D("42500")          # salary + refund, once
    assert whole["total_debits"] == D("10000")           # refunds never reduce debits
    assert whole["net_cash_flow"] == D("32500")
    assert ledger.balance_on(ACCOUNT, CROSS_MONTH, "2026-10-31", "2026-10-31") == D("20000.00") + whole["net_cash_flow"]


def test_refund_reduces_net_not_gross_spend():
    whole = ledger.period_summary(CROSS_MONTH, BY_ID)
    assert whole["gross_spend"]["Shopping"] == D("6000")
    assert whole["net_spend"]["Shopping"] == D("3500")


def test_income_excludes_refunds_and_transfers():
    rows = CROSS_MONTH + [txn("T6", "2026-10-09", "Credit", "5000.00", "Transfer In")]
    whole = ledger.period_summary(rows, {t["transaction_id"]: t for t in rows})
    assert whole["income"] == D("40000")
    assert whole["other_inflows"] == D("5000")
    assert whole["refund_total"] == D("2500")


def test_balance_unavailable_before_opening_balance_date():
    assert ledger.balance_on(ACCOUNT, CROSS_MONTH, "2026-08-31", "2026-10-31") is None
    assert ledger.balance_on(ACCOUNT, CROSS_MONTH, "2026-11-01", "2026-10-31") is None
    assert ledger.balance_on(ACCOUNT, CROSS_MONTH, "2026-09-01", "2026-10-31") == D("60000.00")


def test_months_between_covers_partial_months():
    assert ledger.months_between("2026-05-01", "2026-11-21") == [
        "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]


CARD = {"card_id": "CRD9001", "issued_on": "2025-01-10", "expires_on": "2030-01", "blocked_on": "2026-09-04"}


def test_blocked_card_history_remains_valid():
    assert ledger.card_valid_on(CARD, "2026-09-03")          # day before the block
    assert not ledger.card_valid_on(CARD, "2026-09-04")      # block applies from the start of the day
    assert not ledger.card_valid_on(CARD, "2025-01-09")      # before issue
    assert ledger.card_status_on(CARD, "2026-09-03") == "Active"
    assert ledger.card_status_on(CARD, "2026-11-21") == "Blocked"


def test_expiry_and_renewal_overlap():
    old = {"issued_on": "2021-08-10", "expires_on": "2026-08", "blocked_on": ""}
    new = {"issued_on": "2026-08-12", "expires_on": "2031-08", "blocked_on": ""}

    assert ledger.expiry_end("2026-08") == date(2026, 8, 31)
    assert ledger.card_valid_on(old, "2026-08-31") and not ledger.card_valid_on(old, "2026-09-01")
    assert ledger.card_valid_on(old, "2026-08-20") and ledger.card_valid_on(new, "2026-08-20")   # overlap
    assert ledger.card_status_on(old, "2026-11-21") == "Expired"
    assert ledger.card_status_on(new, "2026-08-11") == "Not issued"


# ---------- Stage 1C gate 1: refund accounting, periods, spending, customers ----------

import pytest                                   # noqa: E402

REFUND_CASE = CROSS_MONTH                       # T2 debit 10 Sep, T4 refund of T2 on 3 Oct


def test_refund_changes_the_balance_on_its_posting_date():
    as_of = "2026-10-31"
    before, on = (ledger.balance_on(ACCOUNT, REFUND_CASE, day, as_of) for day in ("2026-10-02", "2026-10-03"))
    assert on - before == D("2500.00")


def test_refund_is_a_credit_in_net_cash_flow_on_its_posting_date():
    day_before = ledger.period_summary(REFUND_CASE, BY_ID, "2026-10-02", "2026-10-02")
    posting_day = ledger.period_summary(REFUND_CASE, BY_ID, "2026-10-03", "2026-10-03")
    assert day_before["total_credits"] == D("0") and posting_day["total_credits"] == D("2500.00")
    assert posting_day["net_cash_flow"] == D("2500.00") and posting_day["refund_total"] == D("2500.00")


def test_refund_reduces_the_original_category_and_is_never_income():
    october = ledger.period_summary(REFUND_CASE, BY_ID, "2026-10-01", "2026-10-31")
    assert october["refunds_by_category"] == {"Shopping": D("2500")}
    assert october["income"] == D("0") and october["other_inflows"] == D("0")
    assert october["spending_net"] == october["spending_gross"] - D("2500")


def test_original_debit_and_refund_stay_separate_entries():
    whole = ledger.period_summary(REFUND_CASE, BY_ID)
    assert (whole["count"], whole["debit_count"], whole["credit_count"]) == (5, 3, 2)
    assert whole["gross_spend"]["Shopping"] == D("6000")            # original untouched
    assert whole["total_credits"] == D("42500")                     # refund counted once
    assert BY_ID["T2"]["amount"] == "6000.00"


SPENDING_CASE = [
    txn("S1", "2026-09-01", "Credit", "50000.00", "Salary"),
    txn("S2", "2026-09-02", "Credit", "5000.00", "Transfer In"),
    txn("S3", "2026-09-05", "Debit", "20000.00", "Transfer Out"),       # e.g. SIP / EMI
    txn("S4", "2026-09-06", "Debit", "3000.00", "Cash Withdrawal"),
    txn("S5", "2026-09-07", "Debit", "4000.00", "Shopping"),
    txn("S6", "2026-09-20", "Credit", "1000.00", "Refund", related="S5"),
]
SPENDING_BY_ID = {t["transaction_id"]: t for t in SPENDING_CASE}


def test_spending_excludes_transfer_out_and_includes_cash():
    f = ledger.period_summary(SPENDING_CASE, SPENDING_BY_ID)
    assert f["transfer_out_total"] == D("20000")
    assert f["spending_gross"] == D("7000")                         # cash 3,000 + shopping 4,000
    assert f["spending_net"] == D("6000")
    assert (f["income"], f["other_inflows"], f["refund_total"]) == (D("50000"), D("5000"), D("1000"))
    assert f["net_cash_flow"] == D("29000")


def assert_identities(f):
    assert f["total_credits"] == f["income"] + f["other_inflows"] + f["refund_total"]
    assert f["total_debits"] == f["transfer_out_total"] + f["spending_gross"]
    assert f["spending_net"] == f["spending_gross"] - f["refund_total"]
    assert f["net_cash_flow"] == f["income"] + f["other_inflows"] - f["transfer_out_total"] - f["spending_net"]
    assert f["spending_net"] == sum((v for c, v in f["net_spend"].items() if c != "Transfer Out"), D(0))
    assert f["count"] == f["debit_count"] + f["credit_count"]


def test_identities_hold_for_worked_examples():
    for rows, by_id in ((SPENDING_CASE, SPENDING_BY_ID), (CROSS_MONTH, BY_ID)):
        assert_identities(ledger.period_summary(rows, by_id))


# ---------- Periods ----------

def test_periods_are_inclusive_and_shared():
    assert ledger.in_period("2026-10-01", "2026-10-01", "2026-10-31")
    assert ledger.in_period("2026-10-31", "2026-10-01", "2026-10-31")
    assert not ledger.in_period("2026-11-01", "2026-10-01", "2026-10-31")
    assert ledger.in_period("2026-10-15")                                  # open both ends
    selected = ledger.select_transactions(CROSS_MONTH, "2026-09-10", "2026-10-03")
    summary = ledger.period_summary(CROSS_MONTH, BY_ID, "2026-09-10", "2026-10-03")
    assert [t["transaction_id"] for t in selected] == ["T2", "T3", "T4"] and summary["count"] == 3


def test_select_transactions_filters():
    rows = CROSS_MONTH
    assert [t["transaction_id"] for t in ledger.select_transactions(rows, transaction_type="Credit")] == ["T1", "T4"]
    assert [t["transaction_id"] for t in ledger.select_transactions(rows, category="Shopping")] == ["T2"]
    assert [t["transaction_id"] for t in ledger.select_transactions(rows, min_amount="3000")] == ["T1", "T2", "T5"]
    tagged = [dict(t, account_id="ACC1" if t["transaction_id"] in ("T1", "T2") else "ACC2") for t in rows]
    assert [t["transaction_id"] for t in ledger.select_transactions(tagged, account_id="ACC2")] == ["T3", "T4", "T5"]


def test_clamp_period():
    window = ("2026-05-01", "2026-11-21")
    p = ledger.clamp_period(None, None, *window)
    assert (p.start, p.end, p.clamped, p.empty) == (date(2026, 5, 1), date(2026, 11, 21), False, False)
    p = ledger.clamp_period("2026-01-01", "2026-12-31", *window)
    assert (p.start, p.end, p.clamped) == (date(2026, 5, 1), date(2026, 11, 21), True)
    p = ledger.clamp_period("2026-10-01", "2026-10-31", *window)
    assert (p.start, p.end, p.clamped) == (date(2026, 10, 1), date(2026, 10, 31), False)
    assert ledger.clamp_period("2027-01-01", "2027-02-01", *window).empty
    with pytest.raises(ValueError):
        ledger.clamp_period("2026-10-31", "2026-10-01", *window)


# ---------- Customer level ----------

ACCOUNTS = [
    {"account_id": "A1", "opened_on": "2020-01-01", "opening_balance_date": "2026-09-01", "opening_balance": "1000.00"},
    {"account_id": "A2", "opened_on": "2026-09-15", "opening_balance_date": "2026-09-15", "opening_balance": "0.00"},
]
TWO_ACCOUNTS = [
    dict(txn("X1", "2026-09-01", "Credit", "5000.00", "Salary"), account_id="A1"),
    dict(txn("X2", "2026-09-15", "Credit", "2000.00", "Transfer In"), account_id="A2"),
    dict(txn("X3", "2026-10-02", "Debit", "500.00", "Dining"), account_id="A2"),
    dict(txn("X4", "2026-10-05", "Debit", "1500.00", "Groceries"), account_id="A1"),
]
TWO_BY_ID = {t["transaction_id"]: t for t in TWO_ACCOUNTS}
WINDOW = ("2026-09-01", "2026-10-31")


def test_customer_balance_excludes_unopened_accounts():
    assert ledger.customer_balance_on(ACCOUNTS, TWO_ACCOUNTS, "2026-09-14", *WINDOW) == D("6000.00")
    assert ledger.customer_balance_on(ACCOUNTS, TWO_ACCOUNTS, "2026-09-15", *WINDOW) == D("8000.00")
    assert ledger.customer_balance_on(ACCOUNTS, TWO_ACCOUNTS, "2026-08-31", *WINDOW) is None
    assert ledger.customer_balance_on(ACCOUNTS, TWO_ACCOUNTS, "2026-11-01", *WINDOW) is None
    assert ledger.customer_opening_balance(ACCOUNTS, TWO_ACCOUNTS, "2026-09-01", *WINDOW) == D("1000.00")
    assert ledger.customer_opening_balance(ACCOUNTS, TWO_ACCOUNTS, "2026-10-01", *WINDOW) == D("8000.00")


def test_customer_monthly_chain_and_partial_periods():
    rows = ledger.customer_monthly(ACCOUNTS, TWO_ACCOUNTS, TWO_BY_ID, *WINDOW)
    assert [(r["month"], r["opening_balance"], r["closing_balance"]) for r in rows] == [
        ("2026-09", D("1000.00"), D("8000.00")), ("2026-10", D("8000.00"), D("6000.00"))]
    for row in rows:
        assert row["closing_balance"] == row["opening_balance"] + row["figures"]["net_cash_flow"]

    partial = ledger.customer_monthly(ACCOUNTS, TWO_ACCOUNTS, TWO_BY_ID, *WINDOW, start="2026-09-10", end="2026-10-03")
    assert [(str(r["start"]), str(r["end"]), r["figures"]["count"]) for r in partial] == [
        ("2026-09-10", "2026-09-30", 1), ("2026-10-01", "2026-10-03", 1)]
    assert partial[0]["opening_balance"] == D("6000.00") and partial[-1]["closing_balance"] == D("7500.00")
    assert ledger.customer_monthly(ACCOUNTS, TWO_ACCOUNTS, TWO_BY_ID, *WINDOW, start="2027-01-01") == []


def test_own_account_transfer_pairs():
    rows = [
        dict(txn("O1", "2026-09-05", "Debit", "1000.00", "Transfer Out"), account_id="A1", customer_id="C1"),
        dict(txn("I1", "2026-09-05", "Credit", "1000.00", "Transfer In"), account_id="A2", customer_id="C1"),
        dict(txn("I2", "2026-09-05", "Credit", "1000.00", "Transfer In"), account_id="B1", customer_id="C2"),
        dict(txn("I3", "2026-09-06", "Credit", "1000.00", "Transfer In"), account_id="A2", customer_id="C1"),
    ]
    assert ledger.own_account_transfer_pairs(rows) == [("O1", "I1")]


# ---------- Approved dataset (generated in memory) ----------

@pytest.fixture(scope="module")
def approved():
    from scripts import generate_data as gen
    dataset, meta, _ = gen.generate()
    return dataset, meta


def test_customer_figures_match_the_generators_independent_balances(approved):
    dataset, meta = approved
    txns = dataset["transactions"]
    by_id = {t["transaction_id"]: t for t in txns}
    window = (meta["window_start"], meta["as_of_date"])

    for customer in dataset["customers"]:
        cid = customer["customer_id"]
        accounts = [a for a in dataset["accounts"] if a["customer_id"] == cid]
        rows = [t for t in txns if t["customer_id"] == cid]
        monthly = ledger.customer_monthly(accounts, rows, by_id, *window)

        for row in monthly:
            assert_identities(row["figures"])
            assert row["closing_balance"] == row["opening_balance"] + row["figures"]["net_cash_flow"]
            recorded = sum((D(meta["monthly_closing_balances"][a["account_id"]][row["month"]])
                            for a in accounts if row["month"] in meta["monthly_closing_balances"][a["account_id"]]), D(0))
            assert row["closing_balance"] == recorded, (cid, row["month"])

        whole = ledger.period_summary(rows, by_id)
        assert_identities(whole)
        assert sum((r["figures"]["net_cash_flow"] for r in monthly), D(0)) == whole["net_cash_flow"]
        assert monthly[-1]["closing_balance"] == sum((D(meta["closing_balances"][a["account_id"]]) for a in accounts), D(0))


def test_approved_dataset_has_no_own_account_transfers(approved):
    dataset, _ = approved
    assert ledger.own_account_transfer_pairs(dataset["transactions"]) == []


def test_select_transactions_by_merchant():
    rows = [dict(txn("M1", "2026-09-01", "Debit", "10.00", "Shopping"), merchant="HomeNest"),
            dict(txn("M2", "2026-09-02", "Credit", "10.00", "Refund", related="M1"), merchant="Refund - HomeNest"),
            dict(txn("M3", "2026-09-03", "Debit", "5.00", "Dining"), merchant="QuickBite")]
    assert [t["transaction_id"] for t in ledger.select_transactions(rows, merchant="homenest")] == ["M1", "M2"]
    assert [t["transaction_id"] for t in ledger.select_transactions(rows, merchant="QUICK")] == ["M3"]
    assert ledger.select_transactions(rows, merchant="Swiggy") == []
    assert len(ledger.select_transactions(rows, merchant=None)) == 3
