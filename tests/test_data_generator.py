"""
Synthetic dataset generator (Stage 1A): determinism, approved counts and
edge cases (invariant group I30). Writes only to pytest's temp folders.
"""

from collections import Counter, defaultdict
from decimal import Decimal

import pytest

from app import ledger
from app.data_validation import TABLES, file_hashes, read_dataset, validate_dataset
from scripts import generate_data as gen


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dataset")
    assert gen.main(["--output-dir", str(folder)]) == 0
    return folder, read_dataset(folder)


def test_same_seed_is_byte_identical(written, tmp_path):
    folder, _ = written
    assert gen.main(["--output-dir", str(tmp_path)]) == 0
    for name in [f"{t}.csv" for t in TABLES] + ["dataset.json"]:
        assert (tmp_path / name).read_bytes() == (folder / name).read_bytes(), name


def test_manifest_matches_files(written):
    folder, dataset = written
    meta = dataset["meta"]
    assert file_hashes(folder) == meta["sha256"]
    assert meta["seed"] == gen.DEFAULT_SEED and meta["generator_version"] == gen.GENERATOR_VERSION
    assert (meta["window_start"], meta["as_of_date"], meta["currency"]) == ("2026-05-01", "2026-11-21", "INR")


def test_other_seeds_differ_and_stay_valid():
    for seed in (1, 42):
        dataset, meta, _ = gen.generate(seed)
        assert validate_dataset(gen.as_validation_input(dataset, meta)) == []
        assert dataset["transactions"] != gen.generate()[0]["transactions"]


def test_refuses_the_application_data_folder():
    with pytest.raises(SystemExit):
        gen.main(["--output-dir", str(gen.PROJECT_ROOT / "data")])


def test_approved_counts(written):
    _, ds = written
    assert {t: len(ds[t]) for t in ("customers", "accounts", "cards", "disputes")} == {
        "customers": 30, "accounts": 42, "cards": 39, "disputes": 8}
    assert 1500 <= len(ds["transactions"]) <= 2500
    assert Counter(Counter(a["customer_id"] for a in ds["accounts"]).values()) == {1: 20, 2: 8, 3: 2}
    assert Counter(c["status"] for c in ds["cards"]) == {"Active": 36, "Blocked": 1, "Expired": 2}
    assert Counter(d["status"] for d in ds["disputes"]) == {
        "Open": 2, "Under Investigation": 2, "Resolved": 3, "Rejected": 1}
    assert Counter(d["resolution_outcome"] for d in ds["disputes"] if d["status"] == "Resolved") == {
        "Full refund": 1, "Partial refund": 1, "No financial adjustment": 1}


def test_dispute_and_refund_counts(written):
    _, ds = written
    by_id = {t["transaction_id"]: t for t in ds["transactions"]}
    refunds = [t for t in ds["transactions"] if t["category"] == "Refund"]
    linked = {d["refund_transaction_id"] for d in ds["disputes"] if d["refund_transaction_id"]}
    kinds = Counter(
        ("dispute" if r["transaction_id"] in linked else "other",
         "full" if r["amount"] == by_id[r["related_transaction_id"]]["amount"] else "partial")
        for r in refunds)
    assert len(refunds) == 12
    assert kinds == {("other", "full"): 7, ("other", "partial"): 3, ("dispute", "full"): 1, ("dispute", "partial"): 1}
    next_month = [r for r in refunds
                  if r["transaction_date"][:7] != by_id[r["related_transaction_id"]]["transaction_date"][:7]]
    assert len(next_month) == 4


def test_edge_cases_are_present(written):
    _, ds = written
    txns = ds["transactions"]
    customers = {c["customer_id"]: c for c in ds["customers"]}
    active = {t["customer_id"] for t in txns}

    assert [(c["customer_id"], c["customer_name"], c["city"]) for c in ds["customers"][:3]] == [
        ("C101", "Arjun Mehta", "Mumbai"), ("C102", "Priya Sharma", "Pune"), ("C103", "Rahul Nair", "Bengaluru")]
    assert set(customers) - active == {"C130"} and customers["C130"]["customer_since"] == "2026-11-18"
    assert len({a["account_id"] for a in ds["accounts"]} - {c["account_id"] for c in ds["cards"]}) == 5
    inside = [a for a in ds["accounts"] if a["opened_on"] >= "2026-05-01"]
    assert len(inside) == 4 and all(a["opening_balance"] == "0.00" for a in inside)
    assert sorted(c["replaces_card_id"] != "" for c in ds["cards"]).count(True) == 2
    assert any(t["transaction_date"] == "2026-05-01" for t in txns)
    assert any(t["transaction_date"] == "2026-11-21" for t in txns)
    assert {t["category"] for t in txns} == set(ledger.DEBIT_CATEGORIES + ledger.CREDIT_CATEGORIES)

    closing = {k: Decimal(v) for k, v in ds["meta"]["closing_balances"].items()}
    low = [a for a in ds["accounts"] if a["customer_id"] == "C117"][0]["account_id"]
    assert Decimal("0") <= closing[low] < Decimal("1000")


def test_no_transfers_between_a_customers_own_accounts(written):
    _, ds = written
    outs = {(t["customer_id"], t["transaction_date"], t["amount"]) for t in ds["transactions"]
            if t["category"] == "Transfer Out"}
    ins = {(t["customer_id"], t["transaction_date"], t["amount"]) for t in ds["transactions"]
           if t["category"] == "Transfer In"}
    assert not outs & ins
    assert "Own account" not in {t["merchant"] for t in ds["transactions"]}


def test_net_cash_flow_identity_every_account_and_month(written):
    _, ds = written
    by_id = {t["transaction_id"]: t for t in ds["transactions"]}
    for account in ds["accounts"]:
        rows = [t for t in ds["transactions"] if t["account_id"] == account["account_id"]]
        for statement in ledger.monthly_statements(account, rows, by_id, ds["meta"]["as_of_date"]):
            f = statement["figures"]
            net_spend = sum(f["net_spend"].values(), Decimal("0"))
            assert f["net_cash_flow"] == f["income"] + f["other_inflows"] - net_spend           # R5
            assert statement["closing"] == statement["opening"] + f["net_cash_flow"]           # R6


def test_default_seed_has_no_guard_skips():
    _, _, diagnostics = gen.generate()
    assert diagnostics["guard_skips"] == []


def test_cli_reports_diagnostics(tmp_path, capsys):
    assert gen.main(["--output-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "refused by the insufficient-balance guard: 0" in out
    assert "budget did not allow them" in out


def test_monthly_closing_balances_cover_every_account_month(written):
    _, ds = written
    monthly = ds["meta"]["monthly_closing_balances"]
    assert set(monthly) == {a["account_id"] for a in ds["accounts"]}
    for account in ds["accounts"]:
        months = ledger.months_between(account["opening_balance_date"], ds["meta"]["as_of_date"])
        assert list(monthly[account["account_id"]]) == months
        assert monthly[account["account_id"]][months[-1]] == ds["meta"]["closing_balances"][account["account_id"]]


def test_premier_and_senior_commitments(written):
    _, ds = written
    segment = {c["customer_id"]: c["segment"] for c in ds["customers"]}
    merchants = defaultdict(set)
    for t in ds["transactions"]:
        merchants[segment[t["customer_id"]]].add(t["merchant"])
    for name in (gen.HOME_LOAN, gen.SIP_FUND, gen.LIFE_INSURANCE):
        assert name in merchants["Premier"]
    for name in (gen.HOUSEHOLD_STAFF, gen.HEALTH_INSURANCE, gen.PHARMACY, gen.SIP_FUND):
        assert name in merchants["Senior"]
    commitments = [t for t in ds["transactions"]
                   if t["merchant"] in (gen.HOME_LOAN, gen.SIP_FUND, gen.LIFE_INSURANCE, gen.HEALTH_INSURANCE,
                                        gen.HOUSEHOLD_STAFF, gen.FAMILY_SUPPORT)]
    assert commitments and all(t["category"] == "Transfer Out" and t["transaction_type"] == "Debit"
                               for t in commitments)
    assert not any(t["category"] == "Rent" and segment[t["customer_id"]] == "Premier" for t in ds["transactions"])


def test_student_cash_and_education_are_affordable(written):
    _, ds = written
    students = {c["customer_id"] for c in ds["customers"] if c["segment"] == "Student"}
    cash = [Decimal(t["amount"]) for t in ds["transactions"]
            if t["customer_id"] in students and t["category"] == "Cash Withdrawal"]
    assert cash and max(cash) <= Decimal("3000")
    education = Counter(t["customer_id"] for t in ds["transactions"]
                        if t["customer_id"] in students and t["category"] == "Education"
                        # the scripted low-balance fee (C117, 20 Nov) is a separate edge case
                        and (t["customer_id"], t["transaction_date"]) != ("C117", "2026-11-20"))
    assert all(count <= gen.STUDENT_EDUCATION["payments"] for count in education.values())
