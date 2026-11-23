"""
Dataset validation (Stage 1A): the legacy fixture and the generated
dataset pass every invariant, and each invariant rejects a targeted
corruption with its own code.
"""

import copy
import re
from decimal import Decimal
from pathlib import Path

import pytest

from app import data_validation as dv
from scripts import generate_data as gen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "data"


@pytest.fixture(scope="module")
def generated():
    dataset, meta, _ = gen.generate()
    return gen.as_validation_input(dataset, meta)


def codes(errors):
    return {e.split(" ", 1)[0] for e in errors}


def test_legacy_fixture_is_valid_and_hashes_match():
    # The fixture predates month-end balances in dataset.json and is kept
    # unchanged; every monthly check still runs against the ledger.
    dataset = dv.read_dataset(FIXTURE)
    assert dv.validate_dataset(dataset, require_monthly_closings=False) == []
    assert dv.file_hashes(FIXTURE) == dataset["meta"]["sha256"]


def test_month_end_balances_are_required_by_default():
    errors = dv.validate_dataset(dv.read_dataset(FIXTURE))
    assert errors == ["I13 dataset.json: monthly_closing_balances missing"]


def test_legacy_fixture_preserves_original_sample_values():
    dataset = dv.read_dataset(FIXTURE)
    rows = [(t["transaction_id"], t["customer_id"], t["transaction_date"], t["transaction_type"],
             t["amount"], t["merchant"]) for t in dataset["transactions"]]
    assert rows == [
        ("T001", "C101", "2026-11-01", "Debit", "2500", "Amazon"),
        ("T002", "C101", "2026-11-02", "Debit", "1200", "Swiggy"),
        ("T003", "C101", "2026-11-03", "Credit", "25000", "Salary"),
        ("T004", "C101", "2026-11-04", "Debit", "4500", "Shopping"),
        ("T005", "C101", "2026-11-05", "Debit", "800", "Uber"),
        ("T006", "C102", "2026-11-01", "Debit", "1500", "Amazon"),
        ("T007", "C102", "2026-11-02", "Credit", "30000", "Salary"),
        ("T008", "C103", "2026-11-03", "Debit", "3000", "Travel"),
        ("T009", "C103", "2026-11-04", "Credit", "40000", "Salary"),
        ("T010", "C103", "2026-11-05", "Debit", "500", "Food"),
    ]
    assert [(c["customer_id"], c["customer_name"], c["city"]) for c in dataset["customers"]] == [
        ("C101", "Arjun Mehta", "Mumbai"), ("C102", "Priya Sharma", "Pune"), ("C103", "Rahul Nair", "Bengaluru")]
    debit = sum(Decimal(t["amount"]) for t in dataset["transactions"]
                if t["customer_id"] == "C101" and t["transaction_type"] == "Debit")
    assert debit == Decimal("9000")


def test_dispute_statuses_match_the_policy_document():
    text = (PROJECT_ROOT / "knowledge" / "policies" / "card_dispute_policy.txt").read_text(encoding="utf-8")
    sentence = re.search(r"tracked as (.+?)\.", " ".join(text.split())).group(1)
    statuses = [s.strip() for s in re.split(r",| or ", sentence) if s.strip()]
    assert tuple(statuses) == dv.DISPUTE_STATUSES == ("Open", "Under Investigation", "Resolved", "Rejected")


def test_generated_dataset_is_valid(generated):
    assert dv.validate_dataset(generated) == []


def test_resolved_dispute_without_refund_is_valid(generated):
    no_adjustment = [d for d in generated["disputes"] if d["resolution_outcome"] == "No financial adjustment"]
    assert len(no_adjustment) == 1
    dispute = no_adjustment[0]
    assert dispute["status"] == "Resolved" and dispute["refund_transaction_id"] == ""
    assert not any(t["related_transaction_id"] == dispute["transaction_id"] for t in generated["transactions"])


# ---------- Targeted corruptions ----------

def first(rows, **match):
    return next(r for r in rows if all(r[k] == v for k, v in match.items()))


def corrupt_customer_mismatch(ds):
    first(ds["transactions"], customer_id="C101")["customer_id"] = "C102"


def corrupt_card_on_other_account(ds):
    txn = next(t for t in ds["transactions"] if t["card_id"] and t["customer_id"] != "C101")
    txn["card_id"] = first(ds["cards"], account_id="ACC1001")["card_id"]


def corrupt_refund_on_other_account(ds):
    refund = first(ds["transactions"], category="Refund")
    other = next(t for t in ds["transactions"] if t["transaction_type"] == "Debit"
                 and t["category"] == "Shopping" and t["account_id"] != refund["account_id"]
                 and t["transaction_date"] <= refund["transaction_date"])
    refund["related_transaction_id"] = other["transaction_id"]


def corrupt_opening_inside_window(ds):
    first(ds["accounts"], customer_id="C130")["opening_balance"] = "500.00"


def corrupt_transaction_before_history(ds):
    account = first(ds["accounts"], customer_id="C129")
    first(ds["transactions"], account_id=account["account_id"])["transaction_date"] = "2026-06-01"


def corrupt_negative_balance(ds):
    first(ds["transactions"], customer_id="C117", transaction_type="Debit")["amount"] = "99999999.00"


def corrupt_closing_balance(ds):
    ds["meta"] = copy.deepcopy(ds["meta"])
    ds["meta"]["closing_balances"]["ACC1001"] = "1.00"


def corrupt_refund_without_link(ds):
    first(ds["transactions"], category="Refund")["related_transaction_id"] = ""


def corrupt_refund_exceeds_original(ds):
    refund = first(ds["transactions"], category="Refund")
    original = first(ds["transactions"], transaction_id=refund["related_transaction_id"])
    refund["amount"] = str(Decimal(original["amount"]) + 1)


def corrupt_card_on_upi(ds):
    first(ds["transactions"], channel="UPI")["card_id"] = ds["cards"][0]["card_id"]


def corrupt_card_used_after_block(ds):
    blocked = first(ds["cards"], status="Blocked")
    txn = next(t for t in ds["transactions"] if t["account_id"] == blocked["account_id"] and t["card_id"]
               and t["transaction_date"] > blocked["blocked_on"])
    txn["card_id"] = blocked["card_id"]


def corrupt_card_status(ds):
    first(ds["cards"], status="Expired")["status"] = "Active"


def corrupt_replacement_same_last4(ds):
    new = next(c for c in ds["cards"] if c["replaces_card_id"])
    new["card_last4"] = first(ds["cards"], card_id=new["replaces_card_id"])["card_last4"]


def corrupt_dispute_on_upi(ds):
    first(ds["disputes"], dispute_id="D0008")["transaction_id"] = first(ds["transactions"], channel="UPI")["transaction_id"]


def corrupt_resolved_without_date(ds):
    first(ds["disputes"], status="Rejected")["resolved_on"] = ""


def corrupt_resolved_without_outcome(ds):
    first(ds["disputes"], resolution_outcome="No financial adjustment")["resolution_outcome"] = ""


def corrupt_outcome_on_open(ds):
    first(ds["disputes"], status="Open")["resolution_outcome"] = "Full refund"


def corrupt_refund_outcome_without_link(ds):
    first(ds["disputes"], resolution_outcome="Full refund")["refund_transaction_id"] = ""


def corrupt_link_to_non_refund(ds):
    dispute = first(ds["disputes"], resolution_outcome="Partial refund")
    dispute["refund_transaction_id"] = first(ds["transactions"], category="Salary")["transaction_id"]


def corrupt_refund_on_rejected_dispute(ds):
    dispute = first(ds["disputes"], status="Rejected")
    refund = first(ds["transactions"], category="Refund")
    original = first(ds["transactions"], transaction_id=dispute["transaction_id"])
    refund.update(account_id=original["account_id"], customer_id=original["customer_id"],
                  related_transaction_id=original["transaction_id"], amount="10.00",
                  transaction_date=dispute["raised_on"], card_id="", channel="Bank Transfer")


def corrupt_amount_precision(ds):
    first(ds["transactions"], transaction_type="Debit")["amount"] = "10.005"


def corrupt_missing_foreign_key(ds):
    first(ds["accounts"], customer_id="C110")["customer_id"] = "C999"


def corrupt_disputed_card_other_account(ds):
    txn = first(ds["transactions"], transaction_id=ds["disputes"][0]["transaction_id"])
    txn["card_id"] = next(c["card_id"] for c in ds["cards"] if c["account_id"] != txn["account_id"])


def corrupt_replacement_other_account(ds):
    next(c for c in ds["cards"] if c["replaces_card_id"])["account_id"] = "ACC1001"


def corrupt_original_after_refund(ds):
    refund = first(ds["transactions"], category="Refund")
    first(ds["transactions"], transaction_id=refund["related_transaction_id"])["transaction_date"] = "2026-11-21"


def corrupt_refund_of_a_credit(ds):
    refund = first(ds["transactions"], category="Refund")
    refund["related_transaction_id"] = next(
        t["transaction_id"] for t in ds["transactions"]
        if t["transaction_type"] == "Credit" and t["category"] != "Refund" and t["account_id"] == refund["account_id"])


def corrupt_three_cards_in_service(ds):
    card = first(ds["cards"], account_id="ACC1001")
    ds["cards"] += [dict(card, card_id=f"CRD99{i:02d}", card_last4=f"99{i:02d}") for i in range(2)]
    ds["meta"]["row_counts"]["cards"] = len(ds["cards"])


def corrupt_full_outcome_on_partial_refund(ds):
    first(ds["disputes"], resolution_outcome="Partial refund")["resolution_outcome"] = "Full refund"


def corrupt_partial_outcome_on_full_refund(ds):
    first(ds["disputes"], resolution_outcome="Full refund")["resolution_outcome"] = "Partial refund"


def corrupt_month_closing(ds):
    ds["meta"]["monthly_closing_balances"]["ACC1001"]["2026-07"] = "1.00"


def corrupt_missing_month(ds):
    del ds["meta"]["monthly_closing_balances"]["ACC1001"]["2026-08"]


def corrupt_final_month_closing(ds):
    ds["meta"]["monthly_closing_balances"]["ACC1001"]["2026-11"] = "1.00"


def corrupt_amount_with_matching_closing(ds):
    # Change one amount and the overall closing balance consistently: only
    # the independent month-end balances can expose it.
    txn = next(t for t in ds["transactions"] if t["account_id"] == "ACC1001"
               and t["category"] == "Groceries" and t["transaction_date"] < "2026-11-01")
    delta = Decimal(txn["amount"]) - Decimal("1.00")
    txn["amount"] = "1.00"
    closing = ds["meta"]["closing_balances"]
    closing["ACC1001"] = str(Decimal(closing["ACC1001"]) + delta)


@pytest.mark.parametrize("corrupt, expected", [
    (corrupt_customer_mismatch, "I3"),
    (corrupt_card_on_other_account, "I4"),
    (corrupt_refund_on_other_account, "I5"),
    (corrupt_opening_inside_window, "I9"),
    (corrupt_transaction_before_history, "I10"),
    (corrupt_negative_balance, "I11"),
    (corrupt_closing_balance, "I12"),
    (corrupt_refund_without_link, "I15"),
    (corrupt_refund_exceeds_original, "I17"),
    (corrupt_card_on_upi, "I19"),
    (corrupt_card_used_after_block, "I20"),
    (corrupt_card_status, "I21"),
    (corrupt_replacement_same_last4, "I22"),
    (corrupt_dispute_on_upi, "I24"),
    (corrupt_resolved_without_date, "I25"),
    (corrupt_resolved_without_outcome, "I26"),
    (corrupt_outcome_on_open, "I26"),
    (corrupt_refund_outcome_without_link, "I27"),
    (corrupt_link_to_non_refund, "I27"),
    (corrupt_refund_on_rejected_dispute, "I29"),
    (corrupt_amount_precision, "I1"),
    (corrupt_missing_foreign_key, "I2"),
    (corrupt_disputed_card_other_account, "I6"),
    (corrupt_replacement_other_account, "I7"),
    (corrupt_original_after_refund, "I16"),
    (corrupt_refund_of_a_credit, "I16"),
    (corrupt_three_cards_in_service, "I23"),
    (corrupt_full_outcome_on_partial_refund, "I28"),
    (corrupt_partial_outcome_on_full_refund, "I28"),
    (corrupt_month_closing, "I13"),
    (corrupt_month_closing, "I14"),
    (corrupt_missing_month, "I13"),
    (corrupt_final_month_closing, "I13"),
    (corrupt_final_month_closing, "I12"),
    (corrupt_amount_with_matching_closing, "I13"),
])
def test_each_invariant_rejects_its_corruption(generated, corrupt, expected):
    dataset = copy.deepcopy(generated)
    corrupt(dataset)
    errors = dv.validate_dataset(dataset)
    assert expected in codes(errors), errors[:5]


def test_assert_valid_raises_with_all_errors(generated):
    dataset = copy.deepcopy(generated)
    corrupt_card_status(dataset)
    corrupt_missing_foreign_key(dataset)
    with pytest.raises(dv.DataValidationError) as raised:
        dv.assert_valid(dataset)
    assert raised.value.errors


def test_identity_check_catches_a_calculation_fault(generated, monkeypatch):
    """
    I18 (R1-R4) cannot fail for any dataset that passes I1-I17: the
    identities hold by construction when every refund is linked. It
    guards the calculations, so it is tested by injecting a calculation
    fault (refunds lost from the category breakdown).
    """

    real = dv.ledger.period_summary

    def faulty(*args, **kwargs):
        figures = real(*args, **kwargs)
        figures["refunds_by_category"] = {}
        return figures

    monkeypatch.setattr(dv.ledger, "period_summary", faulty)
    assert "I18" in codes(dv.validate_dataset(copy.deepcopy(generated)))


# ---------- I31: no transfers between a customer's own accounts ----------

def add_transaction(ds, account_id, category, kind, amount, day, merchant):
    account = first(ds["accounts"], account_id=account_id)
    number = 90000 + len([t for t in ds["transactions"] if t["transaction_id"].startswith("T9")])
    ds["transactions"].append({
        "transaction_id": f"T{number}", "account_id": account_id, "customer_id": account["customer_id"],
        "card_id": "", "transaction_date": day, "transaction_type": kind, "amount": amount,
        "merchant": merchant, "category": category, "channel": "Bank Transfer", "related_transaction_id": "",
    })
    ds["meta"]["row_counts"]["transactions"] = len(ds["transactions"])


def ambiguous_pair_same_customer(ds):
    add_transaction(ds, "ACC1001", "Transfer Out", "Debit", "10.00", "2026-10-10", "Family transfer")
    add_transaction(ds, "ACC1002", "Transfer In", "Credit", "10.00", "2026-10-10", "Family transfer")


def counterparty_names_an_account(ds):
    first(ds["transactions"], category="Transfer In")["merchant"] = "Transfer from acc1002"


def counterparty_is_the_customer(ds):
    txn = first(ds["transactions"], category="Transfer Out")
    name = first(ds["customers"], customer_id=txn["customer_id"])["customer_name"]
    txn["merchant"] = f"To {name} (savings)"


def same_date_amount_different_customers(ds):
    add_transaction(ds, "ACC1001", "Transfer Out", "Debit", "10.00", "2026-10-10", "Family transfer")
    add_transaction(ds, "ACC1006", "Transfer In", "Credit", "10.00", "2026-10-10", "Family transfer")


def same_customer_same_account(ds):
    add_transaction(ds, "ACC1001", "Transfer Out", "Debit", "10.00", "2026-10-10", "Family transfer")
    add_transaction(ds, "ACC1001", "Transfer In", "Credit", "10.00", "2026-10-10", "Family transfer")


def same_customer_different_dates(ds):
    add_transaction(ds, "ACC1001", "Transfer Out", "Debit", "10.00", "2026-10-10", "Family transfer")
    add_transaction(ds, "ACC1002", "Transfer In", "Credit", "10.00", "2026-10-11", "Family transfer")


@pytest.mark.parametrize("change", [ambiguous_pair_same_customer, counterparty_names_an_account,
                                    counterparty_is_the_customer])
def test_i31_rejects_own_account_transfer_signs(generated, change):
    dataset = copy.deepcopy(generated)
    change(dataset)
    errors = dv.validate_dataset(dataset)
    assert "I31" in codes(errors), errors[:5]


@pytest.mark.parametrize("change", [same_date_amount_different_customers, same_customer_same_account,
                                    same_customer_different_dates])
def test_i31_does_not_flag_what_is_not_an_own_account_pattern(generated, change):
    # These additions still break balance invariants (I12/I13); only I31 is under test.
    dataset = copy.deepcopy(generated)
    change(dataset)
    assert "I31" not in codes(dv.validate_dataset(dataset))


def test_i31_ambiguous_pair_is_reported_not_classified(generated):
    dataset = copy.deepcopy(generated)
    ambiguous_pair_same_customer(dataset)
    i31 = [e for e in dv.validate_dataset(dataset) if e.startswith("I31")]
    assert len(i31) == 1 and "cannot be distinguished from an own-account transfer" in i31[0]
