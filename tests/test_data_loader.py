"""
Data loader (Stage 1B): configurable folder, validation before exposure,
legacy-shaped records for existing consumers, no fallback on failure.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import app.data_loader as dl
from app.data_validation import TABLES
from scripts import generate_data as gen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "data"

# SHA-256 manifest of the approved Stage 1A.1 dataset (seed 20261121).
APPROVED_SHA256 = {
    "customers.csv": "8efe2b8e9c8d44c1d565f97277dc112e8eff4a255e0aed8b537edb243f2fb933",
    "accounts.csv": "6482ff641436af5fadb14e6b24d85303ba8b9808a86dc6738bd946daac93a921",
    "cards.csv": "ed958c69416f4e23b92f8ed38386b41f58e83dce099661dd6945b522ced83a52",
    "transactions.csv": "fdd47ee049a9b0a47d02713aa42e423aeb285d53fba0aee425f079ea41c53b2e",
    "disputes.csv": "2ae4bafcefb7a321d9191416c8eb8cbfb5424be5727126435d1a93899cfb6a98",
}

LEGACY_CUSTOMERS = (
    "customer_id,customer_name,city,account_type\n"
    "C101,Arjun Mehta,Mumbai,Savings\n"
    "C102,Priya Sharma,Pune,Current\n"
)
LEGACY_TRANSACTIONS = (
    "transaction_id,customer_id,transaction_date,transaction_type,amount,merchant\n"
    "T001,C101,2026-11-01,Debit,2500,Amazon\n"
    "T007,C102,2026-11-02,Credit,30000,Salary\n"
)


@pytest.fixture(scope="module")
def approved(tmp_path_factory):
    folder = tmp_path_factory.mktemp("approved")
    assert gen.main(["--output-dir", str(folder)]) == 0
    return folder


@pytest.fixture
def fixture_copy(tmp_path):
    folder = tmp_path / "data"
    shutil.copytree(FIXTURE, folder)
    return folder


def refresh_manifest(folder):
    """Re-sign dataset.json after a deliberate edit, so validation (not the hash check) is exercised."""
    meta_path = folder / "dataset.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["sha256"] = {f"{t}.csv": hashlib.sha256((folder / f"{t}.csv").read_bytes()).hexdigest() for t in TABLES}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def replace_in(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def load_error(folder):
    with pytest.raises(dl.DataLoadError) as raised:
        dl.load_dataset(folder)
    return "\n".join(raised.value.errors)


# ---------- Configuration ----------

def test_resolve_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(dl.DATA_DIR_ENV, raising=False)
    assert dl.resolve_data_dir() == PROJECT_ROOT / "data"
    assert dl.resolve_data_dir("tests/fixtures/data") == FIXTURE          # relative to the project, not cwd
    assert dl.resolve_data_dir(str(tmp_path)) == tmp_path
    monkeypatch.setenv(dl.DATA_DIR_ENV, "tests/fixtures/data")
    assert dl.resolve_data_dir() == FIXTURE
    monkeypatch.setenv(dl.DATA_DIR_ENV, "   ")
    assert dl.resolve_data_dir() == PROJECT_ROOT / "data"


def test_module_uses_the_configured_fixture():
    assert dl.DATA_DIR == FIXTURE
    dataset = dl.get_dataset()
    assert dataset.format == dl.FORMAT_V1
    assert set(dataset.tables) == set(TABLES)
    assert len(dataset.tables["accounts"]) == 3


# ---------- Successful loading ----------

def test_fixture_exposes_original_record_shape():
    dataset = dl.load_dataset(FIXTURE)
    assert dataset.customers == [
        {"customer_id": "C101", "customer_name": "Arjun Mehta", "city": "Mumbai", "account_type": "Savings"},
        {"customer_id": "C102", "customer_name": "Priya Sharma", "city": "Pune", "account_type": "Current"},
        {"customer_id": "C103", "customer_name": "Rahul Nair", "city": "Bengaluru", "account_type": "Savings"},
    ]
    assert dataset.transactions[0] == {"transaction_id": "T001", "customer_id": "C101",
                                       "transaction_date": "2026-11-01", "transaction_type": "Debit",
                                       "amount": "2500", "merchant": "Amazon"}
    assert all(list(t) == dl.LEGACY_TRANSACTION_COLUMNS for t in dataset.transactions)


def test_find_functions():
    assert dl.find_customer("C101")["customer_name"] == "Arjun Mehta"
    assert dl.find_customer("C999") is None
    rows = dl.find_transactions("C101")
    assert [t["transaction_id"] for t in rows] == ["T001", "T002", "T003", "T004", "T005"]
    rows.clear()                                        # callers get a copy
    assert len(dl.find_transactions("C101")) == 5
    assert dl.find_transactions("C999") == []


def test_approved_dataset_loads(approved):
    assert {name: hashlib.sha256((approved / name).read_bytes()).hexdigest()
            for name in APPROVED_SHA256} == APPROVED_SHA256
    dataset = dl.load_dataset(approved)
    assert dataset.format == dl.FORMAT_V1
    assert (len(dataset.customers), len(dataset.transactions)) == (30, 2460)
    assert dataset.find_customer("C101")["account_type"] == "Savings + Current"
    assert dataset.find_customer("C130")["account_type"] == "Savings"
    assert dataset.find_transactions("C130") == []
    assert all(list(t) == dl.LEGACY_TRANSACTION_COLUMNS for t in dataset.transactions)
    assert dataset.meta["as_of_date"] == "2026-11-21"


def test_legacy_format_folder_loads(tmp_path):
    (tmp_path / "customers.csv").write_text(LEGACY_CUSTOMERS, encoding="utf-8")
    (tmp_path / "transactions.csv").write_text(LEGACY_TRANSACTIONS, encoding="utf-8")
    dataset = dl.load_dataset(tmp_path)
    assert dataset.format == dl.FORMAT_LEGACY
    assert dataset.find_customer("C102")["account_type"] == "Current"
    assert [t["transaction_id"] for t in dataset.find_transactions("C101")] == ["T001"]


# ---------- Failures (no partial data is ever exposed) ----------

def test_missing_folder(tmp_path):
    assert "does not exist" in load_error(tmp_path / "nowhere")


def test_missing_file(fixture_copy):
    (fixture_copy / "transactions.csv").unlink()
    assert "missing file transactions.csv" in load_error(fixture_copy)


def test_v1_files_without_dataset_json(fixture_copy):
    (fixture_copy / "dataset.json").unlink()
    assert "dataset.json is missing" in load_error(fixture_copy)


def test_malformed_dataset_json(fixture_copy):
    (fixture_copy / "dataset.json").write_text("{not json", encoding="utf-8")
    assert "not valid JSON" in load_error(fixture_copy)


def test_edited_file_fails_the_manifest(fixture_copy):
    replace_in(fixture_copy / "transactions.csv", ",2500,", ",2600,")
    assert "transactions.csv does not match the dataset.json sha256 manifest" in load_error(fixture_copy)


def test_malformed_schema(fixture_copy):
    replace_in(fixture_copy / "accounts.csv", "opening_balance_date", "opening_date")
    refresh_manifest(fixture_copy)
    assert "I1 accounts.csv: header must be" in load_error(fixture_copy)


def test_invalid_reference(fixture_copy):
    replace_in(fixture_copy / "accounts.csv", "ACC0103,C103", "ACC0103,C999")
    refresh_manifest(fixture_copy)
    assert "I2 accounts ACC0103: customer_id='C999' does not exist in customers" in load_error(fixture_copy)


def test_invalid_amount(fixture_copy):
    replace_in(fixture_copy / "transactions.csv", ",2500,", ",-2500,")
    refresh_manifest(fixture_copy)
    assert "I1 transactions T001: amount='-2500'" in load_error(fixture_copy)


def test_generated_dataset_requires_month_end_balances(approved, tmp_path):
    folder = tmp_path / "data"
    shutil.copytree(approved, folder)
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    del meta["monthly_closing_balances"]
    (folder / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")
    assert "monthly_closing_balances missing" in load_error(folder)


def test_legacy_format_invalid_reference(tmp_path):
    (tmp_path / "customers.csv").write_text(LEGACY_CUSTOMERS, encoding="utf-8")
    (tmp_path / "transactions.csv").write_text(LEGACY_TRANSACTIONS.replace("C102", "C109"), encoding="utf-8")
    assert "I2 transactions T007: customer_id='C109' does not exist" in load_error(tmp_path)


def test_legacy_format_malformed_header(tmp_path):
    (tmp_path / "customers.csv").write_text(LEGACY_CUSTOMERS.replace("city", "town"), encoding="utf-8")
    (tmp_path / "transactions.csv").write_text(LEGACY_TRANSACTIONS, encoding="utf-8")
    assert "I1 customers.csv: header must be" in load_error(tmp_path)


def test_invalid_configured_folder_fails_at_import_without_fallback(fixture_copy):
    replace_in(fixture_copy / "transactions.csv", ",2500,", ",2600,")
    env = dict(os.environ, BANKMATE_DATA_DIR=str(fixture_copy), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", "import app.data_loader"],
                            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "DataLoadError" in result.stderr and str(fixture_copy) in result.stderr
