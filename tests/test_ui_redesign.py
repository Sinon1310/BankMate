"""
UI sprint: data layer (ui/insights.py) correctness and the redesigned pages
on the full approved dataset (generated into a temp folder).
"""

import re
from decimal import Decimal as D
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.data_loader as dl
from app import ledger
from scripts import generate_data as gen
from ui import insights, navigation

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture(scope="module")
def approved_dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("approved")
    assert gen.main(["--output-dir", str(folder)]) == 0
    return dl.load_dataset(folder)


@pytest.fixture
def data(monkeypatch, approved_dataset):
    monkeypatch.setattr(dl, "_dataset", approved_dataset)
    return approved_dataset


def page(path, **query):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if path != navigation.HOME["path"]:
        for key, value in query.items():
            at.query_params[key] = value
        at.switch_page(path).run()
    assert not at.exception, at.exception
    return at


def text(at):
    return "\n".join(m.value for m in at.markdown if "<style" not in m.value)


# ---------- Data layer: every figure comes from app.ledger ----------

def test_overview_matches_ledger_and_recorded_balances(data):
    o = insights.overview()
    assert (o["customers"], o["accounts"], o["cards"], o["transactions"], o["disputes"]) == (30, 42, 39, 2460, 8)
    assert o["active_cards"] == 36 and o["open_disputes"] == 4
    assert o["total_balance"] == sum(D(v) for v in data.meta["closing_balances"].values())
    tx = data.tables["transactions"]
    assert o["figures"] == ledger.period_summary(tx, {t["transaction_id"]: t for t in tx})


def test_monthly_closing_balances_match_the_generator(data):
    months = insights.monthly()
    recorded = data.meta["monthly_closing_balances"]
    for row in months:
        expected = sum((D(m[row["month"]]) for m in recorded.values() if row["month"] in m), D(0))
        assert row["closing_balance"] == expected, row["month"]


def test_transactions_page_totals_cover_every_match(data):
    first = insights.transactions_page(merchant="homenest", limit=3)
    every = [t for t in data.tables["transactions"] if "homenest" in t["merchant"].lower()]
    assert first["total"] == len(every) and len(first["rows"]) == 3
    assert first["totals"]["total_debits"] == sum(D(t["amount"]) for t in every if t["transaction_type"] == "Debit")
    second = insights.transactions_page(merchant="homenest", limit=3, offset=3)
    assert not {r["transaction_id"] for r in first["rows"]} & {r["transaction_id"] for r in second["rows"]}


def test_disputes_show_a_refund_only_when_one_exists(data):
    by_id = {d["dispute_id"]: d for d in insights.disputes()}
    assert by_id["D0003"]["status"] == "Resolved" and by_id["D0003"]["refund"] is None
    assert by_id["D0001"]["refund"]["amount"] == D("2750.00")
    assert sum(1 for d in by_id.values() if d["refund"]) == 2
    assert insights.dispute_counts() == {"Open": 2, "Under Investigation": 2, "Resolved": 3, "Rejected": 1}


def test_account_summaries_close_at_recorded_balances(data):
    for row in insights.account_summaries():
        assert row["closing"] == D(data.meta["closing_balances"][row["account_id"]])
        assert row["closing"] == row["opening"] + row["net"]


def test_categories_exclude_transfers_and_net_out_refunds(data):
    f = insights.figures()
    cats = insights.categories(f)
    assert "Transfer Out" not in {c for c, *_ in cats}
    assert sum((net for *_, net in cats), D(0)) == f["spending_net"]


# ---------- Pages on the full dataset ----------

def test_analytics_never_renders_the_whole_dataset(data):
    at = page(navigation.ANALYTICS["path"])
    html = text(at)
    assert "Showing <strong>1–25</strong> of <strong>2,460</strong>" in html
    assert len(html) < 120_000                       # previously ~928 KB for all 2,460 rows
    assert "Closing balance" in html and "Net transaction flow" in html and "Spending (net)" in html


def test_transaction_explorer_filters_and_pages(data):
    at = page(navigation.TRANSACTIONS["path"])
    assert "of <strong>2,460</strong>" in text(at)
    at.button(key="transactions-next").click().run()
    assert "Showing <strong>26–50</strong>" in text(at)
    at.text_input(key="txn-merchant").set_value("HomeNest").run()
    html = text(at)
    assert "Showing <strong>1–" in html                     # page resets when filters change
    expected = sum(1 for t in data.tables["transactions"] if "homenest" in t["merchant"].lower())
    assert f'Matching transactions</div><div class="bm-kpi-value">{expected:,}</div>' in html
    at.text_input(key="txn-min").set_value("abc").run()
    assert "Minimum amount must be a positive number" in text(at)


def test_disputes_page(data):
    at = page(navigation.DISPUTES["path"])
    html = text(at)
    assert html.count("<tr>") == 1 + 8
    row = re.search(r"D0003.*?</tr>", html, re.S).group(0)
    assert "No financial adjustment" in row and "None</td>" in row          # resolved, no refund
    at.radio(key="dispute-status").set_value("Resolved").run()
    assert text(at).count("<tr>") == 1 + 3


def test_customer_360_shows_safe_card_references_and_pages(data):
    at = page(navigation.CUSTOMERS["path"])
    at.button(key="cust-btn-C111").click().run()
    assert not at.exception, at.exception
    html = text(at)
    assert "Siddharth Rao" in html and "•••• 0291" in html and "Blocked" in html
    assert not re.search(r"\d{9,}", html.replace(",", ""))                   # no long numbers
    at.button(key="customer-transactions-next").click().run()
    assert "Showing <strong>11–20</strong>" in text(at)


def test_disputes_page_empty_state_on_the_fixture():
    at = page(navigation.DISPUTES["path"])
    assert "No disputes" in text(at)


def test_ui_has_no_second_ai_pipeline():
    root = Path(APP).parent / "ui"
    sources = "\n".join(p.read_text() for p in root.rglob("*.py"))
    assert "ChatGroq" not in sources and "langchain_groq" not in sources
    assert "from app.assistant import ask_bankmate_with_sources" in (root / "views" / "assistant.py").read_text()


# ---------- Regressions and relocated behaviour ----------

def test_customer_list_renders_for_short_and_long_lists(monkeypatch, approved_dataset):
    """Regression: st.container(height=None) crashed Customer 360 for lists of 12 or fewer."""

    short = page(navigation.CUSTOMERS["path"])                       # fixture: 3 customers
    assert sorted(b.key for b in short.button if b.key.startswith("cust-btn-")) == [
        "cust-btn-C101", "cust-btn-C102", "cust-btn-C103"]
    short.text_input(key="explorer-search").set_value("pune").run()   # filtered down to one
    assert not short.exception, short.exception

    monkeypatch.setattr(dl, "_dataset", approved_dataset)            # 30 customers: scrolling list
    long = page(navigation.CUSTOMERS["path"])
    assert len([b for b in long.button if b.key.startswith("cust-btn-")]) == 30
    assert "30 customers" in text(long)


def test_transaction_type_filter_scopes_totals_and_rows():
    """The Debit/Credit filter moved from Analytics to the Transactions explorer."""

    at = page(navigation.TRANSACTIONS["path"])
    at.selectbox(key="txn-type").set_value("Credit").run()
    html = text(at)
    assert not at.exception, at.exception
    assert 'Matching transactions</div><div class="bm-kpi-value">3</div>' in html
    assert "₹95,000" in html and 'Debits</div><div class="bm-kpi-value">₹0</div>' in html
    assert html.count("bm-type-credit") == 3 and "bm-type-debit" not in html


def test_navigation_through_every_page_in_one_session(data):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    for registered in navigation.PAGES + list(reversed(navigation.PAGES)):
        at.switch_page(registered["path"]).run()
        assert not at.exception, (registered["title"], at.exception)
        assert not at.error, (registered["title"], [e.value for e in at.error])


def test_disputes_table_columns(data):
    html = text(page(navigation.DISPUTES["path"]))
    headers = ["Dispute", "Customer", "Disputed transaction", "Amount", "Reason", "Status", "Outcome", "Refund issued"]
    assert all(f">{h}</th>" in html for h in headers)


def test_customer_without_transactions_shows_clear_states(data):
    at = page(navigation.CUSTOMERS["path"])
    at.text_input(key="explorer-search").set_value("C130").run()
    at.button(key="cust-btn-C130").click().run()
    html = text(at)
    assert not at.exception, at.exception
    assert "Dev Malhotra" in html and "₹0" in html
    assert "No transactions match these filters." in html and "No disputes for this customer." in html
    assert "No debit cards on file." in html


def test_transactions_empty_state(data):
    at = page(navigation.TRANSACTIONS["path"])
    at.text_input(key="txn-merchant").set_value("Swiggy").run()
    assert "No matching transactions" in text(at)


def test_compact_customer_selector_stays_in_sync_with_the_list():
    """Narrow screens use a compact selector (CSS-switched); it must drive and follow the selection."""

    at = page(navigation.CUSTOMERS["path"])
    picker = at.selectbox(key="explorer-compact-select")
    assert picker.value == "C101" and picker.options[:3] == ["Arjun Mehta · C101", "Priya Sharma · C102",
                                                             "Rahul Nair · C103"]
    picker.set_value("C103").run()
    assert not at.exception, at.exception
    assert "Rahul Nair" in text(at) and at.query_params.get("customer") in ("C103", ["C103"])

    at.button(key="cust-btn-C102").click().run()                     # desktop list selection
    assert at.selectbox(key="explorer-compact-select").value == "C102"
    assert "Priya Sharma" in text(at)


def test_month_labels_fit_narrow_screens_but_stay_unambiguous():
    from ui import charts

    def labels(months):
        rows = [{"month": m, "figures": {"total_credits": D(1), "total_debits": D(1), "net_cash_flow": D(0)}}
                for m in months]
        return sorted(set(charts.monthly_flow(rows).to_dict()["datasets"].popitem()[1][i]["month"]
                          for i in range(2 * len(months))))

    assert labels(["2026-05", "2026-06"]) == ["Jun", "May"]                  # one year: short labels
    assert labels(["2025-12", "2026-01"]) == ["Dec 25", "Jan 26"]            # spans years: keep the year


# ---------- Final polish: formatting, wording and mobile access ----------

def test_negative_amounts_use_one_minus_sign_everywhere(data):
    from ui.data import format_inr
    assert format_inr("-49988.38", signed=True) == "\u2212₹49,988.38"     # Decimal precision kept
    for path in (navigation.ANALYTICS["path"], navigation.TRANSACTIONS["path"], navigation.CUSTOMERS["path"],
                 navigation.HOME["path"]):
        html = text(page(path))
        assert not re.search(r"-₹", html), path                             # never a hyphen before an amount


def test_rejected_dispute_is_closed_not_resolved(data):
    html = text(page(navigation.DISPUTES["path"]))
    rejected = re.search(r"D0004.*?</tr>", html, re.S).group(0)
    assert "Rejected" in rejected and "closed 10 Aug 2026" in rejected and "resolved" not in rejected
    resolved = re.search(r"D0003.*?</tr>", html, re.S).group(0)
    assert "resolved 25 Sep 2026" in resolved


def test_dispute_rows_carry_status_and_amount_for_phones(data):
    html = text(page(navigation.DISPUTES["path"]))
    row = re.search(r"<tr><td><strong>D0008.*?</tr>", html, re.S).group(0)
    glance = re.search(r'bm-dispute-glance" aria-hidden="true">(.*?)</div>', row, re.S).group(1)
    assert "Open" in glance and "₹3,150" in glance


def test_masked_card_references_do_not_wrap(data):
    html = text(page(navigation.DISPUTES["path"]))
    assert '<span class="bm-nowrap">•••• 5863</span>' in html
    assert "•••• " not in html.replace('<span class="bm-nowrap">•••• ', "")   # every reference is wrapped


def test_analytics_tables_are_panelled_and_scrollable(data):
    at = page(navigation.ANALYTICS["path"])
    html = text(at)
    assert "Monthly summary" in html                     # the table now has its own panel heading
    assert html.count("bm-scroll-hint") >= 4             # monthly, gross vs net, accounts, transaction details
    assert '<td class="bm-nowrap">May 2026</td>' in html  # month labels never wrap
