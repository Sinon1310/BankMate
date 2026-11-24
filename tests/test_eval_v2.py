"""
Evaluator V2: required vs optional figures, label-bound figures, non-financial
facts, amount detection beyond ₹-marked text, coverage, replay re-scoring with
strict execution checks, latency accounting and cross-mode comparison.
No Groq, no embeddings, no production ChromaDB.
"""

import copy
import json
from decimal import Decimal as D
from pathlib import Path

import pytest

import app.data_loader as dl
from app import assistant, ledger, llm, tools
from evals import answers, compare, run, scorers
from evals import cases as eval_cases
from evals.cassette import CassetteMismatch
from evals.expected import FigureError, fact
from evals.fakes import FakeGroq, FakePolicySearch
from evals.scorers import MODE_DEPENDENT, check_fact, score_turn
from scripts import generate_data as gen

BASELINE = Path(__file__).resolve().parent.parent / "evals" / "baselines" / "phase2-live-6.cassette.json"


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def approved_dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("approved")
    assert gen.main(["--output-dir", str(folder)]) == 0
    return dl.load_dataset(folder)


@pytest.fixture
def data(monkeypatch, approved_dataset):
    monkeypatch.setattr(dl, "_dataset", approved_dataset)
    return approved_dataset


@pytest.fixture
def no_groq(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a test tried to reach the real Groq model")

    monkeypatch.setattr(assistant, "_get_model", forbidden)
    monkeypatch.setattr(llm, "ChatGroq", forbidden)


def observed(answer="", **overrides):
    base = {"answer": answer, "sources": [], "requested_tools": [], "executed_tools": [], "blocked": [],
            "model_calls": [], "revisions": 0, "tool_limit": False, "error_kind": None, "latency_ms": 1.0}
    base.update(overrides)
    return base


def fig(spec, value, required=True):
    return {"label": spec.get("field") or spec.get("type") or spec["figure"], "value": D(value), "spec": spec,
            "required": required}


CREDITS = {"figure": "period", "field": "total_credits", "customer_id": "C102"}
DEBITS = {"figure": "period", "field": "total_debits", "customer_id": "C102"}
NET = {"figure": "period", "field": "spending_net", "customer_id": "C106"}
GROSS = {"figure": "period", "field": "spending_gross", "customer_id": "C106"}


def score(answer, figures=(), facts=None, supported=(), question="q"):
    return score_turn({}, observed(answer), {D(s) for s in supported}, list(figures), question, facts)


# ---------- Required vs optional figures ----------

def test_missing_optional_figure_never_fails_but_missing_required_does():
    figures = [fig(NET, "283041.30"), fig(GROSS, "288931.30", required=False)]
    ok = score("Spending after refunds was ₹2,83,041.30.", figures, supported=["283041.30"])
    assert ok["checks"]["figures"] is True and ok["passed"]
    assert [f["label_status"] for f in ok["detail"]["figures"]] == ["labelled", "missing"]
    bad = score("Gross spending was ₹2,88,931.30.", figures, supported=["288931.30"])
    assert bad["checks"]["figures"] is False and not bad["passed"]


def test_only_optional_figures_leave_the_figure_check_not_applicable():
    result = score("The dispute D0003 is resolved.", [fig({"figure": "dispute_amount", "dispute_id": "D0003"},
                                                          "7999.00", required=False)])
    assert result["checks"]["figures"] is None


def test_required_and_optional_recall_are_reported_separately():
    figures = [fig(NET, "283041.30"), fig(GROSS, "288931.30", required=False)]
    turns = [{"observed": observed(), "score": score("Net spending ₹2,83,041.30.", figures, supported=["283041.30"])}]
    metrics = scorers.aggregate([{"id": "x", "category": "metrics", "passed": True, "turns": turns}])
    assert metrics["figure_recall"] == 1.0 and metrics["optional_figure_recall"] == 0.0


# ---------- Figures bound to their labels ----------

def test_swapped_credit_and_debit_figures_are_mislabelled_and_fail():
    figures = [fig(CREDITS, "306300.00"), fig(DEBITS, "65384.42")]
    right = score("Credits: ₹3,06,300.00. Debits: ₹65,384.42.", figures, supported=["306300.00", "65384.42"])
    swapped = score("Credits: ₹65,384.42. Debits: ₹3,06,300.00.", figures, supported=["306300.00", "65384.42"])
    assert right["checks"]["figures"] is True
    assert {f["label_status"] for f in right["detail"]["figures"]} == {"labelled"}
    assert swapped["checks"]["figures"] is False and swapped["checks"]["amounts"] is True   # grounded, wrong label
    assert {f["label_status"] for f in swapped["detail"]["figures"]} == {"mislabelled"}


def test_gross_shown_as_net_is_mislabelled():
    result = score("Net spending was ₹2,88,931.30.", [fig(NET, "283041.30"), fig(GROSS, "288931.30")],
                   supported=["288931.30"])
    statuses = {f["figure"]: f["label_status"] for f in result["detail"]["figures"]}
    assert statuses == {"spending_net": "missing", "spending_gross": "mislabelled"}


def test_a_figure_with_no_recognisable_label_is_accepted_as_unlabelled():
    result = score("C102: ₹3,06,300.00 in and ₹65,384.42 out.", [fig(CREDITS, "306300.00")],
                   supported=["306300.00", "65384.42"])
    assert result["detail"]["figures"][0]["label_status"] == "unlabelled" and result["checks"]["figures"]


def test_table_cells_take_their_label_from_the_row_and_header():
    table = "| Account | Balance (₹) |\n|---|---|\n| ACC1001 | 1,05,465.46 |\n| ACC1002 | 81,761.34 |"
    spec = {"figure": "account_balance", "account_id": "ACC1002"}
    result = score(table, [fig(spec, "81761.34")], supported=["105465.46", "81761.34"])
    assert result["detail"]["figures"][0]["label_status"] == "labelled"


# ---------- Amounts without currency symbols, formats, coverage ----------

def test_amounts_in_table_cells_and_european_formatting_are_detected_and_checked():
    table = ("| Transaction ID | Date | Amount (₹) | Card (last 4) |\n|---|---|---|---|\n"
             "| T01603 | 2026‑09‑16 | 4 686,00 | 8323 |\n| T00951 | 2026‑07‑19 | 1 953,78 | 8323 |")
    reading = answers.read(table)
    assert [(a.value, a.source) for a in reading.amounts] == [(D("4686.00"), "table"), (D("1953.78"), "table")]
    assert all(set(a.issues) == {"missing_currency", "comma_decimal"} for a in reading.amounts)
    assert reading.other_numbers["card"] == 2 and reading.unclassified == []      # dates, IDs, cards excluded
    result = score(table, supported=["4686.00"])
    assert result["checks"]["amounts"] is False and result["detail"]["amounts"]["unsupported"] == ["1953.78"]
    assert result["detail"]["formatting"]["issues"] == {"missing_currency": 2, "comma_decimal": 2}


def test_unmarked_money_shaped_numbers_are_checked_and_formats_reported():
    reading = answers.read("Total 1,87,226.80 and 81,761.3 then ₹95,000 and ₹4,686.00.")
    found = {a.text: a for a in reading.amounts}
    assert set(found) == {"1,87,226.80", "81,761.3", "95,000", "4,686.00"}
    assert found["1,87,226.80"].source == "shaped" and found["1,87,226.80"].grouping == "indian"
    assert found["81,761.3"].issues == ["missing_currency", "decimal_precision"]
    assert found["95,000"].issues == ["decimal_precision"] and found["4,686.00"].issues == []


def test_dates_ids_years_counts_and_percentages_are_not_amounts():
    text = ("C106 made 5 transactions (T01603, T00637) on 2026-09-16, 12‑Apr‑2019 and 03/09/2026 at 10:30; "
            "ACC1011 grew 4.5% in 2026.")
    reading = answers.read(text)
    assert reading.amounts == [] and reading.unclassified == []
    assert reading.other_numbers["count"] == 1 and reading.other_numbers["year"] == 1


def test_grounding_is_never_claimed_when_no_amount_was_checked():
    result = score("The dispute is resolved with no financial adjustment.")
    assert result["checks"]["amounts"] is None and result["detail"]["amounts"]["checked"] == 0
    turns = [{"observed": observed(), "score": result}]
    metrics = scorers.aggregate([{"id": "x", "category": "disputes", "passed": True, "turns": turns}])
    assert metrics["amounts_checked"] == 0 and metrics["amount_grounding_rate"] is None


def test_unclassified_numbers_lower_the_coverage():
    result = score("Also 123456 and balance ₹1,000.00.", supported=["1000.00"])   # no identifier word: unclassified
    assert result["detail"]["amounts"]["unclassified_numbers"] == ["123456"]
    assert result["detail"]["amounts"]["coverage"] == 0.5


# ---------- Non-financial facts ----------

def test_facts_come_from_the_dataset_and_ledger_not_from_tool_output(data, monkeypatch):
    def poisoned(*args, **kwargs):
        raise AssertionError("facts must not call the assistant's tools")

    for name in ("get_customer_disputes", "get_customer_transactions", "get_customer_metrics"):
        monkeypatch.setattr(tools, name, poisoned)
    assert fact({"fact": "dispute_status", "customer_id": "C111"})["expected"] == [
        {"dispute_id": "D0003", "status": "Resolved"}]
    assert fact({"fact": "dispute_outcome", "customer_id": "C111"})["expected"] == [
        {"dispute_id": "D0003", "outcome": "No financial adjustment", "refunded": False}]
    rows = [t for t in data.tables["transactions"] if t["customer_id"] == "C106" and "homenest" in t["merchant"].lower()]
    ids = fact({"fact": "transaction_ids", "customer_id": "C106", "filters": {"merchant": "HomeNest"}})["expected"]
    assert ids == sorted(t["transaction_id"] for t in rows) and len(ids) == 5
    assert fact({"fact": "transaction_count", "customer_id": "C106", "filters": {"merchant": "HomeNest"}})[
        "expected"] == 5
    period = ledger.clamp_period(None, None, data.meta["window_start"], data.meta["as_of_date"])
    assert fact({"fact": "period_range", "customer_id": "C106"})["expected"] == [
        period.start.isoformat(), period.end.isoformat()]
    with pytest.raises(FigureError):
        fact({"fact": "dispute_status", "customer_id": "C999"})


STATUS = {"kind": "dispute_status", "label": "s", "expected": [{"dispute_id": "D0003", "status": "Resolved"}]}
OUTCOME = {"kind": "dispute_outcome", "label": "o",
           "expected": [{"dispute_id": "D0003", "outcome": "No financial adjustment", "refunded": False}]}


def test_dispute_status_and_outcome_facts():
    good = "Dispute D0003 is **Resolved**; the outcome was no financial adjustment."
    assert check_fact(STATUS, good)["passed"] and check_fact(OUTCOME, good)["passed"]
    assert check_fact(STATUS, "The dispute is still under review.")["problems"] == ["status 'Resolved' not stated"]
    wrong = check_fact(STATUS, "Dispute D0003 was rejected.")
    assert "conflicting status 'Rejected' stated" in wrong["problems"] and not wrong["passed"]
    assert not check_fact(OUTCOME, "Resolved: a full refund was issued.")["passed"]
    assert check_fact(OUTCOME, "It is resolved.")["problems"] == ["outcome 'No financial adjustment' not stated"]


def test_transaction_id_and_count_facts():
    ids = {"kind": "transaction_ids", "label": "i", "expected": ["T00637", "T01603"]}
    count = {"kind": "transaction_count", "label": "c", "expected": 2}
    assert check_fact(ids, "T01603 and T00637.")["passed"]
    assert check_fact(ids, "Only T01603.")["problems"] == ["missing transaction IDs ['T00637']"]
    assert check_fact(ids, "T01603, T00637 and T09999.")["problems"] == [
        "transaction IDs not in the selection ['T09999']"]
    assert check_fact(count, "C106 has 2 HomeNest transactions.")["passed"]
    assert check_fact(count, "C106 has two HomeNest transactions.")["passed"]
    assert check_fact(count, "| T01603 |\n| T00637 |")["passed"]                       # listed rows count
    assert not check_fact(count, "C106 has 3 HomeNest transactions: T01603, T00637.")["passed"]
    assert not check_fact(count, "Here they are: T01603.")["passed"]
    assert check_fact(count, "(2026‑05‑01 to 2026‑11‑21)\n\n| Transaction ID |\n| T01603 |\n| T00637 |")["passed"]


def test_period_range_fact_accepts_common_date_formats():
    period = {"kind": "period_range", "label": "p", "expected": ["2026-05-01", "2026-11-21"]}
    for text in ("from 2026‑05‑01 to 2026‑11‑21", "between 1 May 2026 and 21 November 2026",
                 "May 1, 2026 – Nov 21, 2026", "01/05/2026 to 21/11/2026"):
        assert check_fact(period, text)["passed"], text
    assert check_fact(period, "Spending after refunds was ₹2,83,041.30.")["problems"] == [
        "period start 2026-05-01 not stated", "period end 2026-11-21 not stated"]


# ---------- Case schema ----------

def _write_case(tmp_path, mutate):
    case = copy.deepcopy(next(c for c in eval_cases.load_cases() if c["id"] == "dispute_status_c111"))
    mutate(case["turns"][0]["expect"])
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(case) + "\n")
    return path


@pytest.mark.parametrize("mutate, message", [
    (lambda e: e["figures"][0].update(required="no"), "'required' must be true or false"),
    (lambda e: e["figures"][0].update(colour="red"), "unknown keys"),
    (lambda e: e["figures"].append({"figure": "period", "customer_id": "C111"}), "missing \\['field'\\]"),
    (lambda e: e["figures"].append({"figure": "period", "field": "spend", "customer_id": "C111"}),
     "unknown period field"),
    (lambda e: e["figures"][0].update(label="(unclosed"), "not a valid regex"),
    (lambda e: e["facts"].append({"fact": "dispute_mood", "customer_id": "C111"}), "unknown fact kind"),
    (lambda e: e["facts"].append({"fact": "dispute_status"}), "exactly one of customer_id / dispute_id"),
    (lambda e: e["facts"].append({"fact": "transaction_ids", "customer_id": "C111", "filters": {"card": "1"}}),
     "filters must use"),
    (lambda e: e["facts"].append({"fact": "period_range", "start": "May 2026"}), "must be an ISO date"),
])
def test_figure_and_fact_specs_are_validated_strictly(tmp_path, mutate, message):
    with pytest.raises(eval_cases.CaseError, match=message):
        eval_cases.load_cases(_write_case(tmp_path, mutate))


def test_the_corrected_cases_ask_only_what_their_questions_ask():
    cases = {c["id"]: c["turns"][0] for c in eval_cases.load_cases()}
    req = lambda t: [f.get("field") or f.get("type") or f["figure"] for f in t["expect"].get("figures", [])  # noqa
                     if f.get("required", True)]
    assert req(cases["net_spending_c106"]) == ["spending_net"]
    assert req(cases["transactions_merchant_c106"]) == [] and req(cases["dispute_status_c111"]) == []
    assert req(cases["transactions_merchant_totals_c106"]) == ["Debit", "Credit"]
    assert "total" in cases["transactions_merchant_totals_c106"]["question"].lower()
    assert {f["fact"] for f in cases["dispute_status_c111"]["expect"]["facts"]} == {"dispute_status",
                                                                                    "dispute_outcome"}


# ---------- Replay: expectation changes re-score, execution changes are refused ----------

IDS = ["metrics_october_c102", "net_spending_c106", "policy_refund_approved"]


@pytest.fixture
def recording(data, no_groq, tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    (index / "x").write_text("x")
    cases = eval_cases.load_cases(only=IDS)
    clock = type("C", (), {"now": 0.0, "__call__": lambda self: self.now,
                           "sleep": lambda self, s: setattr(self, "now", self.now + s)})()
    report, cassette = run.run_live(only=IDS, confirm=True, model=FakeGroq(cases),
                                    policy_tool=FakePolicySearch(run.policy_chunks(), cases), index_source=index,
                                    check_index=lambda p: None, clock=clock, sleep=clock.sleep)
    report_path, cassette_path = run.save_live(report, cassette, "rec", out_dir=tmp_path)
    return report, report_path, cassette_path, tmp_path


def _cases_file(tmp_path, mutate):
    lines = []
    for case in eval_cases.load_cases():
        mutate(case)
        lines.append(json.dumps(case, ensure_ascii=False))
    path = tmp_path / "edited.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_expectation_only_change_is_rescored_from_the_recording(recording):
    report, report_path, cassette_path, tmp_path = recording

    def require_gross(case):
        if case["id"] == "net_spending_c106":
            case["turns"][0]["expect"]["figures"][1]["required"] = True
            case["turns"][0]["expect"]["facts"].append({"fact": "transaction_count", "customer_id": "C106"})

    replayed = run.run_replay(cassette_path, _cases_file(tmp_path, require_gross))
    info = replayed["run"]["replay_of"]
    assert info["rescored_cases"] == ["net_spending_c106"] and info["execution_source"] == "cassette"
    case = next(c for c in replayed["cases"] if c["id"] == "net_spending_c106")
    assert case["turns"][0]["score"]["checks"]["facts"] is False        # new expectation applied to old answer
    diff = compare.compare(json.loads(report_path.read_text()), replayed)
    assert diff["same_execution"] is True and diff["same_evaluation"] is False
    assert diff["case_changes"] == [{"id": "net_spending_c106", "baseline": True, "candidate": False,
                                     "change": "regressed"}]


def test_changed_question_is_refused(recording):
    _, _, cassette_path, tmp_path = recording

    def reword(case):
        if case["id"] == "metrics_october_c102":
            case["turns"][0]["question"] = "What were C102's credits in October 2026?"

    with pytest.raises(CassetteMismatch, match="questions or injected tool failures changed"):
        run.run_replay(cassette_path, _cases_file(tmp_path, reword))


def test_changed_injected_failure_is_refused(recording):
    _, _, cassette_path, tmp_path = recording

    def inject(case):
        if case["id"] == "metrics_october_c102":
            case["fail_tools"] = ["get_customer_metrics"]

    with pytest.raises(CassetteMismatch, match="questions or injected tool failures changed"):
        run.run_replay(cassette_path, _cases_file(tmp_path, inject))


@pytest.mark.parametrize("component", ["system_prompt_sha256", "tool_schemas_sha256", "dataset_sha256",
                                       "max_tool_steps"])
def test_changed_execution_component_is_refused(recording, component):
    _, _, cassette_path, tmp_path = recording
    cassette = json.loads(cassette_path.read_text())
    cassette["run"]["execution"][component] = "changed"
    path = tmp_path / "edited.cassette.json"
    path.write_text(json.dumps(cassette))
    with pytest.raises(CassetteMismatch, match=f"{component} changed"):
        run.run_replay(path)
    assert run.run_replay(path, strict_inputs=False)["run"]["replay_of"]["notes"][0].startswith("ALLOWED DRIFT")


def test_changed_tool_output_is_refused(recording):
    _, _, cassette_path, tmp_path = recording
    cassette = json.loads(cassette_path.read_text())
    entry = cassette["cases"]["policy_refund_approved"]["turns"][0]["policy"][0]
    entry["output"] = entry["output"].replace("refund", "rebate")
    path = tmp_path / "edited.cassette.json"
    path.write_text(json.dumps(cassette))
    with pytest.raises(CassetteMismatch, match="model input differs"):
        run.run_replay(path)


def test_legacy_cassettes_are_verified_against_their_baseline_report(recording):
    _, report_path, cassette_path, tmp_path = recording
    legacy = json.loads(cassette_path.read_text())
    for key in ("execution", "case_input_sha256", "case_expect_sha256"):
        legacy["run"].pop(key)
    path = tmp_path / "old.cassette.json"
    path.write_text(json.dumps(legacy))
    with pytest.raises(CassetteMismatch, match="no baseline report"):
        run.run_replay(path)
    (tmp_path / "old.json").write_text(report_path.read_text())
    info = run.run_replay(path)["run"]["replay_of"]
    assert info["execution_source"] == "baseline report old.json" and info["rescored_cases"] == []
    other = json.loads(report_path.read_text())
    other["run"]["fingerprint"] = "different"
    (tmp_path / "old.json").write_text(json.dumps(other))
    with pytest.raises(CassetteMismatch, match="fingerprints differ"):
        run.run_replay(path)


# ---------- Latency, tokens and cross-mode comparison ----------

def test_latency_excludes_pacing_and_provider_latency_is_separate(recording):
    report, report_path, cassette_path, _ = recording
    turns = [t for c in report["cases"] for t in c["turns"]]
    assert sum(t["observed"]["paced_ms"] for t in turns) > 60000              # simulated waits happened...
    assert all(t["observed"]["latency_ms"] < 30000 for t in turns)             # ...and are not latency
    m = report["metrics"]
    assert m["pacing_ms_total"] > 60000 and m["provider_latency_ms_mean"] is not None
    assert m["tokens_consumed"] == m["tokens_total"] > 0

    replayed = run.run_replay(cassette_path)
    r = replayed["metrics"]
    assert r["tokens_total"] == m["tokens_total"] and r["tokens_consumed"] == 0     # recorded, not consumed
    assert r["provider_latency_ms_mean"] is None and r["pacing_ms_total"] == 0
    assert "recorded in the cassette" in replayed["run"]["token_usage"]


def test_cross_mode_comparison_marks_latency_not_comparable(recording):
    report, report_path, cassette_path, _ = recording
    replayed = run.run_replay(cassette_path)
    diff = compare.compare(json.loads(report_path.read_text()), replayed)
    status = {m["metric"]: m["status"] for m in diff["metrics"]}
    assert {status[name] for name in MODE_DEPENDENT} == {"not_comparable"}
    assert diff["regressions"] == [] and diff["same_execution"] is True and diff["same_evaluation"] is True
    same_mode = compare.compare(replayed, copy.deepcopy(replayed))
    assert not any(m["status"] == "not_comparable" for m in same_mode["metrics"])


# ---------- The real phase2-live-6 cassette, re-scored offline ----------

@pytest.mark.skipif(not BASELINE.exists(), reason="phase2-live-6 baseline not present")
def test_existing_live_cassette_is_rescored_without_groq(data, no_groq, monkeypatch):
    def no_index(*args, **kwargs):
        raise AssertionError("replay must not query ChromaDB")

    monkeypatch.setattr(tools, "search_relevant_documents", no_index)
    before = BASELINE.read_bytes()
    report = run.run_replay(BASELINE)
    info = report["run"]["replay_of"]
    assert info["execution_source"] == "baseline report phase2-live-6.json"
    assert set(info["rescored_cases"]) == {"net_spending_c106", "transactions_merchant_c106", "dispute_status_c111"}
    cases = {c["id"]: c for c in report["cases"]}
    assert cases["dispute_status_c111"]["passed"] and cases["transactions_merchant_c106"]["passed"]
    net = cases["net_spending_c106"]["turns"][0]
    # The recorded model answer omitted the period (rule 50). The model inputs of this single-turn case are
    # unchanged (every input digest is verified), and the assistant now appends the period the metrics tool
    # reported (app/answer_scope.py), so the same recorded response passes the period fact.
    assert net["score"]["checks"]["figures"] is True and net["score"]["checks"]["facts"] is True
    assert net["observed"]["answer"].endswith("Period covered: 2026-05-01 to 2026-11-21 (inclusive).")
    merchant = cases["transactions_merchant_c106"]["turns"][0]["score"]["detail"]
    assert merchant["amounts"]["checked"] == 5 and merchant["formatting"]["issues"]["comma_decimal"] == 5
    assert report["metrics"]["tokens_consumed"] == 0 and report["metrics"]["tokens_total"] == 34330
    assert BASELINE.read_bytes() == before                                           # baseline preserved


# ---------- Review fix pass: regression tests for findings 1-4 ----------

DEBITS_V, CREDITS_V, GROSS_V, NET_V = "65384.42", "306300.00", "288931.30", "283041.30"
ALL_SUPPORTED = [DEBITS_V, CREDITS_V, GROSS_V, NET_V]


def statuses(answer, *items):
    result = score(answer, [fig(spec, value, required) for spec, value, required in items], supported=ALL_SUPPORTED)
    return {f["figure"]: f["label_status"] for f in result["detail"]["figures"]}, result["checks"]["figures"]


@pytest.mark.parametrize("answer", [
    "Credits of ₹65,384.42 and debits of ₹3,06,300.00.",           # labels lead the amounts
    "₹65,384.42 in credits and ₹3,06,300.00 in debits.",           # labels trail the amounts
    "Credits: ₹65,384.42, debits: ₹3,06,300.00",
    "| Type | Amount |\n|---|---|\n| Credits | ₹65,384.42 |\n| Debits | ₹3,06,300.00 |",
])
def test_swapped_debits_and_credits_fail_whether_one_or_both_are_required(answer):
    both, ok_both = statuses(answer, (CREDITS, CREDITS_V, True), (DEBITS, DEBITS_V, True))
    assert ok_both is False and set(both.values()) == {"mislabelled"}
    only_debits, ok_one = statuses(answer, (DEBITS, DEBITS_V, True))
    assert ok_one is False and only_debits == {"total_debits": "mislabelled"}
    only_credits, ok_other = statuses(answer, (CREDITS, CREDITS_V, True))
    assert ok_other is False and only_credits == {"total_credits": "mislabelled"}


@pytest.mark.parametrize("answer", [
    "Gross spending ₹2,83,041.30 and net spending ₹2,88,931.30.",
    "Spending was ₹2,88,931.30 after refunds and ₹2,83,041.30 before refunds.",
])
def test_swapped_gross_and_net_fail_whether_one_or_both_are_required(answer):
    both, ok_both = statuses(answer, (NET, NET_V, True), (GROSS, GROSS_V, True))
    assert ok_both is False and set(both.values()) == {"mislabelled"}
    only_net, ok_one = statuses(answer, (NET, NET_V, True))
    assert ok_one is False and only_net == {"spending_net": "mislabelled"}


@pytest.mark.parametrize("answer", [
    "Credits of ₹3,06,300.00 and debits of ₹65,384.42.",
    "C102 received ₹3,06,300.00 in credits and had ₹65,384.42 in debits.",
    "Debits: ₹65,384.42; credits: ₹3,06,300.00.",
    "| Type | Amount |\n|---|---|\n| Credits | ₹3,06,300.00 |\n| Debits | ₹65,384.42 |",
])
def test_correctly_labelled_figures_still_pass(answer):
    found, ok = statuses(answer, (CREDITS, CREDITS_V, True), (DEBITS, DEBITS_V, True))
    assert ok is True and set(found.values()) == {"labelled"}


def test_a_value_shown_under_both_labels_counts_as_labelled_but_never_from_the_wrong_label_alone():
    found, ok = statuses("Debits: ₹65,384.42. (Credits: ₹65,384.42 was a typo.)", (DEBITS, DEBITS_V, True))
    assert found == {"total_debits": "labelled"} and ok
    found, ok = statuses("Credits: ₹65,384.42.", (DEBITS, DEBITS_V, True))
    assert found == {"total_debits": "mislabelled"} and ok is False


NO_REFUND = {"kind": "dispute_outcome", "label": "o",
             "expected": [{"dispute_id": "D0003", "outcome": "No financial adjustment", "refunded": False}]}
FULL_REFUND = {"kind": "dispute_outcome", "label": "o",
               "expected": [{"dispute_id": "D0001", "outcome": "Full refund", "refunded": True}]}


@pytest.mark.parametrize("answer", [
    "Resolved. No refund was issued; no financial adjustment.",
    "Resolved — no financial adjustment. No refund was received.",
    "It was resolved without a refund (no financial adjustment).",
    "Resolved with no refund or financial adjustment.",
    "Resolved, no financial adjustment: the refund was not issued.",
    "No financial adjustment was made, and the customer did not receive a refund!",
    "Outcome: no financial adjustment — the charge hasn't been refunded.",
    "No financial adjustment; no full refund, no partial refund.",
])
def test_negated_refund_statements_are_not_refund_claims(answer):
    assert check_fact(NO_REFUND, answer) == {"fact": "o", "kind": "dispute_outcome",
                                             "expected": NO_REFUND["expected"], "passed": True, "problems": []}


@pytest.mark.parametrize("answer", [
    "No financial adjustment. A refund was issued on 25 September.",
    "No financial adjustment; however, the customer was refunded.",
    "Resolved with no fee, but a refund was credited.",
    "No financial adjustment… then a full refund followed.",
])
def test_genuine_refund_claims_are_still_caught(answer):
    assert "claims a refund the dispute does not have" in check_fact(NO_REFUND, answer)["problems"]


def test_refunded_disputes_still_need_their_refund_outcome():
    assert check_fact(FULL_REFUND, "Resolved: a full refund was issued.")["passed"]
    assert not check_fact(FULL_REFUND, "Resolved. No refund was issued.")["passed"]


def test_unmarked_amounts_near_financial_terms_are_checked():
    refund = score("Refund of 500 posted.")
    assert refund["checks"]["amounts"] is False and refund["detail"]["amounts"]["unsupported"] == ["500"]
    debits = score("Debits were 5890 in total.")
    assert debits["detail"]["amounts"]["by_source"]["context"] == 1 and debits["checks"]["amounts"] is False
    assert score("Debits were 5890 in total.", supported=["5890"])["checks"]["amounts"] is True
    for text, value in (("The total spent was 1,05,465.46 overall", "105465.46"), ("Payment of 4686.5 made", "4686.5"),
                        ("Balance stands at 81,761 today", "81761")):
        detail = score(text, supported=[value])["detail"]["amounts"]
        assert detail["checked"] == 1 and detail["unsupported"] == [], text


def test_unrelated_numbers_are_not_amounts_and_do_not_inflate_coverage():
    text = ("C106 has 5 transactions (T01603, T00637) since 2026-05-01 on ACC1011; card ending 8323; "
            "retrieval threshold 1.2 with k=3; a total of 4 disputes; 4 debits and 1 credit.")
    detail = score(text)["detail"]["amounts"]
    assert detail["checked"] == 0 and detail["candidates"] == 0 and detail["unclassified_numbers"] == []
    assert detail["non_financial"]["card"] == 1 and detail["non_financial"]["measure"] == 2
    assert detail["non_financial"]["count"] >= 3


def test_unclassifiable_numbers_lower_coverage_instead_of_disappearing():
    result = score("Spending was ₹1,000.00 and another 7000 appeared.", supported=["1000.00"])
    detail = result["detail"]["amounts"]
    assert detail["candidates"] == 2 and detail["unclassified_numbers"] == ["7000"] and detail["coverage"] == 0.5
    turns = [{"observed": observed(), "score": result}]
    metrics = scorers.aggregate([{"id": "x", "category": "metrics", "passed": True, "turns": turns}])
    assert (metrics["amount_candidates"], metrics["unclassified_numbers"], metrics["amount_check_coverage"]) == (2, 1, 0.5)


COUNT_36 = {"kind": "transaction_count", "label": "c", "expected": 36}


def test_pagination_counts_are_not_mistaken_for_totals():
    assert check_fact(COUNT_36, "C105 has 36 transactions; showing 20 of 36 transactions.")["passed"]
    assert check_fact(COUNT_36, "There are 36 matching transactions; here are the latest 20 transactions.")["passed"]
    assert check_fact(COUNT_36, "Showing the first 20 transactions of 36 transactions in total.")["passed"]


def test_wrong_totals_are_still_caught_with_or_without_pagination():
    assert "conflicting count 40 stated" in check_fact(COUNT_36, "Showing 20 of 40 transactions.")["problems"]
    assert "conflicting count 20 stated" in check_fact(COUNT_36, "C105 has 20 transactions.")["problems"]
    shown_only = check_fact(COUNT_36, "Here are the latest 20 transactions.")
    assert not shown_only["passed"] and shown_only["problems"] == [
        "count 36 neither stated nor shown as 36 listed transactions"]


# ---------- Final fix pass: lists, refund affirmations, respectively/parentheticals, counts and IDs ----------

@pytest.mark.parametrize("text, supported, money, unsupported", [
    ("Refunds of ₹1,953.78 and 500 were posted.", ["1953.78"], ["1953.78", "500"], ["500"]),
    ("Debits: ₹4,686.00, 999 and ₹8,081.00.", ["4686.00", "8081.00"], ["4686.00", "999", "8081.00"], ["999"]),
    ("Debits were 1953.78, 4686 and 8081.", ["1953.78", "4686", "8081"], ["1953.78", "4686", "8081"], []),
    ("Refunds: ₹500.00 & 250 plus 125.", ["500.00", "250", "125"], ["500.00", "250", "125"], []),
])
def test_unmarked_items_of_a_money_list_are_checked(text, supported, money, unsupported):
    result = score(text, supported=supported)
    assert [str(a.value) for a in answers.read(text).amounts] == money
    assert result["detail"]["amounts"]["unsupported"] == unsupported
    assert result["checks"]["amounts"] is (not unsupported)


def test_counts_and_unrelated_numbers_next_to_money_lists_stay_non_financial():
    detail = score("Debits of ₹4,686.00 across 4 transactions and 2 cards; card ending 8323.",
                   supported=["4686.00"])["detail"]["amounts"]
    assert detail["checked"] == 1 and detail["non_financial"]["count"] == 2 and detail["non_financial"]["card"] == 1
    assert detail["unclassified_numbers"] == []


def test_a_missed_money_candidate_lowers_coverage_instead_of_vanishing():
    detail = score("Refunds of ₹1,953.78, then 500.", supported=["1953.78"])["detail"]["amounts"]
    assert detail["unclassified_numbers"] == ["500"] and detail["coverage"] == 0.5   # not silently a "count"
    plain = score("C106 made 5 visits to the branch.")["detail"]["amounts"]
    assert plain["candidates"] == 0 and plain["non_financial"]["count"] == 1        # no money in the sentence


@pytest.mark.parametrize("answer", [
    "No financial adjustment. Not only was a refund issued, it arrived early.",
    "No financial adjustment, and never before had a refund been processed so fast.",
    "No financial adjustment; it wasn't long before a refund was credited.",
    "No financial adjustment — not a partial refund but a full refund was issued.",
    "No financial adjustment. No refund was due; however, a refund was issued anyway.",
])
def test_affirmative_refund_claims_inside_mixed_clauses_are_still_caught(answer):
    assert "claims a refund the dispute does not have" in check_fact(NO_REFUND, answer)["problems"]


@pytest.mark.parametrize("answer", [
    "No financial adjustment. No refund was issued.",
    "No financial adjustment: the refund was not issued.",
    "No financial adjustment; the customer has not yet received a refund.",
    "No financial adjustment, and the charge was never refunded.",
])
def test_genuine_negations_are_still_not_claims(answer):
    assert check_fact(NO_REFUND, answer)["passed"]


@pytest.mark.parametrize("answer, ok", [
    ("Debits and credits were ₹65,384.42 and ₹3,06,300.00 respectively.", True),
    ("Credits and debits were ₹3,06,300.00 and ₹65,384.42, respectively.", True),
    ("Debits and credits were ₹3,06,300.00 and ₹65,384.42 respectively.", False),      # swapped
    ("Total debits (credits excluded) were ₹65,384.42; credits were ₹3,06,300.00.", True),
    ("Total credits (debits excluded) were ₹65,384.42; debits were ₹3,06,300.00.", False),  # swapped
])
def test_respectively_and_parenthetical_labels(answer, ok):
    found, check = statuses(answer, (CREDITS, CREDITS_V, True), (DEBITS, DEBITS_V, True))
    assert check is ok and set(found.values()) == ({"labelled"} if ok else {"mislabelled"})


def test_respectively_with_mismatched_label_count_falls_back_to_ordinary_labels():
    found, check = statuses("Debits were ₹65,384.42 and ₹3,06,300.00 respectively.",
                            (CREDITS, CREDITS_V, True), (DEBITS, DEBITS_V, True))
    assert found["total_debits"] == "labelled" and found["total_credits"] != "labelled"


def test_parenthetical_label_is_used_when_nothing_outside_names_the_amount():
    found, _ = statuses("The October figure (debits) was ₹65,384.42.", (DEBITS, DEBITS_V, True))
    assert found == {"total_debits": "labelled"}


@pytest.mark.parametrize("text, money, kinds", [
    ("Debit count 12, credit count 3.", [], {"count": 2}),
    ("Refund ID 7788 issued.", [], {"identifier": 1}),
    ("Number of debits: 4.", [], {"count": 1}),
    ("Refund reference #4412 noted.", [], {"identifier": 1}),
    ("Debit count 12 and debits of 4686.", ["4686"], {"count": 1}),
    ("Refund of 500 posted (ref 7788).", ["500"], {"identifier": 1}),
])
def test_counts_and_identifiers_are_not_money_but_nearby_amounts_still_are(text, money, kinds):
    reading = answers.read(text)
    assert [str(a.value) for a in reading.amounts] == money
    assert {k: v for k, v in reading.other_numbers.items() if v} == kinds
    if money:
        assert score(text)["checks"]["amounts"] is False                   # invented amounts still fail


def test_day_and_month_without_a_year_is_not_a_number():
    detail = score("From 1 June to 31 August 2026, net flow was ₹79,351.90.", supported=["79351.90"])["detail"]["amounts"]
    assert detail["candidates"] == 1 and detail["unclassified_numbers"] == [] and detail["coverage"] == 1.0
    assert score("Refund of ₹500.00 on June 3 and 7 June.", supported=["500.00"])["detail"]["amounts"][
        "unclassified_numbers"] == []
