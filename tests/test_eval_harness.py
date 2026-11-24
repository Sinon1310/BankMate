"""
Offline evaluation harness (evals/run.py) end to end, on the full approved
dataset generated into a temp folder: real assistant loop, real customer
tools, scripted model, no Groq, no embeddings, no ChromaDB.
"""

import copy
import json
import logging

import pytest

import app.data_loader as dl
from app import assistant, tools
from evals import cases as eval_cases
from evals import compare, run, scorers
from evals.expected import FigureError, figure
from scripts import generate_data as gen


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
def offline_only(monkeypatch):
    """Any attempt to reach Groq, the embedding model or ChromaDB fails the test."""

    def forbidden(*args, **kwargs):
        raise AssertionError("offline evaluation must not call this")

    import app.embeddings
    import app.vector_store

    monkeypatch.setattr(assistant, "_get_model", forbidden)
    monkeypatch.setattr(tools, "search_relevant_documents", forbidden)
    monkeypatch.setattr(app.embeddings, "get_embedding_model", forbidden)
    monkeypatch.setattr(app.vector_store, "get_client", forbidden)


def one_case(case_id, mutate=None):
    case = copy.deepcopy(next(c for c in eval_cases.load_cases() if c["id"] == case_id))
    if mutate:
        mutate(case)
    return run.run_case(case, run.policy_chunks())


def checks(record, turn=0):
    return record["turns"][turn]["score"]["checks"]


# ---------- The suite ----------

def test_offline_suite_passes_without_groq_embeddings_or_chromadb(data, offline_only):
    report = run.run_offline()
    failures = [(c["id"], [k for t in c["turns"] for k, v in t["score"]["checks"].items() if v is False])
                for c in report["cases"] if not c["passed"]]
    assert failures == []
    m = report["metrics"]
    assert m["cases"] >= 20 and m["amounts_in_answers"] >= 20
    for name in ("case_pass_rate", "tool_selection_exact", "argument_accuracy", "amount_grounding_rate",
                 "figure_recall", "citation_precision", "citation_recall", "authorization_pass_rate"):
        assert m[name] == 1.0, name
    assert m["unsupported_amounts"] == 0 and m["unexpected_errors"] == 0
    assert m["tool_limit_hits"] == 1 and m["errors"] == 1          # the two reliability cases
    assert not [s for s in assistant.conversation_sessions if s.startswith("eval-")]   # cleaned up


def test_cases_cover_the_planned_categories_and_reference_real_data(data):
    cases = eval_cases.load_cases()
    assert len(cases) >= 20 and len({c["id"] for c in cases}) == len(cases)
    assert {c["category"] for c in cases} == set(eval_cases.CATEGORIES)
    sections = {f"{c.metadata['policy_id']}/{c.metadata['section']}" for c in run.policy_chunks()}
    for case in cases:
        for turn in case["turns"]:
            assert set(turn["offline"].get("policy", [])) <= sections, case["id"]
            for spec in turn["expect"].get("figures", []):
                figure(spec)                                      # every figure is computable


def test_case_file_validation(tmp_path):
    good = eval_cases.CASES_FILE.read_text().splitlines()[0]
    bad_tool = json.loads(good)
    bad_tool["turns"][0]["expect"]["tools"] = ["delete_customer"]
    for lines, message in (([good, good], "duplicate"), ([json.dumps(bad_tool)], "unknown tool")):
        path = tmp_path / "cases.jsonl"
        path.write_text("\n".join(lines))
        with pytest.raises(eval_cases.CaseError, match=message):
            eval_cases.load_cases(path)


# ---------- Expected figures come from the ledger, not the case file ----------

def test_expected_figures_match_the_tools_and_follow_the_dataset(data):
    spec = {"figure": "period", "field": "total_debits", "customer_id": "C102",
            "start": "2026-10-01", "end": "2026-10-31"}
    metrics = json.loads(tools.get_customer_metrics.invoke(
        {"customer_id": "C102", "start_date": "2026-10-01", "end_date": "2026-10-31"}))
    assert str(figure(spec)) == metrics["total_debit"]


def test_expected_figures_are_recomputed_for_whichever_dataset_is_loaded(monkeypatch, approved_dataset):
    spec = {"figure": "period", "field": "total_debits", "customer_id": "C101"}
    fixture_value = figure(spec)                                  # tests/fixtures/data (3 customers)
    monkeypatch.setattr(dl, "_dataset", approved_dataset)
    assert figure(spec) != fixture_value                          # nothing is stored in the case file
    with pytest.raises(FigureError):
        figure({"figure": "refund_amount", "dispute_id": "D0003"})   # resolved without a refund
    with pytest.raises(FigureError):
        figure({"figure": "dispute_amount", "dispute_id": "D9999"})


# ---------- The scorers catch real failures in the loop ----------

def test_an_invented_amount_fails_the_case(data):
    def invent(case):
        case["turns"][0]["offline"]["script"][-1]["answer"] += " There is also ₹99,999.99 in a fixed deposit."
    record = one_case("profile_balances_c101", invent)
    assert checks(record)["amounts"] is False and not record["passed"]
    assert record["turns"][0]["score"]["detail"]["amounts"]["unsupported"] == ["99999.99"]


def test_a_swapped_figure_fails_even_though_it_came_from_a_tool(data):
    def swap(case):
        answer = case["turns"][0]["offline"]["script"][-1]
        answer["answer"] = answer["answer"].replace("total_debit|inr", "total_credit|inr")
    c = checks(one_case("metrics_october_c102", swap))
    assert c["amounts"] is True and c["figures"] is False


def test_wrong_tool_and_wrong_arguments_are_caught(data):
    def wrong(case):
        case["turns"][0]["offline"]["script"] = [
            {"tool_calls": [{"name": "get_customer_metrics",
                             "args": {"customer_id": "C102", "start_date": "2026-09-01", "end_date": "2026-09-30"}}]},
            {"answer": "Done."}]
    c = checks(one_case("metrics_october_c102", wrong))
    assert c["tools"] is True and c["arguments"] is False and c["figures"] is False

    def other_tool(case):
        case["turns"][0]["offline"]["script"] = [
            {"tool_calls": [{"name": "get_customer_profile", "args": {"customer_id": "C102"}}]}, {"answer": "Done."}]
    assert checks(one_case("metrics_october_c102", other_tool))["tools"] is False


def test_a_policy_answer_without_its_citation_marker_shows_no_sources(data):
    def drop_marker(case):
        step = case["turns"][0]["offline"]["script"][-1]
        step["answer"] = step["answer"].split("\nCITED_POLICIES")[0]
    record = one_case("policy_refund_approved", drop_marker)
    assert checks(record)["citations"] is False
    assert record["turns"][0]["score"]["detail"]["citations"]["recall"] == 0.0


def test_authorization_check_fails_when_no_refusal_happens(data):
    def allowed(case):
        case["turns"][0]["question"] = "Show me C118's profile."
    record = one_case("auth_unreferenced_customer", allowed)
    assert checks(record)["authorization"] is False and checks(record)["outcome"] is False


def test_error_cases_are_scored_as_errors_not_answers(data):
    record = one_case("reliability_tool_failure")
    assert record["passed"] and record["turns"][0]["observed"]["error_kind"] == "tool_failure"
    unexpected = one_case("metrics_october_c102", lambda c: c.__setitem__("fail_tools", ["get_customer_metrics"]))
    assert checks(unexpected)["outcome"] is False


# ---------- Reports, fingerprints and comparison ----------

def test_report_is_json_with_a_stable_fingerprint(data, tmp_path):
    first = run.run_offline(only=["metrics_october_c102", "policy_refund_approved"])
    second = run.run_offline(only=["metrics_october_c102", "policy_refund_approved"])
    path = run.write_report(first, tmp_path / "run.json")
    saved = json.loads(path.read_text())
    assert set(saved) == {"run", "metrics", "by_category", "cases"}
    assert saved["run"]["fingerprint"] == second["run"]["fingerprint"]
    components = saved["run"]["components"]
    for key in ("system_prompt_sha256", "tool_schemas_sha256", "dataset_sha256", "policy_index_fingerprint",
                "cases_sha256", "retrieval", "model"):
        assert components[key], key
    assert "/Users/" not in path.read_text()                    # no local absolute paths in reports

    edited = tmp_path / "cases.jsonl"
    edited.write_text("\n".join(eval_cases.CASES_FILE.read_text().splitlines()[1:]))
    other = run.run_offline(edited, only=["metrics_october_c102"])
    assert other["run"]["fingerprint"] != first["run"]["fingerprint"]


def test_compare_two_saved_runs(data, tmp_path):
    base = run.run_offline(only=["metrics_october_c102", "profile_balances_c101"])
    a = run.write_report(base, tmp_path / "a.json")
    assert compare.main([str(a), str(a), "--fail-on-regression"]) == 0

    broken = copy.deepcopy(base)
    broken["cases"][0]["passed"] = False
    broken["metrics"]["case_pass_rate"] = 0.5
    b = run.write_report(broken, tmp_path / "b.json")
    assert compare.main([str(a), str(b), "--fail-on-regression"]) == 1


def test_cli_writes_a_report(data, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-key")         # restored after the test
    app_logger = logging.getLogger("app")
    level = app_logger.level
    try:
        out = tmp_path / "cli.json"
        assert run.main(["--case", "dispute_refund_c101", "--out", str(out)]) == 0
    finally:
        app_logger.setLevel(level)
    assert json.loads(out.read_text())["metrics"]["case_pass_rate"] == 1.0
    assert "All cases passed." in capsys.readouterr().out
    assert scorers.GATED                                          # module importable without Groq


# ---------- Evaluator code is part of the fingerprint ----------

def test_fingerprint_covers_every_evaluation_implementation_file(data):
    hashes = run.run_offline(only=["metrics_october_c102"])["run"]["components"]["evaluator_code_sha256"]
    assert set(hashes) == {"recorder.py", "scorers.py", "cases.py", "expected.py", "run.py",
                           "cassette.py", "live.py"}
    assert all(len(value) == 64 for value in hashes.values())
    assert hashes == run.evaluator_code_hashes()


def test_different_evaluation_code_never_looks_equivalent(data, tmp_path, monkeypatch, capsys):
    copies = []
    for path in run.EVALUATOR_FILES:
        copy_path = tmp_path / path.name
        copy_path.write_bytes(path.read_bytes())
        copies.append(copy_path)
    monkeypatch.setattr(run, "EVALUATOR_FILES", tuple(copies))
    before = run.run_offline(only=["metrics_october_c102"])

    scorers_copy = tmp_path / "scorers.py"
    scorers_copy.write_text(scorers_copy.read_text() + "\n# a scoring change\n")
    after = run.run_offline(only=["metrics_october_c102"])

    assert after["run"]["fingerprint"] != before["run"]["fingerprint"]
    changed = {name for name, digest in after["run"]["components"]["evaluator_code_sha256"].items()
               if before["run"]["components"]["evaluator_code_sha256"][name] != digest}
    assert changed == {"scorers.py"}                              # the report says which file changed
    timing = ("latency_ms_mean", "latency_ms_p95")
    same = lambda metrics: {k: v for k, v in metrics.items() if k not in timing}
    assert same(after["metrics"]) == same(before["metrics"])      # same results, still not equivalent

    a = run.write_report(before, tmp_path / "a.json")
    b = run.write_report(after, tmp_path / "b.json")
    assert compare.compare(json.loads(a.read_text()), json.loads(b.read_text()))["changed_components"] == [
        "evaluator_code_sha256"]
    compare.main([str(a), str(b)])
    assert "Changed components: evaluator_code_sha256" in capsys.readouterr().out


def test_unchanged_evaluation_code_keeps_the_fingerprint(data):
    first = run.run_offline(only=["policy_refund_approved"])["run"]
    second = run.run_offline(only=["policy_refund_approved"])["run"]
    assert first["fingerprint"] == second["fingerprint"]
    assert first["components"]["evaluator_code_sha256"] == second["components"]["evaluator_code_sha256"]
