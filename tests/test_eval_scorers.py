"""Deterministic evaluation scorers (evals/scorers.py) and run comparison (evals/compare.py)."""

import copy
import json
from decimal import Decimal as D

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals import compare, scorers
from evals.recorder import format_inr, render


def observed(**overrides):
    base = {"answer": "", "sources": [], "requested_tools": [], "executed_tools": [], "blocked": [],
            "model_calls": [], "revisions": 0, "tool_limit": False, "error_kind": None, "latency_ms": 1.0}
    base.update(overrides)
    return base


# ---------- Amount extraction and support ----------

def test_extract_amounts_handles_indian_grouping_signs_and_prefixes():
    text = "Balance ₹1,87,226.80; net −₹40,764.56, refund INR 2,750 and Rs. 500 (not 2026 or C101)."
    assert scorers.extract_amounts(text) == [D("187226.80"), D("40764.56"), D("2750"), D("500")]
    assert scorers.extract_amounts("No money here: 12 transactions in 2026.") == []


def test_numbers_in_walks_json_and_plain_text():
    payload = json.dumps({"total": "187226.80", "rows": [["T1", "2026-09-16", "4686.00"]], "count": 3,
                          "flag": True, "net": "-14720.78"})
    found = scorers.numbers_in(payload)
    assert {D("187226.80"), D("4686.00"), D("3"), D("14720.78")} <= found
    assert D("1") not in found                                   # booleans are not numbers
    assert D("2750.00") in scorers.numbers_in("Refund of 2750.00 posted.")


def test_inr_formatting_round_trips_through_extraction():
    for value in ("187226.80", "-40764.56", "2750.00", "95000", "0.50"):
        assert scorers.extract_amounts(format_inr(value)) == [abs(D(value))]
    assert format_inr("-14720.78") == "−₹14,720.78" and format_inr("12345678") == "₹1,23,45,678"


def test_amount_from_the_question_is_allowed_but_invented_ones_are_not():
    score = scorers.score_turn({}, observed(answer="Transactions of ₹10,000 or more total ₹52,000."),
                               {D("52000")}, [], "Show transactions of ₹10,000 or more.")
    assert score["checks"]["amounts"] is True
    score = scorers.score_turn({}, observed(answer="Total ₹52,001."), {D("52000")}, [], "Show totals.")
    assert score["checks"]["amounts"] is False and score["detail"]["amounts"]["unsupported"] == ["52001"]


def test_expected_figures_must_appear_in_the_answer():
    figures = [("period total_debits C102", D("1234.50"))]
    ok = scorers.score_turn({}, observed(answer="Debits ₹1,234.50."), {D("1234.50")}, figures, "q")
    swapped = scorers.score_turn({}, observed(answer="Debits ₹999."), {D("999")}, figures, "q")
    assert ok["checks"]["figures"] is True and swapped["checks"]["figures"] is False
    assert swapped["checks"]["amounts"] is True        # grounded in a tool, but the wrong figure


# ---------- Tools, arguments, citations, authorization ----------

def test_tool_selection_and_required_arguments():
    calls = [{"name": "get_customer_metrics", "args": {"customer_id": "c102", "start_date": "2026-10-01"}}]
    expect = {"tools": ["get_customer_metrics"],
              "args": [{"tool": "get_customer_metrics", "match": {"customer_id": "C102", "start_date": "2026-10-01"}}]}
    score = scorers.score_turn(expect, observed(requested_tools=calls), set(), [], "q")
    assert score["checks"]["tools"] is True and score["checks"]["arguments"] is True   # case-insensitive ID

    wrong = [{"name": "get_customer_transactions", "args": {"customer_id": "C102"}}]
    score = scorers.score_turn(expect, observed(requested_tools=wrong), set(), [], "q")
    assert score["checks"]["tools"] is False and score["checks"]["arguments"] is False
    assert score["detail"]["tools"]["precision"] == 0.0 and score["detail"]["tools"]["recall"] == 0.0


def test_numeric_arguments_compare_as_numbers():
    expect = {"args": [{"tool": "get_customer_transactions", "match": {"min_amount": "10000"}}]}
    for given in (10000, "10000.00", "10,000"):
        calls = [{"name": "get_customer_transactions", "args": {"min_amount": given}}]
        assert scorers.score_turn(expect, observed(requested_tools=calls), set(), [], "q")["checks"]["arguments"]


def test_citations_must_match_exactly_and_default_to_none():
    cited = observed(sources=[{"policy_id": "POL-002"}])
    assert scorers.score_turn({"citations": ["POL-002"]}, cited, set(), [], "q")["checks"]["citations"] is True
    assert scorers.score_turn({"citations": ["POL-001"]}, cited, set(), [], "q")["checks"]["citations"] is False
    assert scorers.score_turn({}, cited, set(), [], "q")["checks"]["citations"] is False   # unexpected source
    assert scorers.score_turn({}, observed(), set(), [], "q")["checks"]["citations"] is True


def test_blocked_results_and_authorization_check():
    call = {"name": "get_customer_profile", "args": {"customer_id": "C118"}, "id": "c1"}
    denied = json.dumps({"status": "ACCESS_DENIED", "message": "x"})
    blocked = scorers.blocked_results([HumanMessage(content="q"), AIMessage(content="", tool_calls=[call]),
                                       ToolMessage(content=denied, tool_call_id="c1")])
    assert list(blocked.values()) == [("get_customer_profile", {"customer_id": "C118"}, "ACCESS_DENIED")]

    expect = {"outcome": "refused", "blocked": ["ACCESS_DENIED"]}
    good = scorers.score_turn(expect, observed(blocked=list(blocked.values()), requested_tools=[call]), set(), [], "q")
    assert good["checks"]["authorization"] is True and good["outcome"] == "refused"
    leaked = observed(blocked=list(blocked.values()), requested_tools=[call],
                      executed_tools=[{"name": "get_customer_profile", "args": {"customer_id": "C118"}, "status": "ok"}])
    assert scorers.score_turn(expect, leaked, set(), [], "q")["checks"]["authorization"] is False
    assert scorers.score_turn(expect, observed(), set(), [], "q")["checks"]["authorization"] is False


def test_outcome_detection():
    assert scorers.detect_outcome(observed(error_kind="tool_failure")) == "error:tool_failure"
    assert scorers.detect_outcome(observed(tool_limit=True)) == "tool_limit"
    assert scorers.detect_outcome(observed(answer="Please give me the customer ID.")) == "ask_customer_id"
    no_policy = observed(executed_tools=[{"name": "search_banking_policies", "status": "ok",
                                          "output": '{"status": "NO_RELEVANT_POLICY_FOUND"}'}])
    assert scorers.detect_outcome(no_policy) == "no_policy"
    assert scorers.detect_outcome(observed(answer="Done.", requested_tools=[{"name": "x", "args": {}}])) == "answer"


def test_render_fills_templates_from_this_turns_tool_results():
    call = {"name": "get_customer_metrics", "args": {}, "id": "a"}
    messages = [HumanMessage(content="old"), AIMessage(content="", tool_calls=[{**call, "id": "old"}]),
                ToolMessage(content='{"total_debit": "1.00"}', tool_call_id="old"),
                HumanMessage(content="new"), AIMessage(content="", tool_calls=[call]),
                ToolMessage(content='{"total_debit": "402234.28", "rows": [1, 2]}', tool_call_id="a")]
    assert render("{{get_customer_metrics:total_debit|inr}} / {{get_customer_metrics:rows|count}}",
                  messages) == "₹4,02,234.28 / 2"


# ---------- Aggregation and comparison ----------

def _report(passed=True, grounding=1.0):
    turn = {"observed": observed(), "score": scorers.score_turn({}, observed(answer="ok"), set(), [], "q")}
    report = {"run": {"mode": "offline", "components": {"model": "scripted-offline"},
                      "cases": {"sha256": "abc"}},
              "cases": [{"id": "a", "category": "metrics", "passed": passed, "turns": [turn]}]}
    report["metrics"] = scorers.aggregate(report["cases"])
    report["metrics"]["amount_grounding_rate"] = grounding
    return report


def test_aggregate_produces_every_metric():
    metrics = _report()["metrics"]
    for name in scorers.GATED:
        assert name in metrics
    assert metrics["case_pass_rate"] == 1.0 and metrics["tokens_total"] is None


def test_compare_flags_gated_regressions_and_case_changes(tmp_path, capsys):
    base, same = _report(), _report()
    assert compare.compare(base, same)["regressions"] == []

    worse = _report(passed=False, grounding=0.9)
    worse["run"]["components"]["model"] = "other-model"
    result = compare.compare(base, worse)
    assert "amount_grounding_rate" in result["regressions"] and "case:a" in result["regressions"]
    assert result["changed_components"] == ["model"]
    assert result["case_changes"] == [{"id": "a", "baseline": True, "candidate": False, "change": "regressed"}]

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(base)), b.write_text(json.dumps(worse))
    assert compare.main([str(a), str(b)]) == 0
    assert compare.main([str(a), str(b), "--fail-on-regression"]) == 1
    assert compare.main([str(a), str(a), "--fail-on-regression"]) == 0
    assert "WORSE" in capsys.readouterr().out


def test_improvements_are_not_regressions():
    base = _report(grounding=0.9)
    better = copy.deepcopy(base)
    better["metrics"]["amount_grounding_rate"] = 1.0
    status = {m["metric"]: m["status"] for m in compare.compare(base, better)["metrics"]}
    assert status["amount_grounding_rate"] == "better" and compare.compare(base, better)["regressions"] == []
