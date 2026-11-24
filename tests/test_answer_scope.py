"""
Data-scope notes (app/answer_scope.py) and their use in the assistant loop:
period figures carry their date range and balances their as-of date, taken
from the tool results, even when the model leaves them out. Real customer
tools on the approved dataset (temp folder); scripted fake model; no Groq.
"""

import json
from pathlib import Path
import uuid
from decimal import Decimal as D

import pytest
from langchain_core.messages import AIMessage

import app.data_loader as dl
from app import assistant
from app.answer_scope import add_scope_note, dates_mentioned, scope_note, scopes_from_results
from evals.scorers import score_turn
from scripts import generate_data as gen

METRICS = json.dumps({"customer_id": "C106", "period": {"start": "2026-05-01", "end": "2026-11-21", "clamped": False},
                      "spending_gross": "288931.30", "spending_net": "283041.30", "refunds": "5890.00"})
OCTOBER = json.dumps({"customer_id": "C102", "period": {"start": "2026-10-01", "end": "2026-10-31", "clamped": False},
                      "total_credit": "306300.00", "total_debit": "65384.42"})
CLAMPED = json.dumps({"customer_id": "C102", "period": {"start": "2026-05-01", "end": "2026-06-30", "clamped": True},
                      "total_debit": "1000.00"})
TRANSACTIONS = json.dumps({"customer_id": "C106", "period": {"start": "2026-05-01", "end": "2026-11-21",
                                                             "clamped": False},
                           "matching_totals": {"debits": "20610.78", "credits": "5890.00"}, "rows": []})
PROFILE = json.dumps({"customer_id": "C101", "as_of_date": "2026-11-21", "total_balance": "187226.80",
                      "accounts": [{"account_id": "ACC1001", "balance": "105465.46"}]})
DISPUTES = json.dumps({"customer_id": "C111", "disputes": [{"dispute_id": "D0003", "amount": "7999.00"}]})
DENIED = json.dumps({"status": "ACCESS_DENIED", "message": "x",
                     "period": {"start": "2026-01-01", "end": "2026-01-31"}})


# ---------- The note itself ----------

def test_net_spending_without_its_period_gets_the_tool_period():
    answer = "C106's spending after accounting for refunds was ₹283,041.30."
    noted = add_scope_note(answer, [METRICS])
    assert noted == answer + "\n\nPeriod covered: 2026-05-01 to 2026-11-21 (inclusive)."
    assert noted.startswith(answer)                                   # the model's text is never altered


@pytest.mark.parametrize("answer", [
    "Net spending from 2026-05-01 to 2026-11-21 was ₹2,83,041.30.",
    "Between 1 May 2026 and 21 November 2026, gross spending was ₹2,88,931.30.",
    "From May 1, 2026 to November 21, 2026 net spending was ₹2,83,041.30.",
    "Gross spending (2026‑05‑01 to 2026‑11‑21): ₹2,88,931.30.",       # non-breaking hyphens
    "Spending 01/05/2026–21/11/2026: ₹2,83,041.30.",
])
def test_answers_that_already_state_the_period_are_unchanged(answer):
    assert add_scope_note(answer, [METRICS]) == answer


def test_a_partly_stated_period_still_gets_the_note():
    assert scope_note("Up to 21 November 2026 gross spending was ₹2,88,931.30.", [METRICS]) == \
        "Period covered: 2026-05-01 to 2026-11-21 (inclusive)."


def test_monthly_metrics_named_by_month_get_exact_dates():
    note = scope_note("In October 2026 C102 had credits of ₹3,06,300.00 and debits of ₹65,384.42.", [OCTOBER])
    assert note == "Period covered: 2026-10-01 to 2026-10-31 (inclusive)."


def test_transaction_totals_get_their_period_and_clamping_is_disclosed():
    assert scope_note("HomeNest debits total ₹20,610.78.", [TRANSACTIONS]) == \
        "Period covered: 2026-05-01 to 2026-11-21 (inclusive)."
    assert "limited to the available data" in scope_note("Debits were ₹1,000.00.", [CLAMPED])


def test_several_periods_list_only_the_missing_ones():
    answer = "October 2026: debits ₹65,384.42. From 2026-05-01 to 2026-11-21 spending was ₹2,83,041.30."
    assert scope_note(answer, [OCTOBER, METRICS]) == "Periods covered: 2026-10-01 to 2026-10-31 (inclusive)."


def test_balances_get_their_as_of_date():
    answer = "C101's total balance is ₹1,87,226.80."
    assert scope_note(answer, [PROFILE]) == "Balances as of 2026-11-21."
    assert scope_note("Total balance as of 21 November 2026: ₹1,87,226.80.", [PROFILE]) == ""
    assert scope_note("C101 has two accounts (ACC1001, ACC1002).", [PROFILE]) == ""   # no money, no note


def test_no_note_without_money_scope_or_valid_results():
    assert scope_note("C106 has 5 HomeNest transactions.", [TRANSACTIONS]) == ""          # no amount
    assert scope_note("Dispute D0003 (₹7,999.00) is resolved.", [DISPUTES]) == ""         # disputes have no period
    assert scope_note("I could not access that customer. ₹0.00", [DENIED]) == ""          # refusals ignored
    assert scopes_from_results(["not json", METRICS, METRICS]) == ([("2026-05-01", "2026-11-21", False)], [])


def test_the_note_goes_before_the_citation_marker():
    noted = add_scope_note("Spending was ₹2,83,041.30.\nCITED_POLICIES: NONE", [METRICS])
    assert noted.endswith("(inclusive).\nCITED_POLICIES: NONE")


def test_dates_mentioned_parses_common_forms():
    assert dates_mentioned("2026-05-01, 21 Nov 2026, May 3, 2026, 12‑Apr‑2019, 05/06/2026") == {
        "2026-05-01", "2026-11-21", "2026-05-03", "2019-04-12", "2026-06-05"}


def test_the_note_does_not_hide_incorrectly_labelled_figures():
    # Credits and debits swapped: the appended period must not make the answer pass.
    answer = add_scope_note("C102 had credits of ₹65,384.42 and debits of ₹3,06,300.00.", [OCTOBER])
    figures = [{"label": f, "value": D(v), "required": True,
                "spec": {"figure": "period", "field": f, "customer_id": "C102"}}
               for f, v in (("total_credits", "306300.00"), ("total_debits", "65384.42"))]
    obs = {"answer": answer, "sources": [], "requested_tools": [], "executed_tools": [], "blocked": [],
           "model_calls": [], "revisions": 0, "tool_limit": False, "error_kind": None, "latency_ms": 1}
    result = score_turn({}, obs, {D("306300.00"), D("65384.42")}, figures, "q")
    assert result["checks"]["figures"] is False


# ---------- In the assistant loop (real tools, approved dataset) ----------

@pytest.fixture(scope="module")
def approved_dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("approved")
    assert gen.main(["--output-dir", str(folder)]) == 0
    return dl.load_dataset(folder)


@pytest.fixture
def data(monkeypatch, approved_dataset):
    monkeypatch.setattr(dl, "_dataset", approved_dataset)
    return approved_dataset


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, messages):
        return self.responses.pop(0)


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{uuid.uuid4().hex[:6]}",
                                              "type": "tool_call"}])


@pytest.fixture
def session():
    sid = f"scope-{uuid.uuid4().hex[:8]}"
    yield sid
    for store in ("conversation_sessions", "grounding_context", "session_last_active", "conversation_versions"):
        getattr(assistant, store).pop(sid, None)


def ask(question, session, *responses):
    return assistant.ask_bankmate_with_sources(question, session_id=session, model=FakeModel(responses))["answer"]


def test_live_net_spending_answer_now_carries_its_period(data, session):
    # The exact answer the live model gave in phase2-live-6, with the real metrics tool.
    answer = ask("What was C106's spending after refunds?", session, call("get_customer_metrics", customer_id="C106"),
                 AIMessage(content="C106’s spending after accounting for refunds was ₹283,041.30."))
    assert answer == ("C106’s spending after accounting for refunds was ₹283,041.30.\n\n"
                      "Period covered: 2026-05-01 to 2026-11-21 (inclusive).")
    assert assistant.conversation_sessions[session][-1].content == answer   # history matches what the user saw


def test_gross_spending_and_monthly_metrics_through_the_loop(data, session):
    answer = ask("C106's gross spending?", session, call("get_customer_metrics", customer_id="C106"),
                 AIMessage(content="Gross spending was ₹2,88,931.30."))
    assert answer.endswith("Period covered: 2026-05-01 to 2026-11-21 (inclusive).")
    answer = ask("C102 credits and debits in October 2026?", session,
                 call("get_customer_metrics", customer_id="C102", start_date="2026-10-01", end_date="2026-10-31"),
                 AIMessage(content="Credits ₹3,06,300.00 and debits ₹65,384.42 in October."))
    assert answer.endswith("Period covered: 2026-10-01 to 2026-10-31 (inclusive).")


def test_transaction_totals_and_balances_through_the_loop(data, session):
    answer = ask("C106's HomeNest totals?", session,
                 call("get_customer_transactions", customer_id="C106", merchant="HomeNest"),
                 AIMessage(content="HomeNest debits total ₹20,610.78 and credits ₹5,890.00."))
    assert answer.endswith("Period covered: 2026-05-01 to 2026-11-21 (inclusive).")
    answer = ask("What is C101's balance?", session, call("get_customer_profile", customer_id="C101"),
                 AIMessage(content="C101's total balance is ₹1,87,226.80."))
    assert answer.endswith("Balances as of 2026-11-21.")


def test_a_follow_up_without_tools_inherits_the_previous_period(data, session):
    ask("C102 debits in October 2026?", session,
        call("get_customer_metrics", customer_id="C102", start_date="2026-10-01", end_date="2026-10-31"),
        AIMessage(content="Debits were ₹65,384.42 from 2026-10-01 to 2026-10-31."))
    follow_up = ask("And the credits?", session, AIMessage(content="Credits were ₹3,06,300.00."))
    assert follow_up == "Credits were ₹3,06,300.00.\n\nPeriod covered: 2026-10-01 to 2026-10-31 (inclusive)."


def test_answers_that_state_their_scope_or_have_no_figures_are_untouched(data, session):
    text = "From 2026-05-01 to 2026-11-21, C106's net spending was ₹2,83,041.30."
    assert ask("C106's net spending?", session, call("get_customer_metrics", customer_id="C106"),
               AIMessage(content=text)) == text
    assert ask("How many disputes does C111 have?", session, call("get_customer_disputes", customer_id="C111"),
               AIMessage(content="C111 has one dispute, D0003, which is resolved.")) == \
        "C111 has one dispute, D0003, which is resolved."


# ---------- Release pass: relevance, multi-turn isolation, contradictory dates, fingerprint ----------

from app.answer_scope import CONFLICT_NOTE, relevant_results  # noqa: E402
from evals import answers as eval_answers  # noqa: E402
from evals import compare, run  # noqa: E402

SEPTEMBER = json.dumps({"customer_id": "C102", "period": {"start": "2026-09-01", "end": "2026-09-30", "clamped": False},
                        "total_credit": "54500.00", "total_debit": "48484.58"})
PROFILE_FULL = json.dumps({"customer_id": "C101", "customer_since": "2019-04-12", "as_of_date": "2026-11-21",
                           "total_balance": "187226.80", "accounts": [{"account_id": "ACC1001", "balance": "105465.46",
                                                                       "opened_on": "2019-04-12"}]})


def test_only_results_whose_amounts_appear_in_the_answer_lend_their_scope():
    answer = "C102's October debits were ₹65,384.42."
    assert scope_note(answer, [OCTOBER, SEPTEMBER]) == "Period covered: 2026-10-01 to 2026-10-31 (inclusive)."
    assert scope_note("C106 has 5 HomeNest transactions.", [TRANSACTIONS]) == ""        # a count is not an amount
    assert scope_note("Refunds above ₹500.00 need approval.", [METRICS]) == ""          # amount not from the data


@pytest.mark.parametrize("answer", [
    "C106's spending after refunds was 283041.30.",                                    # unmarked
    "Net spending: 2,83,041.30",                                                       # Indian grouping, no ₹
    "| Metric | Value |\n|---|---|\n| Net spending | 283 041,30 |",           # space grouping, comma
    "Net spending was ₹283,041.3.",                                                    # one decimal
])
def test_unmarked_and_unusually_formatted_amounts_are_recognised(answer):
    assert scope_note(answer, [METRICS]) == "Period covered: 2026-05-01 to 2026-11-21 (inclusive)."


@pytest.mark.parametrize("answer", [
    "Debits were 5890 in total.",
    "Refunds of ₹1,953.78 and 5890 were posted.",
    "| Txn | Amount (₹) |\n|---|---|\n| T00637 | 5 890,00 |",
])
def test_scope_relevance_agrees_with_the_evaluators_amount_detection(answer):
    result = json.dumps({"customer_id": "C106", "period": {"start": "2026-05-01", "end": "2026-11-21"},
                         "matching_totals": {"debits": "5890.00", "refunds": "1953.78"}})
    detected = {a.value for a in eval_answers.read(answer).amounts}
    assert detected & {D("5890.00"), D("1953.78")}                                    # evaluator sees money...
    assert relevant_results(answer, [result]) and scope_note(answer, [result])          # ...and so does the note


@pytest.mark.parametrize("answer", [
    "From 2026-06-01 to 2026-11-21, net spending was ₹2,83,041.30.",                    # wrong start, inside window
    "Between 1 May 2026 and 30 November 2026 net spending was ₹2,83,041.30.",           # wrong end
    "As of 2026-04-30, net spending was ₹2,83,041.30.",                                 # date outside the period
])
def test_contradictory_dates_are_flagged_and_no_conflicting_range_is_added(answer):
    noted = add_scope_note(answer, [METRICS])
    assert noted == answer + "\n\n" + CONFLICT_NOTE
    assert "Period covered" not in noted


def test_wrong_balance_as_of_date_is_flagged_but_profile_dates_are_not():
    assert scope_note("Balance as of 2026-10-31: ₹1,87,226.80.", [PROFILE_FULL]) == CONFLICT_NOTE
    assert scope_note("C101 (customer since 12 April 2019) has a total balance of ₹1,87,226.80.",
                      [PROFILE_FULL]) == "Balances as of 2026-11-21."


def test_dates_inside_the_period_and_correct_ranges_in_any_format_are_not_conflicts():
    listing = ("| ID | Date | Amount (₹) |\n|---|---|---|\n| T01603 | 2026‑09‑16 | 4,686.00 |\n"
               "| T00637 | 27 June 2026 | 5,890.00 |")
    result = json.dumps({"customer_id": "C106", "period": {"start": "2026-05-01", "end": "2026-11-21"},
                         "rows": [["T01603", "2026-09-16", "4686.00"], ["T00637", "2026-06-27", "5890.00"]]})
    assert scope_note(listing, [result]) == "Period covered: 2026-05-01 to 2026-11-21 (inclusive)."
    for answer in ("1 May 2026 – 21 November 2026: net spending ₹2,83,041.30.",
                   "From May 1, 2026 through Nov 21, 2026 net spending was ₹2,83,041.30."):
        assert scope_note(answer, [METRICS]) == "", answer


def test_multiple_periods_one_wrong_is_flagged():
    answer = "From 2026-10-01 to 2026-10-31 debits were ₹65,384.42; from 2026-09-01 to 2026-09-29 ₹48,484.58."
    assert scope_note(answer, [OCTOBER, SEPTEMBER]) == CONFLICT_NOTE


def _october_c102(session):
    ask("C102 debits in October 2026?", session,
        call("get_customer_metrics", customer_id="C102", start_date="2026-10-01", end_date="2026-10-31"),
        AIMessage(content="Debits were ₹65,384.42."))


def test_customer_switch_never_inherits_the_previous_customers_period(data, session):
    _october_c102(session)
    switched = ask("And C101?", session, AIMessage(content="I'd need to look that up; C102's figure was ₹65,384.42."))
    assert switched == "I'd need to look that up; C102's figure was ₹65,384.42."            # C101 named: no inheritance


def test_follow_up_about_the_same_customer_inherits_the_period(data, session):
    _october_c102(session)
    same = ask("And C102's credits?", session, AIMessage(content="Credits were ₹3,06,300.00."))
    assert same.endswith("Period covered: 2026-10-01 to 2026-10-31 (inclusive).")


def test_unrelated_follow_ups_policy_turns_and_older_turns_lend_no_dates(data, session, monkeypatch):
    from app import tools as app_tools
    policy = json.dumps({"status": "SUCCESS", "notice": app_tools.UNTRUSTED_EVIDENCE_NOTICE, "sources": [],
                         "evidence": []})
    monkeypatch.setitem(assistant.tool_map, "search_banking_policies",
                        type("P", (), {"invoke": lambda self, args: policy})())
    ask("C106's net spending?", session, call("get_customer_metrics", customer_id="C106"),
        AIMessage(content="Net spending was ₹2,83,041.30."))
    unrelated = ask("Is there a fee for disputes?", session, AIMessage(content="Fees above ₹500.00 are not covered."))
    assert unrelated == "Fees above ₹500.00 are not covered."                               # amount not from data
    older = ask("Thanks. Remind me of the spending?", session, AIMessage(content="It was ₹2,83,041.30."))
    assert older == "It was ₹2,83,041.30."                                                  # previous turn had no tools
    ask("What does the policy say about refunds?", session, call("search_banking_policies", question="refunds"),
        AIMessage(content="The policy describes refund tracking.\nCITED_POLICIES: NONE"))
    after_policy = ask("And the spending again?", session, AIMessage(content="₹2,83,041.30."))
    assert after_policy == "₹2,83,041.30."


def test_several_financial_questions_each_get_their_own_period(data, session):
    first = ask("C102 debits in October 2026?", session,
                call("get_customer_metrics", customer_id="C102", start_date="2026-10-01", end_date="2026-10-31"),
                AIMessage(content="Debits were ₹65,384.42."))
    second = ask("C106's net spending?", session, call("get_customer_metrics", customer_id="C106"),
                 AIMessage(content="Net spending was ₹2,83,041.30."))
    assert first.endswith("2026-10-01 to 2026-10-31 (inclusive).")
    assert second.endswith("2026-05-01 to 2026-11-21 (inclusive).") and "2026-10-01" not in second


# ---------- Answer processing is part of the reproducibility fingerprint ----------

def test_answer_processing_code_is_fingerprinted(tmp_path):
    copy_ = tmp_path / "answer_scope.py"
    copy_.write_text(Path(run.PROJECT_ROOT / "app" / "answer_scope.py").read_text())
    before = run.answer_processing_hashes(files=[copy_], functions=[("app.assistant", "_scope_results")])
    copy_.write_text(copy_.read_text() + "\n# changed\n")
    after = run.answer_processing_hashes(files=[copy_], functions=[("app.assistant", "_scope_results")])
    assert before["answer_scope.py"] != after["answer_scope.py"]
    assert before["app.assistant._scope_results"] == after["app.assistant._scope_results"]
    names = set(run.answer_processing_hashes())
    assert {"answer_scope.py", "citations.py", "grounding.py", "app.assistant._answer",
            "app.assistant._scope_results", "app.security.sanitize_answer"} <= names


def test_changed_answer_processing_shows_in_comparison_and_replay_notes_it(data, tmp_path, monkeypatch):
    baseline = run.run_offline(only=["metrics_october_c102"])
    edited = tmp_path / "answer_scope.py"
    edited.write_text(Path(run.PROJECT_ROOT / "app" / "answer_scope.py").read_text() + "\n# changed\n")
    monkeypatch.setattr(run, "ANSWER_PROCESSING_FILES", (edited,) + run.ANSWER_PROCESSING_FILES[1:])
    candidate = run.run_offline(only=["metrics_october_c102"])
    diff = compare.compare(baseline, candidate)
    assert diff["changed_components"] == ["answer_processing_sha256"] and diff["same_execution"] is False
    case = eval_cases_one("metrics_october_c102")
    recorded = {"execution": {k: baseline["run"]["components"][k] for k in run.EXECUTION_STRICT + run.EXECUTION_NOTED},
                "case_input_sha256": {case["id"]: run.case_input_sha256(case)},
                "case_expect_sha256": {case["id"]: run.case_expect_sha256(case)},
                "dataset_sha256": baseline["run"]["components"]["dataset_sha256"]}
    info = run.verify_replay_inputs(tmp_path / "x.cassette.json", recorded, [case], candidate["run"]["components"])
    assert [n for n in info["notes"] if "answer_processing_sha256" in n]          # noted, not refused
    recorded["execution"]["system_prompt_sha256"] = "changed"                              # strict ones still refuse
    from evals.cassette import CassetteMismatch
    with pytest.raises(CassetteMismatch, match="system_prompt_sha256"):
        run.verify_replay_inputs(tmp_path / "x.cassette.json", recorded, [case], candidate["run"]["components"])


def eval_cases_one(case_id):
    from evals.cases import load_cases
    return load_cases(only=[case_id])[0]
