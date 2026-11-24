"""
Live-mode evaluation infrastructure (evals/live.py, evals/cassette.py,
evals/run.py --mode live/replay) with a fake Groq client, fake clocks
and temporary ChromaDB folders: no network, no real waiting, and the
repository's chroma_db/ and data/ are never touched.
"""

import copy
import json
import time

import groq
import httpx
import pytest

import app.data_loader as dl
from app import assistant, llm, tools
from app import vector_store as vs
from evals import cases as eval_cases
from evals import compare, run
from evals.cassette import (CassetteError, CassetteMismatch, CassetteMissing, CassetteOrderError,
                            messages_digest)
from evals.fakes import FakeGroq, FakePolicySearch
from evals.live import (GuardedModel, LiveRunRefused, ModelCallBudget, StopRun, TokenPacer, classify,
                        estimate_tokens, tree_sha256)
from evals.recorder import count_revisions
from scripts import generate_data as gen

SIX = ["profile_balances_c101", "metrics_october_c102", "net_spending_c106",       # case-file order
       "transactions_merchant_c106", "dispute_status_c111", "policy_refund_approved"]
from evals.scorers import MODE_DEPENDENT  # noqa: E402  (latency, pacing, tokens consumed)


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
    """The real Groq client can never be created or called in these tests."""

    def forbidden(*args, **kwargs):
        raise AssertionError("a test tried to reach the real Groq model")

    monkeypatch.setattr(assistant, "_get_model", forbidden)
    monkeypatch.setattr(llm, "ChatGroq", forbidden)


@pytest.fixture
def fake_index(tmp_path):
    """A stand-in index folder (content irrelevant when check_index is faked)."""

    folder = tmp_path / "index-source"
    folder.mkdir()
    (folder / "chroma.sqlite3").write_bytes(b"not a real index")
    return folder


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def cases_for(ids, path=eval_cases.CASES_FILE):
    return eval_cases.load_cases(path, ids)


def live(ids, fake_index, cases_path=eval_cases.CASES_FILE, model=None, clock=None, **settings):
    cases = cases_for(ids, cases_path)
    model = model or FakeGroq(cases)
    clock = clock or FakeClock()
    report, cassette = run.run_live(
        cases_path, ids, confirm=True, model=model,
        policy_tool=FakePolicySearch(run.policy_chunks(), cases),
        index_source=fake_index, check_index=lambda path: None, clock=clock, sleep=clock.sleep, **settings)
    return report, cassette, model, clock


def save_and_replay(report, cassette, tmp_path, name="baseline", **kwargs):
    report_path, cassette_path = run.save_live(report, cassette, name, out_dir=tmp_path)
    return report_path, cassette_path, run.run_replay(cassette_path, **kwargs)


def rate_limit(retry_after="7"):
    response = httpx.Response(429, headers={"retry-after": retry_after},
                              request=httpx.Request("POST", "https://api.groq.com/x"))
    return groq.RateLimitError("rate limited", response=response, body=None)


def auth_error():
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com/x"))
    return groq.AuthenticationError("invalid key", response=response, body=None)


def connection_error():
    return groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))


def comparable(metrics):
    return {k: v for k, v in metrics.items() if k not in MODE_DEPENDENT}


# ---------- Refusals happen before any model call ----------

def test_live_run_needs_explicit_confirmation_and_named_cases(data, no_groq, fake_index):
    cases = cases_for(SIX)
    model = FakeGroq(cases)
    with pytest.raises(LiveRunRefused, match="confirm"):
        run.run_live(only=SIX, model=model, index_source=fake_index, check_index=lambda p: None)
    with pytest.raises(LiveRunRefused, match="--case"):
        run.run_live(only=None, confirm=True, model=model, index_source=fake_index, check_index=lambda p: None)
    assert model.calls == 0


def test_case_budget_and_impossible_call_budget_are_refused(data, no_groq, fake_index):
    model = FakeGroq(cases_for(SIX))
    with pytest.raises(LiveRunRefused, match="--max-cases is 2"):
        run.run_live(only=SIX[:3], confirm=True, model=model, max_cases=2,
                     index_source=fake_index, check_index=lambda p: None)
    with pytest.raises(LiveRunRefused, match="cannot cover even one case"):
        run.run_live(only=SIX[:1], confirm=True, model=model, max_model_calls=3,
                     index_source=fake_index, check_index=lambda p: None)
    assert model.calls == 0


def test_cli_refuses_live_without_confirmation(data, no_groq, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-key")
    monkeypatch.setattr(run, "BASELINES_DIR", tmp_path)
    assert run.main(["--mode", "live", "--case", SIX[0]]) == 2
    assert "no API call was made" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_live_run_refuses_an_invalid_index_copy(data, no_groq, tmp_path):
    empty = tmp_path / "empty-index"
    empty.mkdir()
    model = FakeGroq(cases_for(SIX))
    with pytest.raises(LiveRunRefused, match="policy index is missing"):
        run.run_live(only=SIX[:1], confirm=True, model=model, index_source=empty)
    assert model.calls == 0 and vs.CHROMA_DIR.name == "chroma_db"


# ---------- The six-case run: record, save, replay ----------

def test_six_case_fake_live_run_records_everything_and_replays_identically(data, no_groq, fake_index, tmp_path):
    dataset_before = tree_sha256(data.folder)
    production_index_before = tree_sha256(vs.CHROMA_DIR)

    report, cassette, model, clock = live(SIX, fake_index)
    info = report["live"]
    assert report["run"]["mode"] == "live" and report["run"]["model"]["id"] == FakeGroq.model_name
    assert info["completed_cases"] == SIX and info["stopped"] is None and info["not_run"] == []
    assert report["metrics"]["case_pass_rate"] == 1.0
    assert info["model_calls"] == model.calls == 12 and info["budget"]["used"] == 12
    assert info["integrity"] == {"dataset_unchanged": True, "index_unchanged": True, "index_isolated": True}
    assert report["metrics"]["tokens_total"] > 0

    # Every model response, with case/turn/step identifiers, usage and an input digest.
    assert set(cassette["cases"]) == set(SIX)
    for case_id in SIX:
        (turn,) = cassette["cases"][case_id]["turns"]
        assert [e["step"] for e in turn["model"]] == [0, 1]
        assert all(len(e["input_sha256"]) == 64 and e["response"]["usage_metadata"]["total_tokens"] > 0
                   for e in turn["model"])
        assert turn["model"][0]["response"]["tool_calls"] and turn["model"][1]["response"]["content"]
    (policy_entry,) = cassette["cases"]["policy_refund_approved"]["turns"][0]["policy"]
    assert policy_entry["step"] == 0 and json.loads(policy_entry["output"])["status"] == "SUCCESS"
    roles = [c["role"] for c in report["cases"][0]["turns"][0]["observed"]["model_calls"]]
    assert roles == ["tool_calls", "answer"]

    report_path, cassette_path, replayed = save_and_replay(report, cassette, tmp_path)
    saved = json.loads(report_path.read_text())
    assert saved["live"]["cassette"]["file"] == cassette_path.name
    assert replayed["run"]["mode"] == "replay" and replayed["run"]["replay_of"]["input_drift"] == []
    assert comparable(replayed["metrics"]) == comparable(report["metrics"])
    for live_case, replay_case in zip(report["cases"], replayed["cases"]):
        assert live_case["id"] == replay_case["id"] and live_case["passed"] == replay_case["passed"]
        for a, b in zip(live_case["turns"], replay_case["turns"]):
            assert a["observed"]["answer"] == b["observed"]["answer"]
            assert a["observed"]["sources"] == b["observed"]["sources"]
            assert a["score"] == b["score"]
    diff = compare.compare(saved, replayed)
    assert diff["regressions"] == [] and diff["changed_components"] == [] and diff["case_changes"] == []

    assert model.calls == 12                                        # replay made no model call
    assert tree_sha256(data.folder) == dataset_before
    assert tree_sha256(vs.CHROMA_DIR) == production_index_before


def test_replay_cli_writes_a_report(data, no_groq, fake_index, tmp_path, capsys):
    report, cassette, _, _ = live(SIX[:2], fake_index)
    _, cassette_path = run.save_live(report, cassette, "b", out_dir=tmp_path)
    out = tmp_path / "replay.json"
    assert run.main(["--mode", "replay", "--cassette", str(cassette_path), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["run"]["mode"] == "replay"
    assert "mode=replay" in capsys.readouterr().out


# ---------- Budgets and pacing ----------

def test_model_call_budget_stops_before_a_case_it_cannot_cover(data, no_groq, fake_index):
    report, cassette, model, _ = live(SIX, fake_index, max_model_calls=8)
    info = report["live"]
    assert info["completed_cases"] == SIX[:2] and model.calls == 4            # 4 left < worst case 6
    assert info["stopped"]["kind"] == "budget" and info["stopped"]["case"] == SIX[2]
    assert info["not_run"] == SIX[2:] and set(cassette["cases"]) == set(SIX[:2])


def test_budget_is_a_hard_cap_on_every_call():
    budget = ModelCallBudget(2)
    model = GuardedModel(type("M", (), {"invoke": lambda self, m: _ai("ok")})(), budget,
                         TokenPacer(10**6, 1000, clock=lambda: 0.0, sleep=lambda s: None), classify)
    model.invoke([]), model.invoke([])
    with pytest.raises(StopRun, match="budget of 2"):
        model.invoke([])
    assert budget.used == 2


def _ai(content, total=None):
    from langchain_core.messages import AIMessage

    message = AIMessage(content=content)
    if total:
        message.usage_metadata = {"input_tokens": total - 10, "output_tokens": 10, "total_tokens": total}
    return message


def test_token_pacer_waits_for_the_window_without_real_sleeping():
    clock = FakeClock()
    pacer = TokenPacer(tokens_per_minute=1000, requests_per_minute=3, clock=clock, sleep=clock.sleep)
    pacer.wait_for(400)
    pacer.record(800)
    assert clock.sleeps == []
    pacer.wait_for(400)                                     # 800 + 400 > 1000: wait for the window
    assert clock.sleeps == [60.5] and clock.now == 1060.5
    for _ in range(3):
        pacer.wait_for(1)
        pacer.record(1)
    pacer.wait_for(1)                                        # request limit reached
    assert len(clock.sleeps) == 2 and pacer.waits == 2
    pacer.events.clear()
    pacer.wait_for(5000)                                     # larger than the limit: sent alone, no deadlock
    assert len(clock.sleeps) == 2


def test_estimates_are_pessimistic():
    from langchain_core.messages import HumanMessage

    assert estimate_tokens([HumanMessage(content="x" * 3000)]) == 1000 + 1024


def test_live_run_paces_calls_and_cases_with_a_fake_clock(data, no_groq, fake_index):
    started = time.monotonic()
    report, _, _, clock = live(SIX[:3], fake_index, tokens_per_minute=4000, case_pause_seconds=5)
    assert report["live"]["completed_cases"] == SIX[:3]
    assert clock.sleeps.count(5) == 2                        # between cases
    assert any(s > 30 for s in clock.sleeps)                 # token window waits
    assert report["live"]["pacing"]["waits"] >= 1 and report["live"]["pacing"]["waited_seconds"] > 60
    assert time.monotonic() - started < 30                   # nothing really slept
    calls = [c for case in report["cases"] for t in case["turns"] for c in t["observed"]["model_calls"]]
    assert any(c.get("paced_ms", 0) > 30000 for c in calls)  # the wait is recorded separately...
    assert all(c["latency_ms"] < 30000 for c in calls)       # ...and never counted as model latency


def test_reported_usage_feeds_the_pacer_and_calibrates_the_estimate():
    from langchain_core.messages import HumanMessage

    clock = FakeClock()
    pacer = TokenPacer(10**6, 1000, clock=clock, sleep=clock.sleep)
    model = GuardedModel(type("M", (), {"invoke": lambda self, m: _ai("ok", total=777)})(),
                         ModelCallBudget(5), pacer, classify)
    model.invoke([HumanMessage(content="x" * 3836)])            # reported input: 767 tokens = 5.0 chars/token
    assert [tokens for _t, tokens in pacer.events] == [777]
    assert model.chars_per_token == pytest.approx(4.5, rel=1e-3)          # 10% below the observed ratio


# ---------- Stop on the first provider failure ----------

@pytest.mark.parametrize("make_error, kind", [(rate_limit, "rate_limited"), (auth_error, "authentication"),
                                              (connection_error, "connection")])
def test_first_fatal_provider_failure_stops_the_whole_run(data, no_groq, fake_index, make_error, kind):
    cases = cases_for(SIX)
    model = FakeGroq(cases, fail_at={3: make_error()})            # first call of the second case
    report, cassette, _, _ = live(SIX, fake_index, model=model)
    info = report["live"]
    assert model.calls == 3                                         # no retry by the runner
    assert info["stopped"]["kind"] == kind and info["stopped"]["case"] == SIX[1]
    assert info["completed_cases"] == SIX[:1] and info["not_run"] == SIX[2:]
    assert set(cassette["cases"]) == {SIX[0]}                       # the aborted case is not kept
    assert [c["id"] for c in report["cases"]] == SIX[:1]
    if kind == "rate_limited":
        assert info["stopped"]["retry_after"] == 7


def test_non_fatal_failure_is_scored_and_replayed(data, no_groq, fake_index, tmp_path):
    cases = cases_for(SIX[:2])
    model = FakeGroq(cases, fail_at={1: TimeoutError("slow")})
    report, cassette, _, _ = live(SIX[:2], fake_index, model=model)
    first = report["cases"][0]["turns"][0]
    assert report["live"]["stopped"] is None and first["observed"]["error_kind"] == "timeout"
    assert cassette["cases"][SIX[0]]["turns"][0]["model"] == [
        {"step": 0, "input_sha256": cassette["cases"][SIX[0]]["turns"][0]["model"][0]["input_sha256"],
         "error": "timeout"}]
    _, _, replayed = save_and_replay(report, cassette, tmp_path)
    assert replayed["cases"][0]["turns"][0]["observed"]["error_kind"] == "timeout"
    assert comparable(replayed["metrics"]) == comparable(report["metrics"])


# ---------- Grounding revisions ----------

@pytest.fixture
def revision_cases(tmp_path):
    """A policy case whose answer names a contact channel the evidence does not support."""

    case = copy.deepcopy(next(c for c in eval_cases.load_cases() if c["id"] == "policy_refund_approved"))
    case["id"] = "policy_revision_probe"
    case["turns"][0]["offline"]["script"][1]["answer"] = (
        "Once approved, the refund is processed. You can call us by phone to check it.\nCITED_POLICIES: POL-002")
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case) + "\n")
    return path


def test_revisions_are_detected_from_the_call_sequence_and_replayed(data, no_groq, fake_index, tmp_path,
                                                                      revision_cases):
    offline = run.run_offline(revision_cases)
    assert offline["cases"][0]["turns"][0]["observed"]["revisions"] == 1

    report, cassette, model, _ = live(["policy_revision_probe"], fake_index, cases_path=revision_cases)
    observed = report["cases"][0]["turns"][0]["observed"]
    assert observed["revisions"] == 1 and report["metrics"]["grounding_revisions"] == 1
    assert [c["role"] for c in observed["model_calls"]] == ["tool_calls", "answer", "revision"]
    assert len(cassette["cases"]["policy_revision_probe"]["turns"][0]["model"]) == 3

    _, _, replayed = save_and_replay(report, cassette, tmp_path, cases_path=revision_cases)
    assert replayed["cases"][0]["turns"][0]["observed"]["revisions"] == 1


def test_rate_limit_on_the_revision_call_still_stops_the_run(data, no_groq, fake_index, revision_cases):
    # The assistant swallows exceptions from the revision call; StopRun must get through.
    cases = cases_for(["policy_revision_probe"], revision_cases)
    model = FakeGroq(cases, fail_at={3: rate_limit()})
    report, cassette, _, _ = live(["policy_revision_probe"], fake_index, cases_path=revision_cases, model=model)
    assert report["live"]["stopped"]["kind"] == "rate_limited" and report["cases"] == []
    assert cassette["cases"] == {}


def test_count_revisions():
    assert count_revisions([{"final": False}, {"final": True}]) == 0
    assert count_revisions([{"final": False}, {"final": True}, {"final": True}]) == 1
    assert count_revisions([]) == 0


# ---------- Secrets ----------

class LeakyGroq(FakeGroq):
    """Puts the API key into its answers and metadata, to prove nothing persists it."""

    def __init__(self, cases, key):
        super().__init__(cases)
        self.key = key

    def invoke(self, messages):
        reply = super().invoke(messages)
        if reply.content:
            reply.content += f" (debug {self.key})"
        reply.response_metadata["model_name"] = self.key
        return reply


def test_api_key_is_never_written_to_reports_or_cassettes(data, no_groq, fake_index, tmp_path, monkeypatch):
    key = "gsk_" + "Z9" * 24
    monkeypatch.setenv("GROQ_API_KEY", key)
    cases = cases_for(SIX[:2])
    report, cassette, _, _ = live(SIX[:2], fake_index, model=LeakyGroq(cases, key))
    report_path, cassette_path = run.save_live(report, cassette, "leak", out_dir=tmp_path)
    for path in (report_path, cassette_path):
        text = path.read_text()
        assert key not in text and "[REDACTED]" in text
    other = "custom-secret-value-123"
    report, cassette, _, _ = live(SIX[:1], fake_index, model=LeakyGroq(cases_for(SIX[:1]), other))
    report_path, cassette_path = run.save_live(report, cassette, "leak2", out_dir=tmp_path, secrets=[other])
    assert other not in report_path.read_text() and other not in cassette_path.read_text()


# ---------- Replay failures are explicit and never reach Groq ----------

@pytest.fixture
def recorded(data, no_groq, fake_index, tmp_path):
    ids = ["metrics_october_c102", "policy_refund_approved"]
    report, cassette, _, _ = live(ids, fake_index)
    return cassette, tmp_path


def _replay_mutated(recorded, mutate, **kwargs):
    cassette, tmp_path = recorded
    broken = copy.deepcopy(cassette)
    mutate(broken)
    path = tmp_path / "broken.cassette.json"
    path.write_text(json.dumps(broken))
    return run.run_replay(path, **kwargs)


def _model_entries(cassette, case_id="metrics_october_c102"):
    return cassette["cases"][case_id]["turns"][0]["model"]


def test_replay_fails_on_a_missing_model_response(recorded):
    with pytest.raises(CassetteMissing, match="model entry #2"):
        _replay_mutated(recorded, lambda c: _model_entries(c).pop())


def test_replay_fails_on_out_of_sequence_entries(recorded):
    def swap(c):
        entries = _model_entries(c)
        entries[0], entries[1] = entries[1], entries[0]
    with pytest.raises(CassetteOrderError, match="out of sequence"):
        _replay_mutated(recorded, swap)


def test_replay_fails_when_recorded_entries_go_unused(recorded):
    def extra(c):
        entries = _model_entries(c)
        entries.append({**copy.deepcopy(entries[-1]), "step": len(entries)})
    with pytest.raises(CassetteOrderError, match="unused"):
        _replay_mutated(recorded, extra)


def test_replay_fails_on_changed_model_input_unless_drift_is_allowed(recorded):
    def drift(c):
        _model_entries(c)[1]["input_sha256"] = "0" * 64
    with pytest.raises(CassetteMismatch, match="model input differs"):
        _replay_mutated(recorded, drift)
    report = _replay_mutated(recorded, drift, strict_inputs=False)
    assert report["run"]["replay_of"]["input_drift"] == ["case metrics_october_c102 turn 1 model call #2"]


def test_replay_fails_on_a_missing_or_changed_policy_entry(recorded):
    policy = lambda c: c["cases"]["policy_refund_approved"]["turns"][0]["policy"]      # noqa: E731
    with pytest.raises(CassetteMissing, match="policy entry #1"):
        _replay_mutated(recorded, lambda c: policy(c).clear())
    with pytest.raises(CassetteMismatch, match="query differs"):
        _replay_mutated(recorded, lambda c: policy(c)[0].update(args={"question": "something else"}))


def test_replay_fails_on_missing_case_changed_case_or_wrong_file(recorded, tmp_path):
    cassette, _ = recorded
    path = tmp_path / "ok.cassette.json"
    path.write_text(json.dumps(cassette))
    with pytest.raises(CassetteError, match="not in the cassette: net_spending_c106"):
        run.run_replay(path, only=["net_spending_c106"])
    with pytest.raises(CassetteMismatch, match="questions or injected tool failures changed"):
        _replay_mutated(recorded, lambda c: c["run"]["case_input_sha256"].update(metrics_october_c102="x"))
    with pytest.raises(CassetteMismatch, match="dataset differs"):
        _replay_mutated(recorded, lambda c: c["run"].update(dataset_sha256="x"))
    not_cassette = tmp_path / "report.json"
    not_cassette.write_text(json.dumps({"run": {}}))
    with pytest.raises(CassetteMismatch, match="not an evaluation cassette"):
        run.run_replay(not_cassette)


def test_replay_can_never_create_the_live_model(data):
    with run.no_live_model():
        with pytest.raises(CassetteError, match="never calls Groq"):
            assistant._get_model()
        with pytest.raises(CassetteError):
            llm.get_assistant_model()
    assert llm.get_assistant_model is not None and assistant._get_model.__name__ == "_get_model"


def test_cassette_errors_escape_the_assistants_error_handling(recorded):
    # A CassetteError is a BaseException: never converted into an AssistantError or swallowed.
    assert not issubclass(CassetteError, Exception) and not issubclass(StopRun, Exception)


def test_message_digest_is_sensitive_to_content_and_tool_calls():
    from langchain_core.messages import AIMessage, HumanMessage

    a = [HumanMessage(content="q"), AIMessage(content="", tool_calls=[{"name": "t", "args": {"x": 1}, "id": "1"}])]
    b = [HumanMessage(content="q"), AIMessage(content="", tool_calls=[{"name": "t", "args": {"x": 2}, "id": "1"}])]
    assert messages_digest(a) == messages_digest(copy.deepcopy(a)) != messages_digest(b)


# ---------- Production wiring and index isolation ----------

def test_live_mode_uses_the_apps_own_model_and_tools(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(assistant, "_model", sentinel)
    model = run.production_model()                                   # constructing makes no request
    assert run.model_id_of(model) == "openai/gpt-oss-20b"
    assert model.bound.temperature == 0.2 and model.bound.request_timeout == llm.LLM_TIMEOUT_SECONDS
    assert {t["function"]["name"] for t in model.kwargs["tools"]} == set(assistant.tool_map)
    assert run.sdk_retries_of(model) == {"chat_model": 0, "sdk_client": 0}
    # The application is unaffected: its cached model, retry setting and fresh models keep one SDK retry.
    assert assistant._model is sentinel and llm.LLM_MAX_RETRIES == 1
    assert run.sdk_retries_of(llm.get_assistant_model()) == {"chat_model": 1, "sdk_client": 1}

    driver = run.LiveDriver(sentinel, run.CassetteWriter())
    mapped = driver.tool_map(assistant.tool_map)
    assert set(mapped) == set(assistant.tool_map)
    assert mapped["search_banking_policies"].tool is assistant.tool_map["search_banking_policies"]
    assert all(mapped[name] is assistant.tool_map[name] for name in mapped if name != "search_banking_policies")


def test_live_run_is_refused_if_retries_cannot_be_disabled(monkeypatch):
    monkeypatch.setattr(run, "sdk_retries_of", lambda model: {"chat_model": 0, "sdk_client": 2})
    with pytest.raises(LiveRunRefused, match="automatic retries"):
        run.production_model()


# ---------- Exactly one request per provider failure (real ChatGroq, HTTP intercepted) ----------

class GroqTransport:
    """Replaces httpx.Client.send: counts requests to Groq and fails each one. Nothing reaches the network."""

    def __init__(self, failure):
        self.failure, self.requests = failure, []

    def send(self, client, request, **kwargs):
        self.requests.append(str(request.url))
        if self.failure == "connection":
            raise httpx.ConnectError("connection refused", request=request)
        status = {"rate_limited": 429, "authentication": 401}[self.failure]
        return httpx.Response(status, request=request, json={"error": {"message": "fail"}},
                              headers={"retry-after": "7", "retry-after-ms": "1"})


@pytest.fixture
def transport(monkeypatch):
    def install(failure):
        fake = GroqTransport(failure)
        monkeypatch.setattr(httpx.Client, "send", lambda client, request, **kw: fake.send(client, request, **kw))
        return fake
    return install


@pytest.mark.parametrize("failure", ["rate_limited", "authentication", "connection"])
def test_provider_failure_is_exactly_one_request_and_stops_the_run(data, fake_index, transport, failure):
    fake = transport(failure)
    clock = FakeClock()
    report, cassette = run.run_live(only=SIX, confirm=True, index_source=fake_index,       # production model
                                    check_index=lambda path: None, clock=clock, sleep=clock.sleep)
    info = report["live"]
    assert len(fake.requests) == 1 and fake.requests[0].endswith("/chat/completions")
    assert info["provider_retries"] == {"chat_model": 0, "sdk_client": 0}
    assert info["model_calls"] == 1 and info["budget"]["used"] == 1
    assert info["stopped"]["kind"] == failure and info["stopped"]["case"] == SIX[0]
    assert info["completed_cases"] == [] and info["not_run"] == SIX[1:]
    assert report["run"]["model"]["id"] == "openai/gpt-oss-20b"
    (turn,) = info["stopped"]["recorded"]                     # the failed call is recorded...
    assert turn["model"] == [{"step": 0, "input_sha256": turn["model"][0]["input_sha256"], "error": failure}]
    assert cassette["cases"] == {} and report["cases"] == []   # ...but never replayed or scored
    if failure == "rate_limited":
        assert info["stopped"]["retry_after"] == 7


def test_the_application_model_still_retries_once(data, transport):
    # Contrast: the same failure through the app's normal model is retried by the SDK (2 requests).
    fake = transport("rate_limited")
    from langchain_core.messages import HumanMessage
    with pytest.raises(groq.RateLimitError):
        llm.get_assistant_model().invoke([HumanMessage(content="hi")])
    assert len(fake.requests) == 2


def test_saved_report_of_a_stopped_run_records_the_failure(data, fake_index, transport, tmp_path):
    transport("rate_limited")
    clock = FakeClock()
    report, cassette = run.run_live(only=SIX[:2], confirm=True, index_source=fake_index,
                                    check_index=lambda path: None, clock=clock, sleep=clock.sleep)
    report_path, cassette_path = run.save_live(report, cassette, "stopped", out_dir=tmp_path)
    saved = json.loads(report_path.read_text())
    assert saved["live"]["stopped"]["recorded"][0]["model"][0]["error"] == "rate_limited"
    with pytest.raises(CassetteError, match="no completed cases"):
        run.run_replay(cassette_path)


class FakeEmbeddings:
    """Deterministic unit vectors (no model download)."""

    def _vector(self, text):
        import hashlib
        import math

        digest = hashlib.sha256(text.encode()).digest()
        values = [b / 255.0 + 0.01 for b in digest[:16]]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def temp_chromadb(tmp_path, monkeypatch):
    """A real, valid policy index in a temp folder, built from the policy files with fake embeddings."""

    vs.reset_index_state()
    folder = tmp_path / "chroma-source"
    vs.build_index(folder, chunks=vs.expected_chunks(), embeddings=FakeEmbeddings(), embedding_id=vs.EMBEDDING_ID)
    monkeypatch.setattr(vs, "get_embedding_model", lambda: FakeEmbeddings())
    vs.reset_index_state()
    yield folder
    vs.reset_index_state()


def test_live_policy_search_runs_on_an_isolated_copy_of_the_index(data, no_groq, temp_chromadb, tmp_path,
                                                                  monkeypatch):
    production_before = tree_sha256(vs.CHROMA_DIR)
    source_before = tree_sha256(temp_chromadb)
    original_dir = vs.CHROMA_DIR
    searched_in = []
    real_search = tools.search_relevant_documents

    def spy(*args, **kwargs):
        searched_in.append(vs.CHROMA_DIR)
        return real_search(*args, **kwargs)

    monkeypatch.setattr(tools, "search_relevant_documents", spy)
    ids = ["policy_refund_approved"]
    cases = cases_for(ids)
    clock = FakeClock()
    report, cassette = run.run_live(only=ids, confirm=True, model=FakeGroq(cases), index_source=temp_chromadb,
                                    clock=clock, sleep=clock.sleep)          # real policy tool, real index check
    (copy_dir,) = searched_in
    assert copy_dir.name == "chroma_db" and copy_dir != temp_chromadb and copy_dir != original_dir
    assert not copy_dir.exists()                                             # temp copy removed
    assert vs.CHROMA_DIR == original_dir
    assert report["live"]["integrity"]["index_unchanged"] is True
    assert tree_sha256(temp_chromadb) == source_before
    assert tree_sha256(vs.CHROMA_DIR) == production_before
    (entry,) = cassette["cases"]["policy_refund_approved"]["turns"][0]["policy"]
    assert json.loads(entry["output"])["status"] in ("SUCCESS", "NO_RELEVANT_POLICY_FOUND")

    _, _, replayed = save_and_replay(report, cassette, tmp_path)
    assert comparable(replayed["metrics"]) == comparable(report["metrics"])
    assert searched_in == [copy_dir]                                         # replay never searched ChromaDB
