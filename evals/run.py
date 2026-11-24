"""
Run the assistant evaluation and save a structured JSON report.

    python -m evals.run                          # offline: scripted model, no Groq, no ChromaDB
    python -m evals.run --case metrics_october_c102 --case policy_refund_approved
    python -m evals.run --out evals/results/my-run.json

    # live (real Groq model): refused without --confirm-live; budgets and pacing apply
    python -m evals.run --mode live --confirm-live --baseline NAME --case ID [--case ID ...]

    # replay a live recording offline through the real loop (never calls Groq)
    python -m evals.run --mode replay --cassette evals/baselines/NAME.cassette.json

Offline mode drives the real assistant loop (authorization, argument
validation, real customer tools on the active dataset, grounding,
citation selection, answer sanitising) with a scripted model and an
offline policy search. It measures the pipeline and the scorers, not the
LLM's own judgement.

Live mode uses the production model (built by app.llm.get_assistant_model:
the configured Groq model bound to the production tools, with the SDK's
automatic retries disabled so every failure is a single attempt) and the
production tool map, with the policy search running on a temporary copy of
chroma_db/. Every model response and policy-search output is recorded in
a cassette next to the report (evals/baselines/), so the run can be
replayed and re-scored offline. Replay mode feeds a cassette back
through the same loop and stops with a clear error if the recording
does not match.
"""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evals import scorers
from evals.cases import CASES_FILE, file_sha256, load_cases
from evals.cassette import (CassetteError, CassetteMismatch, CassetteModel, CassettePolicyTool, CassetteWriter,
                            Replay, ReplayModel, ReplayPolicySearch, load_cassette)
from evals.expected import FigureError, describe, fact, figure
from evals.live import (GuardedModel, LiveRunRefused, ModelCallBudget, StopRun, TokenPacer, check_index_copy,
                        classify, classify_kind, isolated_policy_index, safe_json, secrets_in_environment,
                        tree_sha256)
from evals.recorder import (POLICY_TOOL, Recorder, RecordingModel, ScriptedModel, ScriptedPolicySearch,
                            ScriptError, count_revisions, recording_tools)
from evals.scorers import blocked_results, numbers_in, score_turn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
BASELINES_DIR = PROJECT_ROOT / "evals" / "baselines"
SCHEMA_VERSION = 1
# Offline runs never need a credential; this placeholder satisfies app.llm's
# import-time check and keeps a real key from .env out of the process.
OFFLINE_KEY = "offline-evaluation-no-key"
# Evaluation code whose content is part of the run fingerprint: a run made
# with different scoring, recording, case loading, figures or runner code
# can never look equivalent to another run.
EVALUATOR_FILES = tuple(Path(__file__).resolve().parent / name
                        for name in ("recorder.py", "scorers.py", "cases.py", "expected.py", "run.py",
                                     "cassette.py", "live.py", "answers.py"))
# Execution components: what the model saw and how the loop ran. A replay refuses any change to the
# strict ones (they would make the recorded responses meaningless) and notes changes to the others
# (they only affect a future live run: replay uses the recorded policy outputs).
EXECUTION_STRICT = ("system_prompt_sha256", "tool_schemas_sha256", "dataset_sha256", "max_tool_steps")
# Answer processing (grounding repair, sanitising, citations, data-scope notes) runs after the model's
# last call of a turn: a change alters the answers, not the recorded model inputs (those are checked
# call by call), so replay notes it and re-runs the current code.
EXECUTION_NOTED = ("policy_index_fingerprint", "retrieval", "answer_processing_sha256")
ANSWER_PROCESSING_FILES = tuple(PROJECT_ROOT / "app" / name for name in ("answer_scope.py", "citations.py",
                                                                         "grounding.py"))
ANSWER_PROCESSING_FUNCTIONS = (("app.assistant", "_answer"), ("app.assistant", "_scope_results"),
                               ("app.assistant", "_enforce_grounding"), ("app.assistant", "_ensure_markers"),
                               ("app.security", "sanitize_answer"))
SESSION_STORES = ("conversation_sessions", "grounding_context", "session_last_active",
                  "conversation_versions", "request_results", "requests_in_flight")


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def policy_chunks():
    """The policy chunks the index is built from (reads the policy files; no embeddings, no ChromaDB)."""

    from app.vector_store import expected_chunks

    with contextlib.redirect_stdout(io.StringIO()):
        return expected_chunks()


def _forget_session(session_id):
    from app import assistant

    for name in SESSION_STORES:
        store = getattr(assistant, name, None)
        if isinstance(store, dict):
            store.pop(session_id, None)


# ---------- Drivers: where model responses and policy results come from ----------

class OfflineDriver:
    """Scripted model + offline policy search (no Groq, no ChromaDB)."""

    mode = "offline"

    def __init__(self, chunks):
        self.scripted = ScriptedModel()
        self.policy = ScriptedPolicySearch(chunks)

    def model(self):
        return self.scripted

    def tool_map(self, base):
        return {**base, POLICY_TOOL: self.policy}

    def start_turn(self, case, index, turn):
        self.scripted.start_turn(turn["offline"]["script"])
        self.policy.selection = turn["offline"].get("policy", [])

    def end_turn(self, case, index):
        pass

    def abort_case(self, case):
        pass


class LiveDriver:
    """The given (guarded) live model and the production tools, recording into a cassette."""

    mode = "live"

    def __init__(self, model, writer):
        self.writer = writer
        self._model = CassetteModel(model, writer, classify_kind)

    def model(self):
        return self._model

    def tool_map(self, base):
        return {**base, POLICY_TOOL: CassettePolicyTool(base[POLICY_TOOL], self.writer)}

    def start_turn(self, case, index, turn):
        self.writer.begin_turn(case["id"], index)

    def end_turn(self, case, index):
        pass

    def abort_case(self, case):
        self.writer.drop_case(case["id"])


class ReplayDriver:
    """Recorded responses and policy outputs only; real customer tools on the active dataset."""

    mode = "replay"

    def __init__(self, replay):
        self.replay = replay
        self._model = ReplayModel(replay)
        self.policy = ReplayPolicySearch(replay)

    def model(self):
        return self._model

    def tool_map(self, base):
        return {**base, POLICY_TOOL: self.policy}

    def start_turn(self, case, index, turn):
        self._model.tape = self.policy.tape = self.replay.tape(case["id"], index)

    def end_turn(self, case, index):
        self._model.tape.finish()

    def abort_case(self, case):
        pass


def run_case(case, chunks=None, driver=None):
    """Run every turn of one case in its own conversation; return the scored record."""

    from app import assistant
    from app.errors import AssistantError

    driver = driver or OfflineDriver(chunks)
    recorder = Recorder()
    model = RecordingModel(driver.model(), recorder)
    tools = recording_tools(driver.tool_map(assistant.tool_map), recorder, fail=case.get("fail_tools", ()))
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
    supported = set()
    turns = []

    try:
        for index, turn in enumerate(case["turns"]):
            recorder.reset()
            driver.start_turn(case, index, turn)
            expect = turn.get("expect", {})
            result, error_kind = None, None
            started = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):   # the loop prints tool names
                    result = assistant.ask_bankmate_with_sources(
                        turn["question"], session_id=session_id, model=model, tools=tools)
            except AssistantError as error:
                error_kind = error.kind
            except ScriptError as error:
                error_kind = "script_error"
                logging.getLogger(__name__).error("Case %s: %s", case["id"], error)
            elapsed_ms = (time.perf_counter() - started) * 1000
            driver.end_turn(case, index)
            # Live pacing waits happen inside the turn; they are reported separately, never as latency.
            paced_ms = round(sum(c.get("paced_ms", 0) or 0 for c in recorder.model_calls), 3)
            latency_ms = round(max(elapsed_ms - paced_ms, 0.0), 3)

            executed = recorder.tool_executions
            for entry in executed:
                if entry.get("status") == "ok":
                    supported |= numbers_in(entry["output"])
            answer = result["answer"] if result else None
            observed = {
                "answer": answer,
                "sources": [{"policy_id": s.get("policy_id"), "policy_name": s.get("policy_name")}
                            for s in (result or {}).get("sources", [])],
                "requested_tools": [c for call in recorder.model_calls for c in call["tool_calls"]],
                "executed_tools": executed,
                "blocked": list(blocked_results(recorder.last_model_input).values()),
                "model_calls": recorder.model_calls,
                "revisions": count_revisions(recorder.model_calls),
                "tool_limit": answer == assistant.TOOL_LIMIT_MESSAGE,
                "error_kind": error_kind,
                "latency_ms": latency_ms,
                "provider_latency_ms": round(sum(c.get("latency_ms") or 0 for c in recorder.model_calls), 3),
                "paced_ms": paced_ms,
            }
            figure_values, facts, expectation_errors = [], [], {}
            try:
                figure_values = [{"label": describe(spec), "value": figure(spec), "spec": spec,
                                  "required": spec.get("required", True)} for spec in expect.get("figures", [])]
            except FigureError as error:
                expectation_errors["figures"] = str(error)
            try:
                facts = [fact(spec) for spec in expect.get("facts", [])]
            except FigureError as error:
                expectation_errors["facts"] = str(error)
            score = score_turn(expect, observed, supported, figure_values, turn["question"], facts)
            for check, message in expectation_errors.items():
                score["checks"][check] = False
                score["detail"][f"{check}_error"] = message
                score["passed"] = False
            turns.append({"question": turn["question"], "expect": expect,
                          "observed": _public(observed), "score": score})
    except BaseException:
        driver.abort_case(case)       # a stopped live run or a broken replay: the case is not scored
        raise
    finally:
        _forget_session(session_id)

    return {"id": case["id"], "category": case["category"], "description": case.get("description", ""),
            "passed": all(t["score"]["passed"] for t in turns), "turns": turns}


def _public(observed):
    """What goes in the report: tool outputs are summarised (size), not copied."""

    out = dict(observed)
    out["executed_tools"] = [
        {"name": e["name"], "args": e.get("args"), "status": e.get("status"), "error": e.get("error"),
         "latency_ms": e.get("latency_ms"), "output_chars": len(str(e.get("output", "")))}
        for e in observed["executed_tools"]]
    out["blocked"] = [{"tool": name, "status": status} for name, _args, status in observed["blocked"]]
    return out


# ---------- Run metadata and fingerprint ----------

def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def evaluator_code_hashes(files=None):
    """{file name: SHA-256 of its bytes} for the evaluation implementation files."""

    return {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (EVALUATOR_FILES if files is None else files)}


def answer_processing_hashes(files=None, functions=None):
    """
    {name: SHA-256} for the code that turns the model's final message into the
    answer: whole modules (data-scope notes, citations, grounding) and the
    source of the assistant/security functions that apply them.
    """

    import importlib
    import inspect

    hashes = {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for path in (ANSWER_PROCESSING_FILES if files is None else files)}
    for module, name in (ANSWER_PROCESSING_FUNCTIONS if functions is None else functions):
        source = inspect.getsource(getattr(importlib.import_module(module), name))
        hashes[f"{module}.{name}"] = _sha256(source)
    return hashes


def run_metadata(mode, cases_path, cases, chunks, model_id):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from app import assistant, data_loader, llm, retriever, vector_store

    dataset = data_loader.get_dataset()
    manifest = Path(dataset.folder) / "dataset.json"
    try:
        cases_display = str(Path(cases_path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        cases_display = Path(cases_path).name
    components = {
        "system_prompt_sha256": _sha256(assistant.system_message.content),
        "tool_schemas_sha256": _sha256(json.dumps([convert_to_openai_tool(t) for t in llm.tools],
                                                  sort_keys=True)),
        "model": model_id,
        "dataset_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
        "policy_index_fingerprint": vector_store.fingerprint(chunks),
        "retrieval": {"k": retriever.POLICY_SEARCH_K, "max_distance": retriever.POLICY_MAX_DISTANCE},
        "max_tool_steps": assistant.MAX_TOOL_STEPS,
        "answer_processing_sha256": answer_processing_hashes(),
        "cases_sha256": file_sha256(cases_path),
        "case_inputs_sha256": _sha256(json.dumps({c["id"]: case_input_sha256(c) for c in cases}, sort_keys=True)),
        "case_expectations_sha256": _sha256(json.dumps({c["id"]: case_expect_sha256(c) for c in cases},
                                                       sort_keys=True)),
        "evaluator_schema": SCHEMA_VERSION,
        "evaluator_code_sha256": evaluator_code_hashes(),
    }
    commit = _git("rev-parse", "HEAD")
    # Immutable side: everything that shapes what the model is asked and how the loop runs.
    execution = {k: components[k] for k in EXECUTION_STRICT + EXECUTION_NOTED + ("case_inputs_sha256",)}
    # Mutable side: how the run is judged. Changing it allows re-scoring a recorded run.
    evaluation = {k: components[k] for k in ("case_expectations_sha256", "evaluator_schema",
                                             "evaluator_code_sha256")}
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"commit": commit, "dirty": bool(_git("status", "--porcelain")) if commit else None},
        "model": {"id": model_id, "live": mode == "live"},
        "dataset": {"format": dataset.format, "folder": Path(dataset.folder).name,
                    "as_of_date": dataset.meta.get("as_of_date"), "row_counts": dataset.meta.get("row_counts"),
                    "sha256": components["dataset_sha256"]},
        "cases": {"file": cases_display, "sha256": components["cases_sha256"], "count": len(cases)},
        "components": components,
        "fingerprint": _sha256(json.dumps({**components, "git_commit": commit}, sort_keys=True)),
        "execution_fingerprint": _sha256(json.dumps(execution, sort_keys=True)),
        "evaluation_fingerprint": _sha256(json.dumps(evaluation, sort_keys=True)),
        "token_usage": {"offline": "none (scripted model)", "live": "reported by the provider",
                        "replay": "recorded in the cassette; none consumed by the replay"}.get(mode),
        "environment": {"python": platform.python_version(), "platform": platform.system().lower()},
    }


def run_offline(cases_path=CASES_FILE, only=None):
    """Run the offline suite; return the full report (dict)."""

    cases = load_cases(cases_path, only)
    chunks = policy_chunks()
    results = [run_case(case, chunks) for case in cases]
    return {
        "run": run_metadata("offline", cases_path, cases, chunks, "scripted-offline"),
        "metrics": scorers.aggregate(results, "offline"),
        "by_category": scorers.by_category(results),
        "cases": results,
    }


def case_sha256(case):
    """Content hash of the whole case definition."""

    return _sha256(json.dumps(case, sort_keys=True, ensure_ascii=False))


def case_input_sha256(case):
    """What the model is given: the questions, in order, and any injected tool failures. Never changes on replay."""

    return _sha256(json.dumps({"id": case["id"], "questions": [t["question"] for t in case["turns"]],
                               "fail_tools": sorted(case.get("fail_tools", []))}, sort_keys=True, ensure_ascii=False))


def case_expect_sha256(case):
    """How the case is judged: its expectations. May change; a recorded run is then re-scored."""

    return _sha256(json.dumps({"category": case.get("category"), "expect": [t.get("expect", {}) for t in case["turns"]]},
                              sort_keys=True, ensure_ascii=False))


def _worst_case_calls(case):
    from app import assistant

    return len(case["turns"]) * (assistant.MAX_TOOL_STEPS + 1)      # tool rounds + one grounding revision


# ---------- Live ----------

LIVE_DEFAULTS = {"max_cases": 6, "max_model_calls": 30, "tokens_per_minute": 6000,
                 "requests_per_minute": 20, "case_pause_seconds": 5.0}


def production_model():
    """
    The app's own model, built by the app's own factory (app.llm.get_assistant_model:
    same Groq model, temperature, timeout and tool binding) but with the Groq SDK's
    automatic retries disabled, so a failed request is attempted exactly once and the
    run stops on it. The app's cached model (assistant._model) and its retry setting
    are not touched; the override exists only while this fresh instance is built.
    """

    from app import llm

    saved = llm.LLM_MAX_RETRIES
    llm.LLM_MAX_RETRIES = 0
    try:
        model = llm.get_assistant_model()
    finally:
        llm.LLM_MAX_RETRIES = saved
    retries = sdk_retries_of(model)
    if retries != {"chat_model": 0, "sdk_client": 0}:
        raise LiveRunRefused(f"could not disable the Groq client's automatic retries ({retries})")
    return model


def sdk_retries_of(model):
    """Retry settings of a ChatGroq binding: the LangChain model's and the underlying Groq SDK client's."""

    inner = getattr(model, "bound", model)
    sdk = getattr(getattr(inner, "client", None), "_client", None)
    return {"chat_model": getattr(inner, "max_retries", None), "sdk_client": getattr(sdk, "max_retries", None)}


def model_id_of(model):
    inner = getattr(model, "bound", model)
    return getattr(inner, "model_name", None) or getattr(inner, "model", None) or inner.__class__.__name__


def _tool_schema_chars():
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from app import llm

    return len(json.dumps([convert_to_openai_tool(t) for t in llm.tools]))


def _dataset_folder():
    from app import data_loader

    return Path(data_loader.get_dataset().folder)


def run_live(cases_path=CASES_FILE, only=None, *, confirm=False, model=None, policy_tool=None,
             index_source=None, check_index=check_index_copy, clock=time.monotonic, sleep=time.sleep,
             notify=None, **settings):
    """
    Run the selected cases against the live model. Returns (report, cassette).
    Raises LiveRunRefused before any model call when the run is not allowed.
    `model`, `policy_tool`, `index_source`, `check_index`, `clock` and
    `sleep` exist for tests; by default the production model, the
    production policy tool and chroma_db/ are used.
    """

    config = {**LIVE_DEFAULTS, **{k: v for k, v in settings.items() if v is not None}}
    unknown = set(config) - set(LIVE_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown live settings: {sorted(unknown)}")
    notify = notify or (lambda message: None)

    if not confirm:
        raise LiveRunRefused("live runs call the Groq API and need explicit confirmation (--confirm-live)")
    if not only:
        raise LiveRunRefused("name the cases to run with --case (a live run never selects the whole suite)")
    cases = load_cases(cases_path, only)
    if len(cases) > config["max_cases"]:
        raise LiveRunRefused(f"{len(cases)} cases selected but --max-cases is {config['max_cases']}")
    if _worst_case_calls(cases[0]) > config["max_model_calls"]:
        raise LiveRunRefused(f"--max-model-calls {config['max_model_calls']} cannot cover even one case "
                             f"(worst case {_worst_case_calls(cases[0])} calls)")
    try:
        from app import assistant  # noqa: F401  (validates GROQ_API_KEY without revealing it)
    except ValueError as error:
        raise LiveRunRefused(str(error)) from None

    chunks = policy_chunks()
    dataset_folder = _dataset_folder()
    from app import vector_store
    index_source = Path(index_source or vector_store.CHROMA_DIR)
    before = {"dataset": tree_sha256(dataset_folder), "index": tree_sha256(index_source)}

    writer = CassetteWriter()
    budget = ModelCallBudget(config["max_model_calls"])

    def paced_sleep(seconds):
        notify(f"  pacing: waiting {seconds:.0f}s for the rate-limit window")
        sleep(seconds)

    pacer = TokenPacer(config["tokens_per_minute"], config["requests_per_minute"], clock=clock, sleep=paced_sleep)
    results, stopped, skipped = [], None, []

    with isolated_policy_index(index_source) as index_copy:
        check_index(index_copy)                               # before any model call
        live_model = model if model is not None else production_model()
        model_id = model_id_of(live_model)
        guarded = GuardedModel(live_model, budget, pacer, classify, overhead_chars=_tool_schema_chars())
        driver = LiveDriver(guarded, writer)
        if policy_tool is not None:
            driver.tool_map = lambda base: {**base, POLICY_TOOL: CassettePolicyTool(policy_tool, writer)}

        for number, case in enumerate(cases, 1):
            if stopped:
                skipped.append(case["id"])
                continue
            if budget.remaining < _worst_case_calls(case):
                stopped = {"reason": f"remaining model-call budget ({budget.remaining}) is below this case's "
                                     f"worst case ({_worst_case_calls(case)})", "kind": "budget",
                           "case": case["id"], "retry_after": None}
                skipped.append(case["id"])
                continue
            if number > 1:
                pacer.pause(config["case_pause_seconds"])
            notify(f"[{number}/{len(cases)}] {case['id']}")
            try:
                record = run_case(case, driver=driver)
            except StopRun as stop:
                stopped = {"reason": stop.reason, "kind": stop.kind, "case": case["id"],
                           "retry_after": stop.retry_after,
                           # What was recorded for the stopped case, ending with the failed call.
                           "recorded": writer.aborted.get(case["id"], {"turns": []})["turns"]}
                notify(f"  STOPPED: {stop.reason}")
                continue
            results.append(record)
            calls = sum(len(t["observed"]["model_calls"]) for t in record["turns"])
            notify(f"  {'pass' if record['passed'] else 'FAIL'}  ({calls} model calls, budget left {budget.remaining})")

    # The production index is untouched by construction; prove it.
    after = {"dataset": tree_sha256(dataset_folder), "index": tree_sha256(index_source)}
    completed = [c for c in cases if c["id"] in {r["id"] for r in results}]
    run = run_metadata("live", cases_path, completed, chunks, model_id)
    run["model"]["provider"] = "groq" if model is None else "injected"
    report = {
        "run": run,
        "live": {
            "settings": config,
            "model_calls": guarded.calls,
            "budget": {"max": budget.max_calls, "used": budget.used},
            "pacing": {"waits": pacer.waits, "waited_seconds": round(pacer.waited_seconds, 1)},
            "selected_cases": [c["id"] for c in cases],
            "completed_cases": [r["id"] for r in results],
            "stopped": stopped,
            "not_run": skipped,
            "integrity": {"dataset_unchanged": before["dataset"] == after["dataset"],
                          "index_unchanged": before["index"] == after["index"],
                          "index_isolated": True},
            "provider_retries": sdk_retries_of(live_model) if model is None else "injected model",
            "timeout_seconds": _llm_timeout(),
        },
        "metrics": scorers.aggregate(results, "live"),
        "by_category": scorers.by_category(results),
        "cases": results,
    }
    cassette = writer.to_dict({
        "created_utc": run["created_utc"], "fingerprint": run["fingerprint"], "model": model_id,
        "git_commit": run["git"]["commit"], "dataset_sha256": run["components"]["dataset_sha256"],
        "cases_file": run["cases"]["file"],
        "case_sha256": {c["id"]: case_sha256(c) for c in completed},
        "case_input_sha256": {c["id"]: case_input_sha256(c) for c in completed},
        "case_expect_sha256": {c["id"]: case_expect_sha256(c) for c in completed},
        "execution": {k: run["components"][k] for k in EXECUTION_STRICT + EXECUTION_NOTED},
    })
    return report, cassette


def _llm_timeout():
    from app import llm

    return llm.LLM_TIMEOUT_SECONDS


def save_live(report, cassette, name=None, out_dir=BASELINES_DIR, secrets=None):
    """
    Write the report and its cassette, redacted and checked: the configured
    GROQ_API_KEY (and any `secrets` given) can never reach either file.
    Returns (report_path, cassette_path).
    """

    secrets = list(secrets or []) + secrets_in_environment()
    run = report["run"]
    if name is None:
        stamp = run["created_utc"].replace("-", "").replace(":", "")
        name = f"live-{stamp}-{(run['git']['commit'] or 'nogit')[:7]}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path, cassette_path = out_dir / f"{name}.json", out_dir / f"{name}.cassette.json"
    cassette_text = safe_json(cassette, secrets)
    report["live"]["cassette"] = {"file": cassette_path.name,
                                  "sha256": hashlib.sha256(cassette_text.encode("utf-8")).hexdigest()}
    report_text = safe_json(report, secrets)
    cassette_path.write_text(cassette_text, encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    return report_path, cassette_path


# ---------- Replay ----------

@contextlib.contextmanager
def no_live_model():
    """During replay, any attempt to build the production model fails loudly."""

    from app import assistant, llm

    def forbidden(*args, **kwargs):
        raise CassetteError("replay attempted to create the live model; replay never calls Groq")

    saved = (assistant._get_model, llm.get_assistant_model)
    assistant._get_model = llm.get_assistant_model = forbidden
    try:
        yield
    finally:
        assistant._get_model, llm.get_assistant_model = saved


def _execution_reference(cassette_path, recorded):
    """
    The recorded execution components and per-case inputs, with their source:
    the cassette itself, or, for a cassette written before Evaluator V2, the
    baseline report saved next to it (its fingerprint must match the cassette's).
    """

    if recorded.get("execution"):
        return {"source": "cassette", "components": recorded["execution"],
                "inputs": recorded.get("case_input_sha256", {}), "expect": recorded.get("case_expect_sha256", {}),
                "questions": None}
    path = Path(cassette_path)
    sibling = path.with_name(path.name.replace(".cassette.json", ".json"))
    if sibling == path or not sibling.exists():
        raise CassetteMismatch("this cassette predates execution fingerprints and has no baseline report next to "
                               "it, so its execution inputs cannot be verified")
    report = json.loads(sibling.read_text(encoding="utf-8"))
    if report.get("run", {}).get("fingerprint") != recorded.get("fingerprint"):
        raise CassetteMismatch(f"{sibling.name} is not the report this cassette was recorded with "
                               "(fingerprints differ)")
    components = report["run"]["components"]
    return {"source": f"baseline report {sibling.name}",
            "components": {k: components.get(k) for k in EXECUTION_STRICT + EXECUTION_NOTED},
            "inputs": None, "expect": None,
            "questions": {c["id"]: [t["question"] for t in c["turns"]] for c in report.get("cases", [])}}


def verify_replay_inputs(cassette_path, recorded, cases, components, strict_inputs=True):
    """
    Check everything that shaped the recorded run before replaying it. Raises
    CassetteMismatch for a changed question, injected failure, system prompt,
    tool schema, dataset or tool-loop limit (unless strict_inputs is False, in
    which case they are reported). Expectation changes are allowed and listed:
    the recorded run is re-scored against the current expectations.
    """

    reference = _execution_reference(cassette_path, recorded)
    problems, notes, rescored = [], [], []
    if recorded.get("dataset_sha256") not in (None, components["dataset_sha256"]):
        problems.append("the active dataset differs from the one the cassette was recorded on")
    for key in EXECUTION_STRICT:
        if key in reference["components"] and reference["components"][key] != components[key]:
            problems.append(f"{key} changed since the recording")
    for key in EXECUTION_NOTED:
        if key in reference["components"] and reference["components"][key] != components[key]:
            reason = ("replay re-runs the current answer processing on the recorded responses"
                      if key == "answer_processing_sha256" else "replay uses the recorded policy outputs")
            notes.append(f"{key} changed since the recording ({reason}; model inputs are still verified per call)")
    for case in cases:
        case_id = case["id"]
        if reference["inputs"] is not None:
            if reference["inputs"].get(case_id) != case_input_sha256(case):
                problems.append(f"case {case_id}: questions or injected tool failures changed since it was recorded")
            elif reference["expect"].get(case_id) != case_expect_sha256(case):
                rescored.append(case_id)
        else:
            if reference["questions"].get(case_id) != [t["question"] for t in case["turns"]]:
                problems.append(f"case {case_id}: questions changed since it was recorded")
            elif recorded.get("case_sha256", {}).get(case_id) != case_sha256(case):
                rescored.append(case_id)     # definition changed, questions verified; tool failures via model inputs
    if problems:
        if strict_inputs:
            raise CassetteMismatch("; ".join(problems))
        notes.extend(f"ALLOWED DRIFT: {p}" for p in problems)
    return {"execution_source": reference["source"], "verified": list(EXECUTION_STRICT) + ["case questions"],
            "notes": notes, "rescored_cases": rescored}


def run_replay(cassette_path, cases_path=CASES_FILE, only=None, strict_inputs=True):
    """
    Replay a live cassette through the real loop and score it against the
    current expectations; return the report. Raises CassetteError on any
    mismatch in the recorded execution. Never calls Groq.
    """

    cassette = load_cassette(cassette_path)
    recorded = cassette.get("run", {})
    wanted = [case_id for case_id in cassette["cases"] if not only or case_id in only]
    missing = [case_id for case_id in (only or []) if case_id not in cassette["cases"]]
    if missing:
        raise CassetteError(f"not in the cassette: {', '.join(missing)}")
    if not wanted:
        raise CassetteError("the cassette has no completed cases to replay")
    cases = load_cases(cases_path, wanted)
    chunks = policy_chunks()
    run = run_metadata("replay", cases_path, cases, chunks, recorded.get("model"))
    verification = verify_replay_inputs(cassette_path, recorded, cases, run["components"], strict_inputs)
    replay = Replay(cassette, strict_inputs=strict_inputs)
    with no_live_model():
        results = [run_case(case, driver=ReplayDriver(replay)) for case in cases]
    run["replay_of"] = {"fingerprint": recorded.get("fingerprint"), "created_utc": recorded.get("created_utc"),
                        "git_commit": recorded.get("git_commit"),
                        "cassette_sha256": file_sha256(cassette_path),
                        "strict_inputs": strict_inputs, "input_drift": replay.input_drift, **verification}
    return {"run": run, "metrics": scorers.aggregate(results, "replay"), "by_category": scorers.by_category(results),
            "cases": results}


def write_report(report, out=None):
    if out is None:
        run = report["run"]
        stamp = run["created_utc"].replace("-", "").replace(":", "")
        commit = (run["git"]["commit"] or "nogit")[:7]
        out = RESULTS_DIR / f"{stamp}-{commit}-{run['mode']}.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return out


def print_summary(report, path):
    m, run = report["metrics"], report["run"]
    print(f"\nBANKMATE ASSISTANT EVALUATION  mode={run['mode']}  model={run['model']['id']}  "
          f"cases={m['cases']}  turns={m['turns']}")
    print(f"fingerprint {run['fingerprint'][:16]}  commit {(run['git']['commit'] or '-')[:7]}"
          f"{' (dirty)' if run['git']['dirty'] else ''}")
    print("=" * 78)
    for key, value in m.items():
        if key not in ("cases", "turns"):
            print(f"  {key:<26} {value if value is not None else '-'}")
    print("\nBy category")
    for category, bucket in report["by_category"].items():
        print(f"  {category:<15} {bucket['passed']}/{bucket['cases']}")
    failures = [(c["id"], n, t) for c in report["cases"] for n, t in enumerate(c["turns"], 1)
                if not t["score"]["passed"]]
    print("\nFailures" if failures else "\nAll cases passed.")
    for case_id, number, turn in failures:
        failed = [k for k, v in turn["score"]["checks"].items() if v is False]
        print(f"  {case_id} turn {number}: {', '.join(failed)}")
    print(f"\nReport: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the BankMate assistant evaluation.")
    parser.add_argument("--mode", choices=("offline", "live", "replay"), default="offline")
    parser.add_argument("--cases", default=str(CASES_FILE), help="Cases file (JSONL)")
    parser.add_argument("--case", action="append", dest="only", help="Run only this case id (repeatable)")
    parser.add_argument("--out", help="Report path (offline/replay; default: evals/results/<utc>-<commit>-<mode>.json)")
    parser.add_argument("--verbose", action="store_true", help="Show the assistant's warning logs")
    live = parser.add_argument_group("live mode (calls the Groq API)")
    live.add_argument("--confirm-live", action="store_true", help="Required: allow real Groq API calls")
    live.add_argument("--baseline", help="Name of the saved baseline in evals/baselines/ (default live-<utc>-<commit>)")
    live.add_argument("--max-cases", type=int, help=f"Refuse more cases than this (default {LIVE_DEFAULTS['max_cases']})")
    live.add_argument("--max-model-calls", type=int,
                      help=f"Hard cap on model calls (default {LIVE_DEFAULTS['max_model_calls']})")
    live.add_argument("--tpm", type=int, dest="tokens_per_minute",
                      help=f"Token pacing limit per minute (default {LIVE_DEFAULTS['tokens_per_minute']})")
    live.add_argument("--rpm", type=int, dest="requests_per_minute",
                      help=f"Request pacing limit per minute (default {LIVE_DEFAULTS['requests_per_minute']})")
    live.add_argument("--case-pause", type=float, dest="case_pause_seconds",
                      help=f"Pause between cases in seconds (default {LIVE_DEFAULTS['case_pause_seconds']})")
    replay = parser.add_argument_group("replay mode (offline)")
    replay.add_argument("--cassette", help="Cassette recorded by a live run")
    replay.add_argument("--allow-input-drift", action="store_true",
                        help="Report (instead of refusing) model inputs that differ from the recording")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        logging.getLogger("app").setLevel(logging.ERROR)   # expected refusals log warnings

    if args.mode == "live":
        # The real key comes from the environment or .env (loaded by app.llm); it is never written anywhere.
        if os.getenv("GROQ_API_KEY") == OFFLINE_KEY:
            del os.environ["GROQ_API_KEY"]
        try:
            report, cassette = run_live(
                args.cases, args.only, confirm=args.confirm_live, notify=lambda m: print(m, flush=True),
                max_cases=args.max_cases, max_model_calls=args.max_model_calls,
                tokens_per_minute=args.tokens_per_minute, requests_per_minute=args.requests_per_minute,
                case_pause_seconds=args.case_pause_seconds)
        except LiveRunRefused as refused:
            print(f"Live run refused (no API call was made): {refused}", file=sys.stderr)
            return 2
        path, cassette_path = save_live(report, cassette, args.baseline)
        print_summary(report, path)
        live_info = report["live"]
        print(f"Cassette: {cassette_path}")
        print(f"Model calls {live_info['model_calls']} (budget {live_info['budget']['max']}), pacing waited "
              f"{live_info['pacing']['waited_seconds']}s, dataset unchanged "
              f"{live_info['integrity']['dataset_unchanged']}, index unchanged {live_info['integrity']['index_unchanged']}")
        if live_info["stopped"]:
            print(f"STOPPED early: {live_info['stopped']['reason']} (case {live_info['stopped']['case']}); "
                  f"not run: {', '.join(live_info['not_run']) or 'none'}")
            return 3
        return 0 if report["metrics"]["case_pass_rate"] == 1.0 else 1

    os.environ["GROQ_API_KEY"] = OFFLINE_KEY
    if args.mode == "replay":
        if not args.cassette:
            parser.error("--mode replay needs --cassette")
        try:
            report = run_replay(args.cassette, args.cases, args.only, strict_inputs=not args.allow_input_drift)
        except CassetteError as error:
            print(f"Replay failed (no API call was made): {error}", file=sys.stderr)
            return 2
        info = report["run"]["replay_of"]
        print(f"Replay verified against the {info['execution_source']}: {', '.join(info['verified'])}; "
              f"re-scored against changed expectations: {', '.join(info['rescored_cases']) or 'none'}")
        for note in info["notes"]:
            print(f"note: {note}")
    else:
        report = run_offline(args.cases, args.only)
    path = write_report(report, args.out)
    print_summary(report, path)
    return 0 if report["metrics"]["case_pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
