"""
Cassettes: what a live run received from the outside world, kept so the
run can be replayed offline through the real assistant loop.

A live run records, per case / turn / step:
- every model response (content, tool calls with their IDs, token usage,
  a few response-metadata fields) or the classified provider error, plus
  a SHA-256 digest of the exact messages the model was given;
- every policy-search call (the normalised query and the tool's JSON
  output, or the error class).

Customer tools are not recorded: they are pure functions of the active
dataset, whose hash the cassette stores, so replay re-runs them for real.

Replay feeds the recorded responses back in order. It never falls back
to a live model: a missing, out-of-order or unused entry, or (by default)
a model input that differs from the recording, raises a CassetteError.
CassetteError derives from BaseException on purpose: the assistant turns
ordinary model exceptions into AssistantError and swallows failures of
the grounding revision call, which would hide a broken replay.

Nothing secret is stored: no API key, no request headers, no model
inputs (only their digest). The runner redacts and checks every file
before writing it.
"""

import hashlib
import json
from pathlib import Path

from langchain_core.messages import AIMessage

CASSETTE_KIND = "bankmate-eval-cassette"
CASSETTE_SCHEMA = 1
RESPONSE_METADATA_KEYS = ("model_name", "finish_reason", "system_fingerprint")


class CassetteError(BaseException):
    """A replay could not follow the recording. Never caught by the assistant."""


class CassetteMissing(CassetteError):
    """The replay asked for an entry the recording does not have."""


class CassetteOrderError(CassetteError):
    """Entries were requested in a different order, or some were never used."""


class CassetteMismatch(CassetteError):
    """The replay's inputs differ from what was recorded (prompt, data, tool output or query changed)."""


# ---------- Serialisation ----------

def _content(content):
    return content if isinstance(content, str) else json.dumps(content, sort_keys=True, default=str)


def messages_digest(messages):
    """SHA-256 over the role, content, tool calls and tool-call IDs of every message."""

    items = []
    for message in messages:
        items.append({
            "type": getattr(message, "type", message.__class__.__name__),
            "content": _content(getattr(message, "content", "")),
            "tool_calls": [{"name": c["name"], "args": c.get("args", {}), "id": c.get("id")}
                           for c in (getattr(message, "tool_calls", None) or [])],
            "tool_call_id": getattr(message, "tool_call_id", None),
        })
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _numbers_only(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        kept = {str(k): _numbers_only(v) for k, v in value.items()}
        return {k: v for k, v in kept.items() if v is not None} or None
    return None


def serialize_response(response):
    """An AIMessage as plain JSON: only what replay and the report need."""

    metadata = getattr(response, "response_metadata", None) or {}
    reasoning = (getattr(response, "additional_kwargs", None) or {}).get("reasoning_content") or ""
    return {
        "content": _content(getattr(response, "content", "")),
        "tool_calls": [{"name": c["name"], "args": c.get("args", {}), "id": c.get("id")}
                       for c in (getattr(response, "tool_calls", None) or [])],
        "usage_metadata": _numbers_only(getattr(response, "usage_metadata", None) or {}),
        "response_metadata": {k: metadata[k] for k in RESPONSE_METADATA_KEYS
                              if isinstance(metadata.get(k), (str, int, float))},
        "reasoning_chars": len(str(reasoning)),
    }


def deserialize_response(data):
    calls = [{"name": c["name"], "args": dict(c.get("args") or {}), "id": c.get("id"), "type": "tool_call"}
             for c in data.get("tool_calls", [])]
    message = AIMessage(content=data.get("content", ""), tool_calls=calls,
                        response_metadata=dict(data.get("response_metadata") or {}))
    usage = data.get("usage_metadata") or {}
    if {"input_tokens", "output_tokens", "total_tokens"} <= set(usage):
        message.usage_metadata = {k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens")}
    return message


class ReplayedProviderError(Exception):
    """Stands in for a recorded provider failure; classified exactly like the original."""

    STATUS = {"rate_limited": 429, "authentication": 401, "service_error": 500}

    def __init__(self, kind):
        super().__init__(f"replayed provider error: {kind}")
        self.kind = kind
        self.status_code = self.STATUS.get(kind)


def replay_exception(kind):
    """An exception app.errors.classify_provider_error maps back to `kind`."""

    if kind == "timeout":
        return TimeoutError("replayed timeout")
    if kind == "connection":
        import groq
        import httpx
        return groq.APIConnectionError(request=httpx.Request("POST", "https://replay.invalid"))
    return ReplayedProviderError(kind)


# ---------- Recording ----------

class CassetteWriter:
    """Collects entries for the case and turn the runner says are current."""

    def __init__(self):
        self.cases = {}
        self.aborted = {}          # entries of a case stopped mid-run (kept for the report, not replayable)
        self._turn = None
        self._where = None

    def begin_turn(self, case_id, turn_index):
        turns = self.cases.setdefault(case_id, {"turns": []})["turns"]
        while len(turns) <= turn_index:
            turns.append({"model": [], "policy": []})
        self._turn, self._where = turns[turn_index], (case_id, turn_index)

    def drop_case(self, case_id):
        if case_id in self.cases:
            self.aborted[case_id] = self.cases.pop(case_id)

    def _current(self):
        if self._turn is None:
            raise RuntimeError("CassetteWriter.begin_turn() must be called before recording")
        return self._turn

    def record_model(self, messages, response=None, error_kind=None):
        turn = self._current()
        entry = {"step": len(turn["model"]), "input_sha256": messages_digest(messages)}
        if error_kind:
            entry["error"] = error_kind
        else:
            entry["response"] = serialize_response(response)
        turn["model"].append(entry)

    def record_policy(self, args, output=None, error=None):
        turn = self._current()
        entry = {"step": len(turn["policy"]), "args": args}
        if error is not None:
            entry["error"] = error
        else:
            entry["output"] = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        turn["policy"].append(entry)

    def to_dict(self, run):
        return {"kind": CASSETTE_KIND, "schema_version": CASSETTE_SCHEMA, "run": run, "cases": self.cases}


class CassetteModel:
    """Live: passes calls through to `model` and records each response or classified failure."""

    def __init__(self, model, writer, classify):
        self.model, self.writer, self.classify = model, writer, classify

    @property
    def last_wait_seconds(self):
        return getattr(self.model, "last_wait_seconds", 0.0)

    def invoke(self, messages):
        messages = list(messages)
        try:
            response = self.model.invoke(messages)
        except BaseException as error:
            self.writer.record_model(messages, error_kind=self.classify(error))
            raise
        self.writer.record_model(messages, response=response)
        return response


class CassettePolicyTool:
    """Live: runs the real policy search and records the query and its output."""

    def __init__(self, tool, writer):
        self.tool, self.writer = tool, writer

    def invoke(self, args):
        try:
            output = self.tool.invoke(args)
        except Exception as error:
            self.writer.record_policy(args, error=error.__class__.__name__)
            raise
        self.writer.record_policy(args, output=output)
        return output


# ---------- Replay ----------

def load_cassette(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") != CASSETTE_KIND:
        raise CassetteMismatch(f"{path} is not an evaluation cassette")
    if data.get("schema_version") != CASSETTE_SCHEMA:
        raise CassetteMismatch(f"{path}: cassette schema {data.get('schema_version')} is not supported "
                               f"(expected {CASSETTE_SCHEMA})")
    return data


class Tape:
    """The recorded entries of one turn, consumed strictly in order."""

    def __init__(self, case_id, turn_index, entries):
        self.case_id, self.turn_index = case_id, turn_index
        self.model, self.policy = list(entries.get("model", [])), list(entries.get("policy", []))
        self.model_pos = self.policy_pos = 0

    @property
    def where(self):
        return f"case {self.case_id} turn {self.turn_index + 1}"

    def next(self, channel):
        entries = self.model if channel == "model" else self.policy
        position = self.model_pos if channel == "model" else self.policy_pos
        if position >= len(entries):
            raise CassetteMissing(f"{self.where}: replay needs {channel} entry #{position + 1}, "
                                  f"but the recording has only {len(entries)}")
        entry = entries[position]
        if entry.get("step") != position:
            raise CassetteOrderError(f"{self.where}: {channel} entry #{position + 1} is recorded as step "
                                     f"{entry.get('step')}; the cassette is out of sequence")
        if channel == "model":
            self.model_pos += 1
        else:
            self.policy_pos += 1
        return entry

    def finish(self):
        unused = [(name, len(entries) - pos) for name, entries, pos in
                  (("model", self.model, self.model_pos), ("policy", self.policy, self.policy_pos))
                  if pos < len(entries)]
        if unused:
            detail = ", ".join(f"{count} {name}" for name, count in unused)
            raise CassetteOrderError(f"{self.where}: the replay ended with recorded entries unused ({detail}); "
                                     "the assistant took a different path than in the recording")


class Replay:
    """Hands out one Tape per (case, turn) and tracks input drift."""

    def __init__(self, cassette, strict_inputs=True):
        self.cassette, self.strict_inputs = cassette, strict_inputs
        self.input_drift = []

    def tape(self, case_id, turn_index):
        case = self.cassette["cases"].get(case_id)
        if case is None:
            raise CassetteMissing(f"case {case_id} is not in the cassette")
        turns = case.get("turns", [])
        if turn_index >= len(turns):
            raise CassetteMissing(f"case {case_id} turn {turn_index + 1} is not in the cassette")
        return Tape(case_id, turn_index, turns[turn_index])


class ReplayModel:
    """Plays recorded model responses for the current tape. There is no live fallback."""

    def __init__(self, replay):
        self.replay, self.tape = replay, None

    def invoke(self, messages):
        if self.tape is None:
            raise CassetteOrderError("a model call was made outside a recorded turn")
        entry = self.tape.next("model")
        digest = messages_digest(messages)
        if digest != entry.get("input_sha256"):
            where = f"{self.tape.where} model call #{entry['step'] + 1}"
            if self.replay.strict_inputs:
                raise CassetteMismatch(f"{where}: the model input differs from the recording (system prompt, "
                                       "dataset, tool output or conversation changed)")
            self.replay.input_drift.append(where)
        if "error" in entry:
            raise replay_exception(entry["error"])
        return deserialize_response(entry["response"])


class ReplayPolicySearch:
    """Plays recorded policy-search outputs; the query must match the recording."""

    def __init__(self, replay):
        self.replay, self.tape = replay, None

    def invoke(self, args):
        if self.tape is None:
            raise CassetteOrderError("a policy search was made outside a recorded turn")
        entry = self.tape.next("policy")
        if entry.get("args") != args:
            raise CassetteMismatch(f"{self.tape.where} policy search #{entry['step'] + 1}: "
                                   "the query differs from the recording")
        if "error" in entry:
            if entry["error"] == "PolicyIndexError":
                from app.vector_store import PolicyIndexError
                raise PolicyIndexError("replayed policy index failure")
            raise RuntimeError(f"replayed policy search failure ({entry['error']})")
        return entry["output"]
