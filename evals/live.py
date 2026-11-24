"""
Safeguards for live evaluation runs against the real Groq model.

- ModelCallBudget: a hard cap on model calls for the whole run.
- TokenPacer: waits before each model call so the requests and tokens
  sent in any 60-second window stay under configured limits (actual
  usage from the provider when reported, a conservative estimate before
  the call). The estimate starts pessimistic (3 characters per token)
  and is calibrated from the provider's reported input tokens, with a
  10% safety margin. Clock and sleep are injectable so tests never wait.
- GuardedModel: applies the budget and pacer around every call, and turns
  rate-limit, authentication and connection failures into StopRun so the
  whole run stops at the first one. It never retries, and the live model
  is built with the Groq SDK's automatic retries disabled
  (evals.run.production_model), so each failure is exactly one request.
- isolated_policy_index: copies chroma_db/ to a temporary folder and
  points app.vector_store at the copy for the duration of the run, so
  live policy searches never touch the production index.
- tree_sha256 / redaction helpers used to prove the dataset and index are
  unchanged and that no secret reaches a saved file.

StopRun derives from BaseException: the assistant converts ordinary model
exceptions into AssistantError and swallows failures of the grounding
revision call, and neither may hide a reason to stop.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

FATAL_KINDS = ("rate_limited", "authentication", "connection")
WINDOW_SECONDS = 60.0
CHARS_PER_TOKEN = 3.0          # deliberately pessimistic until calibrated (English is ~4 chars per token)
CALIBRATION_MARGIN = 0.9        # calibrated chars-per-token is reduced 10%, so estimates stay high
OUTPUT_ALLOWANCE_TOKENS = 1024  # reasoning + answer tokens reserved per call


class StopRun(BaseException):
    """Stop the whole live run now (budget spent or a provider failure that must not be retried)."""

    def __init__(self, reason, kind, retry_after=None):
        super().__init__(reason)
        self.reason, self.kind, self.retry_after = reason, kind, retry_after


class LiveRunRefused(RuntimeError):
    """A live run was refused before any model call (no confirmation, too many cases, bad index...)."""


# ---------- Budget ----------

class ModelCallBudget:
    def __init__(self, max_calls):
        if max_calls < 1:
            raise ValueError("the model-call budget must be at least 1")
        self.max_calls, self.used = int(max_calls), 0

    @property
    def remaining(self):
        return self.max_calls - self.used

    def take(self):
        if self.used >= self.max_calls:
            raise StopRun(f"model-call budget of {self.max_calls} reached", "budget")
        self.used += 1


# ---------- Pacing ----------

class TokenPacer:
    """Sliding-window limiter for requests and tokens per minute."""

    def __init__(self, tokens_per_minute, requests_per_minute, clock=time.monotonic, sleep=time.sleep,
                 window=WINDOW_SECONDS):
        self.tpm, self.rpm, self.window = int(tokens_per_minute), int(requests_per_minute), float(window)
        self.clock, self.sleep = clock, sleep
        self.events = deque()            # (time, tokens)
        self.waited_seconds = 0.0
        self.waits = 0

    def _expire(self, now):
        while self.events and now - self.events[0][0] >= self.window:
            self.events.popleft()

    def wait_for(self, estimate):
        """Block until a call estimated at `estimate` tokens fits in the window."""

        if estimate > self.tpm:
            logger.warning("A single call is estimated at %d tokens, above the %d tokens/minute pacing "
                           "limit; it will be sent alone.", estimate, self.tpm)
        while True:
            now = self.clock()
            self._expire(now)
            used = sum(tokens for _t, tokens in self.events)
            fits_tokens = not self.events or used + estimate <= self.tpm
            if fits_tokens and len(self.events) < self.rpm:
                return
            delay = max(self.events[0][0] + self.window - now, 0.0) + 0.5
            self.waits += 1
            self.waited_seconds += delay
            self.sleep(delay)

    def record(self, tokens):
        self.events.append((self.clock(), int(tokens)))

    def pause(self, seconds):
        if seconds > 0:
            self.waited_seconds += seconds
            self.sleep(seconds)


def input_chars(messages, overhead_chars=0):
    chars = overhead_chars
    for message in messages:
        content = getattr(message, "content", "")
        chars += len(content if isinstance(content, str) else json.dumps(content, default=str))
        for call in getattr(message, "tool_calls", None) or []:
            chars += len(json.dumps(call.get("args", {}), default=str)) + len(call.get("name", ""))
    return chars


def estimate_tokens(messages, overhead_chars=0, chars_per_token=CHARS_PER_TOKEN):
    """Conservative token estimate for one call: input characters / chars_per_token plus an output allowance."""

    return math.ceil(input_chars(messages, overhead_chars) / chars_per_token) + OUTPUT_ALLOWANCE_TOKENS


def _reported(response, key):
    usage = getattr(response, "usage_metadata", None) or {}
    value = usage.get(key)
    return value if isinstance(value, int) and value > 0 else None


# ---------- Guarded model ----------

class GuardedModel:
    """Budget + pacing + stop-on-fatal around the real model. Never retries."""

    def __init__(self, model, budget, pacer, classify, overhead_chars=0):
        self.model, self.budget, self.pacer, self.classify = model, budget, pacer, classify
        self.overhead_chars = overhead_chars
        self.chars_per_token = CHARS_PER_TOKEN
        self.calls = 0
        self.last_wait_seconds = 0.0     # pacing wait before the latest call (excluded from its latency)

    def invoke(self, messages):
        chars = input_chars(messages, self.overhead_chars)
        estimate = estimate_tokens(messages, self.overhead_chars, self.chars_per_token)
        self.budget.take()
        waited = self.pacer.waited_seconds
        self.pacer.wait_for(estimate)
        self.last_wait_seconds = self.pacer.waited_seconds - waited
        self.calls += 1
        try:
            response = self.model.invoke(messages)
        except Exception as error:
            self.pacer.record(estimate)              # a failed request may still count against the limit
            failure = self.classify(error)
            if failure.kind in FATAL_KINDS:
                raise StopRun(f"provider failure: {failure.kind}", failure.kind,
                              getattr(failure, "retry_after", None)) from error
            raise
        self.pacer.record(_reported(response, "total_tokens") or estimate)
        reported_input = _reported(response, "input_tokens")
        if reported_input:
            # Calibrate from real usage; never above the observed ratio (estimates stay on the high side).
            self.chars_per_token = max(1.0, chars / reported_input * CALIBRATION_MARGIN)
        return response


def classify(error):
    from app.errors import classify_provider_error

    return classify_provider_error(error)


def classify_kind(error):
    if isinstance(error, StopRun):
        return error.kind
    return classify(error).kind


# ---------- Production index isolation ----------

def tree_sha256(folder):
    """SHA-256 over the relative paths and bytes of every file under `folder` (None if missing)."""

    folder = Path(folder)
    if not folder.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(folder)).encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@contextlib.contextmanager
def isolated_policy_index(source=None):
    """
    Copy the policy index to a temporary folder and point app.vector_store
    at it; restore the original path and drop cached clients afterwards.
    Yields the temporary index path.
    """

    from app import vector_store

    source = Path(source or vector_store.CHROMA_DIR)
    if not source.is_dir():
        raise LiveRunRefused(f"the policy index folder {source.name}/ does not exist; build it first with "
                             "`python -m app.vector_store`")
    original = vector_store.CHROMA_DIR
    workdir = Path(tempfile.mkdtemp(prefix="bankmate-eval-index-"))
    copy = workdir / "chroma_db"
    shutil.copytree(source, copy)
    vector_store.reset_index_state()
    vector_store.CHROMA_DIR = copy
    try:
        yield copy
    finally:
        vector_store.CHROMA_DIR = original
        vector_store.reset_index_state()
        shutil.rmtree(workdir, ignore_errors=True)


def check_index_copy(path):
    """Refuse to start unless the copied index is already valid (never build during an evaluation)."""

    from app import vector_store

    status = vector_store.inspect_index(path)
    if status.state != "valid":
        raise LiveRunRefused(f"the policy index is {status.state} ({status.reason}); rebuild it with "
                             "`python -m app.vector_store` before a live run")
    vector_store.ensure_policy_index(path)
    return status


# ---------- Redaction ----------

def secrets_in_environment():
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    return [key] if len(key) >= 8 else []


def redact_tree(value, secrets):
    """Copy of a JSON-like value with every string passed through app.security.redact_secrets."""

    from app.security import redact_secrets

    if isinstance(value, str):
        return redact_secrets(value, secrets)
    if isinstance(value, dict):
        return {redact_tree(k, secrets): redact_tree(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_tree(v, secrets) for v in value]
    return value


def safe_json(value, secrets):
    """Redacted JSON text; raises if any secret would still be written."""

    text = json.dumps(redact_tree(value, secrets), indent=2, ensure_ascii=False, default=str) + "\n"
    for secret in secrets:
        if secret and secret in text:
            raise RuntimeError("refusing to write a file that still contains a secret")
    return text
