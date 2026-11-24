"""
Recording wrappers and the offline (scripted) model and policy search.

All of these plug into the assistant's existing injection points:
`ask_bankmate_with_sources(question, session_id, model=..., tools=...)`,
where `model` needs `.invoke(messages)` and `tools` is a {name: tool}
mapping whose values need `.invoke(args)`. Nothing in app/ changes.

- RecordingModel wraps any model (scripted, live Groq or replayed) and
  records each call: latency, tool calls requested, token usage when the
  provider reports it, and its role (tool_calls / answer / revision).
- RecordingTools wraps the tool map and records each tool execution
  (only calls that passed the assistant's authorization and argument
  checks reach a tool). It can inject a failure for error-handling cases.
- ScriptedModel is the offline stand-in for the LLM: it replays a case's
  planned tool calls and writes its final answer from the tool results it
  actually received, so figures always come from the active dataset.
- ScriptedPolicySearch returns real policy chunks (read from the policy
  files) chosen by the case, in the exact JSON contract of
  app.tools.search_banking_policies, without embeddings or ChromaDB.
"""

import json
import re
import time
from decimal import Decimal, InvalidOperation

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

POLICY_TOOL = "search_banking_policies"


class InjectedToolFailure(RuntimeError):
    """Raised by a tool configured to fail, to exercise error handling."""


# ---------- Recording ----------

class Recorder:
    """Collects model calls and tool executions for one assistant turn."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.model_calls = []
        self.tool_executions = []
        self.last_model_input = []

    def snapshot(self):
        return {"model_calls": list(self.model_calls), "tool_executions": list(self.tool_executions),
                "last_model_input": list(self.last_model_input)}


def _usage(response):
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        return None
    return {key: usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")}


class RecordingModel:
    """
    Wraps a model; records latency, requested tool calls, token usage and
    the call's role per call. A failed call is recorded (with its error
    class) and re-raised unchanged, including the BaseException signals
    the live runner and replay use to stop a run.
    """

    def __init__(self, model, recorder):
        self.model, self.recorder = model, recorder

    def invoke(self, messages):
        self.recorder.last_model_input = list(messages)
        started = time.perf_counter()
        entry = {"latency_ms": None, "tool_calls": [], "has_answer": False, "final": False, "usage": None}
        try:
            response = self.model.invoke(messages)
        except BaseException as error:
            entry["error"] = error.__class__.__name__
            raise
        else:
            calls = getattr(response, "tool_calls", None) or []
            entry.update(
                tool_calls=[{"name": c["name"], "args": c.get("args", {}), "id": c.get("id")} for c in calls],
                has_answer=bool(getattr(response, "content", "")) and not calls,
                # The assistant treats any response without tool calls as its final answer.
                final=not calls,
                usage=_usage(response))
            return response
        finally:
            # Live pacing waits happen inside the call; report them separately from provider latency.
            paced = float(getattr(self.model, "last_wait_seconds", 0.0) or 0.0)
            entry["latency_ms"] = round(max((time.perf_counter() - started) - paced, 0.0) * 1000, 3)
            if paced:
                entry["paced_ms"] = round(paced * 1000, 3)
            entry["role"] = _call_role(self.recorder.model_calls, entry)
            self.recorder.model_calls.append(entry)


def _call_role(previous, entry):
    if any(call.get("final") for call in previous):
        return "revision"
    if entry.get("error"):
        return "error"
    return "answer" if entry["final"] else "tool_calls"


def count_revisions(model_calls):
    """
    Grounding revisions in one turn, from the recorded call sequence: the
    assistant's loop ends at the first response without tool calls, so any
    model call after it is a revision request from the grounding check
    (assistant._enforce_grounding). Works for every mode (scripted, live, replay).
    """

    seen_final, revisions = False, 0
    for call in model_calls:
        if seen_final:
            revisions += 1
        elif call.get("final"):
            seen_final = True
    return revisions


class RecordedTool:
    def __init__(self, name, tool, recorder, fail=False):
        self.name, self.tool, self.recorder, self.fail = name, tool, recorder, fail

    def invoke(self, args):
        started = time.perf_counter()
        entry = {"name": self.name, "args": dict(args) if isinstance(args, dict) else args}
        try:
            if self.fail:
                raise InjectedToolFailure(f"injected failure for evaluation ({self.name})")
            output = self.tool.invoke(args)
            entry.update(status="ok", output=output)
            return output
        except Exception as error:
            entry.update(status="error", error=error.__class__.__name__)
            raise
        finally:
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.recorder.tool_executions.append(entry)


def recording_tools(tool_map, recorder, fail=()):
    """{name: tool} -> {name: RecordedTool}; names in `fail` raise when executed."""

    return {name: RecordedTool(name, tool, recorder, fail=name in fail) for name, tool in tool_map.items()}


# ---------- Offline policy search ----------

class ScriptedPolicySearch:
    """
    Offline policy search: returns the chunks of the sections the case
    names ("POL-001/REQUIRED INFORMATION"), or NO_RELEVANT_POLICY_FOUND
    for an empty selection. Same JSON contract as the real tool.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.selection = []

    def invoke(self, args):
        from app.tools import UNTRUSTED_EVIDENCE_NOTICE

        wanted = [tuple(item.split("/", 1)) for item in self.selection]
        chosen = [c for key in wanted for c in self.chunks
                  if (c.metadata["policy_id"], c.metadata["section"]) == key]
        if not chosen:
            return json.dumps({"status": "NO_RELEVANT_POLICY_FOUND",
                               "message": "The available policy documents do not contain sufficiently "
                                          "relevant information to answer this question.",
                               "sources": [], "evidence": []})
        evidence, sources = [], []
        for chunk in chosen:
            meta = chunk.metadata
            source = {"policy_id": meta["policy_id"], "policy_name": meta["policy_name"],
                      "version": meta["version"], "section": meta["section"],
                      "source_file": meta["source_file"], "distance": 0.0}
            evidence.append({**source, "content": chunk.page_content})
            if source not in sources:
                sources.append(source)
        return json.dumps({"status": "SUCCESS", "notice": UNTRUSTED_EVIDENCE_NOTICE,
                           "sources": sources, "evidence": evidence}, ensure_ascii=False)


# ---------- Offline model ----------

TEMPLATE = re.compile(r"\{\{\s*([a-z_]+)(?:#(\d+))?:([^|}]*?)(?:\|([a-z]+))?\s*\}\}")


class ScriptError(ValueError):
    """A scripted answer referenced a tool result that does not exist."""


def format_inr(value):
    """
    Decimal-safe ₹ formatting with Indian grouping and two decimals (₹1,87,226.80; −₹40.50; ₹95,000.00),
    the format the system prompt asks for (rule 56), so scripted answers model compliant output.
    """

    amount = Decimal(str(value))
    sign = "−" if amount < 0 else ""
    amount = abs(amount)
    whole = int(amount)
    digits = str(whole)
    if len(digits) > 3:
        head, tail, groups = digits[:-3], digits[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        digits = ",".join(([head] if head else []) + groups + [tail])
    return f"{sign}₹{digits}" + f"{amount.quantize(Decimal('0.01'))}"[-3:]


def _lookup(data, path):
    for part in [p for p in path.split(".") if p]:
        if isinstance(data, list):
            data = data[int(part)]
        elif isinstance(data, dict):
            data = data[part]
        else:
            raise KeyError(part)
    return data


def turn_tool_results(messages):
    """[(tool_name, parsed_or_raw_output), ...] for the current turn (after the last user message)."""

    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    names, results = {}, []
    for message in messages[start + 1:]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                names[call.get("id")] = call["name"]
        elif isinstance(message, ToolMessage):
            content = message.content
            try:
                content = json.loads(content)
            except (TypeError, ValueError):
                pass
            results.append((names.get(message.tool_call_id, "?"), content))
    return results


def render(template, messages):
    """Fill {{tool[#n]:path|fmt}} from the tool results the model has seen this turn."""

    results = turn_tool_results(messages)

    def fill(match):
        name, index, path, fmt = match.group(1), int(match.group(2) or 0), match.group(3), match.group(4)
        matching = [content for tool, content in results if tool == name]
        try:
            value = _lookup(matching[index], path)
        except (IndexError, KeyError, ValueError, TypeError) as error:
            raise ScriptError(f"no value for {match.group(0)} ({error.__class__.__name__})") from None
        if fmt == "inr":
            try:
                return format_inr(value)
            except (InvalidOperation, ValueError, TypeError):
                raise ScriptError(f"{match.group(0)} is not an amount") from None
        if fmt == "count":
            return str(len(value))
        return str(value)

    return TEMPLATE.sub(fill, template)


class ScriptedModel:
    """
    Plays one scripted step per model call for the current turn. A step is
    {"tool_calls": [{"name", "args"}]} or {"answer": "<template>"}. Calls
    beyond the script (a grounding revision request) repeat the final
    answer and are counted as revisions.
    """

    def __init__(self):
        self.steps, self.position, self.last_answer, self.revisions, self._ids = [], 0, None, 0, 0

    def start_turn(self, steps):
        self.steps, self.position, self.last_answer, self.revisions = list(steps or []), 0, None, 0

    def invoke(self, messages):
        if self.position >= len(self.steps):
            if self.last_answer is None:
                raise ScriptError("script ended without a final answer")
            self.revisions += 1
            return AIMessage(content=self.last_answer)
        step = self.steps[self.position]
        self.position += 1
        if "answer" in step:
            self.last_answer = render(step["answer"], messages)
            return AIMessage(content=self.last_answer)
        calls = []
        for call in step["tool_calls"]:
            self._ids += 1
            calls.append({"name": call["name"], "args": dict(call.get("args", {})),
                          "id": f"eval_call_{self._ids}", "type": "tool_call"})
        return AIMessage(content="", tool_calls=calls)
