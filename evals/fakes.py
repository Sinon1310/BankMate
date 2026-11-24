"""
Test doubles for live-mode evaluation: never used by a real live run.

- FakeGroq behaves like the tool-bound ChatGroq model from the outside: it
  answers each question with that case's offline script (so tool calls and
  answers are realistic), attaches Groq-style usage and response metadata,
  answers a grounding revision request by repeating its answer, and can
  raise a provider error at a chosen call number.
- FakePolicySearch returns real policy chunks for the queries the offline
  scripts send, in the policy tool's JSON contract (no embeddings).
"""

from langchain_core.messages import AIMessage, HumanMessage

from evals.recorder import POLICY_TOOL, ScriptedModel, ScriptedPolicySearch

REVISION_PREFIX = "Revise your previous answer."


class FakeGroq:
    model_name = "fake-groq/offline-double"

    def __init__(self, cases, fail_at=None):
        self.scripts = {turn["question"]: turn["offline"]["script"] for case in cases for turn in case["turns"]}
        self.fail_at = dict(fail_at or {})         # {call number (1-based): exception}
        self.scripted = ScriptedModel()
        self.calls = 0
        self._question = None

    def invoke(self, messages):
        self.calls += 1
        if self.calls in self.fail_at:
            raise self.fail_at[self.calls]
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        question = next((m.content for m in reversed(humans) if not str(m.content).startswith(REVISION_PREFIX)),
                        None)
        if question != self._question:
            self._question = question
            self.scripted.start_turn(self.scripts[question])
        reply = self.scripted.invoke(messages)
        chars = sum(len(str(m.content)) for m in messages)
        output = max(len(str(reply.content)) // 4, 8)
        return AIMessage(content=reply.content, tool_calls=reply.tool_calls,
                         usage_metadata={"input_tokens": chars // 4, "output_tokens": output,
                                         "total_tokens": chars // 4 + output},
                         response_metadata={"model_name": self.model_name, "finish_reason":
                                            "tool_calls" if reply.tool_calls else "stop"})


class FakePolicySearch:
    """Real chunks for the policy queries in the offline scripts; NO_RELEVANT_POLICY_FOUND otherwise."""

    def __init__(self, chunks, cases):
        self.search = ScriptedPolicySearch(chunks)
        self.by_query = {}
        for case in cases:
            for turn in case["turns"]:
                for step in turn["offline"]["script"]:
                    for call in step.get("tool_calls", []):
                        if call["name"] == POLICY_TOOL:
                            self.by_query[call["args"]["question"]] = turn["offline"].get("policy", [])

    def invoke(self, args):
        self.search.selection = self.by_query.get(args.get("question"), [])
        return self.search.invoke(args)
