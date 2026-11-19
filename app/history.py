"""
Conversation history helpers (pure functions, no I/O).

A history is: [system message, turn, turn, ...] where each turn starts
with a HumanMessage and contains everything up to the next HumanMessage:
the assistant's tool-call requests, the tool results and the final
answer. Because the engine only commits complete turns, trimming whole
turns can never separate a tool call from its result.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def split_turns(messages):
    """Return (prefix, turns): messages before the first user message, and turns."""

    prefix = []
    turns = []

    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append([message])
        elif turns:
            turns[-1].append(message)
        else:
            prefix.append(message)

    return prefix, turns


def trim_to_recent_turns(messages, max_turns):
    """
    Keep the prefix (system prompt) and the most recent `max_turns` turns.
    Returns (trimmed_messages, dropped_turn_count).
    """

    prefix, turns = split_turns(messages)

    if max_turns is None or len(turns) <= max_turns:
        return list(messages), 0

    dropped = len(turns) - max_turns
    kept = [message for turn in turns[dropped:] for message in turn]
    return prefix + kept, dropped


def is_complete_history(messages):
    """
    True when every assistant tool call is answered by a tool result
    before the next user message, no tool result is orphaned, and the
    history does not end mid tool-call sequence.
    """

    pending = set()

    for message in messages:
        if isinstance(message, HumanMessage):
            if pending:
                return False

        elif isinstance(message, AIMessage):
            if pending:
                return False
            pending = {call["id"] for call in (message.tool_calls or [])}

        elif isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                return False
            pending.discard(message.tool_call_id)

    return not pending
