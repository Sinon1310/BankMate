"""
Session-state helpers shared across pages.

Conversations
-------------
A browser session can hold several conversations. Each one has its own
`session_id`, which is passed to app.assistant.ask_bankmate_with_sources;
the backend keys its message history and grounding context on that id,
so conversations never share memory.

Conversations live in Streamlit session state (this browser tab) and in
the backend's in-process memory, so they end when the server restarts.

Message shape: {"role": "user" | "assistant", "content": str,
                "request_id": str (user, stable per submission),
                "sources": list (assistant, optional),
                "error": True (assistant failure, optional)}

A conversation may carry "memory_lost_at": the message index where the
backend's memory of it ended (idle expiry or server restart). Messages
before that index are shown but are no longer available to the model.
"""

import time
import uuid

import streamlit as st


CONVERSATIONS = "conversations"
ACTIVE_ID = "active_conversation_id"
PENDING_PROMPT = "pending_prompt"

DEFAULT_TITLE = "New conversation"
TITLE_LENGTH = 48


# ---------- Setup ----------

def _make_conversation():
    conversation_id = str(uuid.uuid4())

    return {
        "id": conversation_id,
        # Kept separate from `id` for clarity: this is the backend key.
        "session_id": conversation_id,
        "title": DEFAULT_TITLE,
        "messages": [],
        "created_at": time.time(),
    }


def init_state():
    if CONVERSATIONS not in st.session_state:
        first = _make_conversation()
        st.session_state[CONVERSATIONS] = {first["id"]: first}
        st.session_state[ACTIVE_ID] = first["id"]

    st.session_state.setdefault(PENDING_PROMPT, None)


# ---------- Conversations ----------

def active_conversation():
    conversations = st.session_state[CONVERSATIONS]
    active = conversations.get(st.session_state.get(ACTIVE_ID))

    if active is None:
        active = new_conversation()

    return active


def conversations_newest_first():
    return sorted(
        st.session_state[CONVERSATIONS].values(),
        key=lambda conversation: conversation["created_at"],
        reverse=True,
    )


def new_conversation():
    """
    Start a fresh conversation and make it active. An existing empty
    conversation is reused so repeated clicks don't pile up blanks.
    """

    for conversation in st.session_state[CONVERSATIONS].values():
        if not conversation["messages"]:
            st.session_state[ACTIVE_ID] = conversation["id"]
            return conversation

    conversation = _make_conversation()
    st.session_state[CONVERSATIONS][conversation["id"]] = conversation
    st.session_state[ACTIVE_ID] = conversation["id"]
    return conversation


def switch_conversation(conversation_id):
    if conversation_id in st.session_state[CONVERSATIONS]:
        st.session_state[ACTIVE_ID] = conversation_id


def make_title(prompt):
    text = " ".join((prompt or "").split())

    if len(text) <= TITLE_LENGTH:
        return text or DEFAULT_TITLE

    return text[:TITLE_LENGTH - 1].rstrip() + "…"


# ---------- Messages ----------

def add_user_message(conversation, prompt):
    if not conversation["messages"]:
        conversation["title"] = make_title(prompt)

    conversation["messages"].append({
        "role": "user",
        "content": prompt,
        # Stable ID for this submission: re-sending it (e.g. after an
        # interrupted run) returns the saved answer instead of asking again.
        "request_id": str(uuid.uuid4()),
    })


def awaiting_reply(conversation):
    messages = conversation["messages"]
    return bool(messages) and messages[-1]["role"] == "user"


def add_assistant_reply(conversation, answer, sources):
    conversation["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources or [],
    })


def add_assistant_error(conversation, message):
    conversation["messages"].append({
        "role": "assistant",
        "content": message,
        "error": True,
    })


def has_reply_since_memory_reset(conversation):
    """True if a successful answer exists after the last memory reset."""

    start = conversation.get("memory_lost_at") or 0
    return any(
        message["role"] == "assistant" and not message.get("error")
        for message in conversation["messages"][start:]
    )


def mark_memory_lost(conversation, at=None):
    """
    Record where the model's memory of this conversation ends: messages
    before index `at` (default: all current messages) are no longer
    available to it.
    """

    conversation["memory_lost_at"] = len(conversation["messages"]) if at is None else at


def retry_last(conversation):
    """
    Remove a trailing error so the preceding question is sent again.
    Returns True if there was something to retry.
    """

    messages = conversation["messages"]

    if len(messages) >= 2 and messages[-1].get("error") and messages[-2]["role"] == "user":
        messages.pop()
        return True

    return False


# ---------- Policy selection (shared by Assistant and Policy pages) ----------

SELECTED_POLICY = "selected_policy_id"


def select_policy(policy_id):
    st.session_state[SELECTED_POLICY] = policy_id


def selected_policy():
    return st.session_state.get(SELECTED_POLICY)


def clear_policy():
    st.session_state[SELECTED_POLICY] = None


# ---------- Cross-page prompt hand-off ----------

def queue_prompt(prompt, new_conversation=False):
    """
    Store a prompt for the assistant page to send on its next run.
    With new_conversation=True it starts in a fresh conversation, so
    entry points like Home don't append to an unrelated thread.
    """

    prompt = (prompt or "").strip()

    st.session_state[PENDING_PROMPT] = (
        {"prompt": prompt, "new_conversation": new_conversation}
        if prompt else None
    )


def pop_pending_prompt():
    pending = st.session_state.get(PENDING_PROMPT)
    st.session_state[PENDING_PROMPT] = None
    return pending
