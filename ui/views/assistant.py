"""
AI Assistant: multi-conversation chat over the existing backend.

Backend contract (unchanged): ask_bankmate_with_sources(question,
session_id=...) -> {"answer": str, "sources": [grouped citation, ...]}.
Each conversation passes its own session_id, so memory and grounding
context stay isolated per conversation. Citations are displayed exactly
as returned; this page never adds, removes or infers sources.

Run order matters: the new question is recorded before the sidebar and
thread render, so titles and history are current on the same run; the
reply is generated last, under a visible "working" indicator.
"""

from html import escape

import streamlit as st

from app.security import MAX_QUESTION_CHARS
from ui import components as ui
from ui import index_status
from ui import state


USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"

ERROR_MESSAGE = (
    "Something went wrong while answering. Your conversation is intact — "
    "try again in a moment."
)

EMPTY_STATE_PROMPTS = [
    ("profile", "Show me the profile of customer C101."),
    ("metrics", "What are C103's transaction metrics?"),
    ("unauthorized", "What should a customer do about an unauthorized transaction?"),
    ("refund", "What happens when a refund is approved?"),
]


# ---------- Backend (loaded lazily: needs GROQ_API_KEY) ----------

try:
    from app.assistant import ask_bankmate_with_sources, has_conversation_memory
    backend_error = None
except ValueError as error:
    # Configuration problems (e.g. GROQ_API_KEY) have value-free messages.
    ask_bankmate_with_sources = None
    has_conversation_memory = None
    backend_error = str(error)
except Exception as error:  # noqa: BLE001 - details go to the server log only
    import logging

    from app.security import redact_secrets

    logging.getLogger(__name__).error("Assistant backend failed to load: %s", redact_secrets(repr(error)))
    ask_bankmate_with_sources = None
    has_conversation_memory = None
    backend_error = "The AI backend could not be started. Details are in the server logs."

MEMORY_LOST_PENDING = (
    "This question wasn't answered before BankMate's memory of this "
    "conversation expired. Use Try again to ask it now — BankMate won't "
    "have the earlier context."
)


# ---------- Resolve this run's input before rendering ----------

typed_prompt = st.chat_input(
    "Message BankMate…",
    key="assistant-input",
    disabled=backend_error is not None,
    max_chars=MAX_QUESTION_CHARS,
)

pending = state.pop_pending_prompt()

if pending:
    if pending["new_conversation"]:
        state.new_conversation()
    prompt = pending["prompt"]
else:
    prompt = typed_prompt

conversation = state.active_conversation()

# If the backend no longer holds this conversation (idle expiry or a
# server restart) although it was answered before, mark where context
# ended instead of implying the model still sees the older messages.
if (
    backend_error is None
    and state.has_reply_since_memory_reset(conversation)
    and not has_conversation_memory(conversation["session_id"])
):
    if state.awaiting_reply(conversation):
        # Context ends before the stale, unanswered question, which is
        # never re-sent silently: the user can ask it again explicitly.
        state.mark_memory_lost(conversation, at=len(conversation["messages"]) - 1)
        state.add_assistant_error(conversation, MEMORY_LOST_PENDING)
    else:
        state.mark_memory_lost(conversation)

prompt_added = bool(prompt) and backend_error is None

if prompt_added:
    state.add_user_message(conversation, prompt)


# ---------- Sidebar: conversations ----------

with st.sidebar:
    if st.button(
        "New chat",
        icon=":material/edit_square:",
        key="assistant-new-chat",
        use_container_width=True,
        type="primary",
    ):
        state.new_conversation()
        st.rerun()

    ui.html('<div class="bm-side-label">Conversations</div>')

    for item in state.conversations_newest_first():
        is_active = item["id"] == conversation["id"]
        container_key = "bm-conv-active" if is_active else f"bm-conv-{item['id']}"

        with st.container(key=container_key):
            if st.button(
                item["title"],
                key=f"conv-btn-{item['id']}",
                icon=":material/chat_bubble:",
                use_container_width=True,
                type="tertiary",
                disabled=is_active,
                help=None if is_active else "Open this conversation",
            ):
                state.switch_conversation(item["id"])
                st.rerun()


# ---------- Thread ----------

def render_message(index, message):
    if message["role"] == "user":
        with st.container(key=f"bm-msg-user-{index}"):
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(message["content"])
        return

    tone = "error" if message.get("error") else "assistant"

    with st.container(key=f"bm-msg-{tone}-{index}"):
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(message["content"])

            if not message.get("error") and message.get("sources"):
                ui.sources_block(message["sources"])

                # Open each cited policy in the library (same session, so
                # this conversation is still here when the user returns).
                with st.container(horizontal=True, gap="small", key=f"bm-srclinks-{index}"):
                    for citation in message["sources"]:
                        policy_id = citation.get("policy_id")
                        if policy_id and st.button(
                            f"Open {policy_id}",
                            key=f"src-open-{index}-{policy_id}",
                            icon=":material/menu_book:",
                            type="tertiary",
                        ):
                            ui.go_to_policy(policy_id)


with st.container(key="bm-thread"):

    messages = conversation["messages"]

    if backend_error is None:
        index_status.show_index_banner()

    if backend_error is not None and not messages:
        # No suggestion chips here: they could not do anything.
        ui.html(
            '<div class="bm-chat-empty">'
            f'<span class="bm-icon-tile bm-chat-empty-icon bm-tile-danger">{ui.icon("cloud_off")}</span>'
            '<h2 class="bm-chat-empty-title">Assistant unavailable</h2>'
            '<p class="bm-chat-empty-body">The AI backend could not be '
            "started, so messages are disabled. The rest of BankMate keeps "
            "working.</p>"
            f'<p class="bm-chat-empty-detail">Details: {escape(backend_error)}</p>'
            "</div>"
        )

    elif not messages:
        ui.html(
            '<div class="bm-chat-empty">'
            f'<span class="bm-icon-tile violet bm-chat-empty-icon">{ui.icon("auto_awesome")}</span>'
            '<h2 class="bm-chat-empty-title">How can I help?</h2>'
            '<p class="bm-chat-empty-body">Ask about a customer, their '
            "transactions or a bank procedure. Policy answers list the "
            "fictional training documents they rely on.</p></div>"
        )

        with st.container(horizontal=True, gap="small", key="bm-chiprow-empty"):
            chosen = None

            for key, suggestion in EMPTY_STATE_PROMPTS:
                if ui.prompt_chip(suggestion, f"empty-{key}"):
                    chosen = suggestion

        if chosen:
            state.queue_prompt(chosen)
            st.rerun()

    else:
        ui.html(
            '<div class="bm-thread-header">'
            f'<div class="bm-thread-title">{escape(conversation["title"])}</div>'
            '<div class="bm-thread-meta">Memory is kept within this conversation only</div>'
            "</div>"
        )

    memory_lost_at = conversation.get("memory_lost_at")

    def render_memory_notice():
        with st.container(key="bm-memory-notice"):
            ui.html(
                '<div class="bm-memory-notice">'
                f'{ui.icon("history_toggle_off")}<div>'
                "<strong>Earlier context has expired</strong>"
                "<p>BankMate no longer remembers the messages above, so new "
                "questions here start without that context.</p></div></div>"
            )
            if st.button(
                "Start a fresh conversation",
                icon=":material/edit_square:",
                key="assistant-fresh-start",
                type="secondary",
            ):
                state.new_conversation()
                st.rerun()

    for index, message in enumerate(messages):
        if index == memory_lost_at:
            render_memory_notice()
        render_message(index, message)

    if memory_lost_at is not None and memory_lost_at >= len(messages):
        render_memory_notice()

    # Retry is offered only for a failure at the end of the thread.
    if messages and messages[-1].get("error") and backend_error is None:
        with st.container(key="bm-retry"):
            if st.button(
                "Try again",
                icon=":material/refresh:",
                key="assistant-retry",
                type="secondary",
            ):
                state.retry_last(conversation)
                st.rerun()

    if backend_error is not None and messages:
        st.error(
            "The assistant is unavailable, so new messages are disabled. "
            f"Details: {backend_error}",
            icon=":material/error:",
        )

    # ---------- Generate the pending reply ----------

    if backend_error is None and state.awaiting_reply(conversation):
        question = messages[-1]["content"]
        request_id = messages[-1].get("request_id")

        with st.container(key=f"bm-msg-assistant-{len(messages)}"):
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                placeholder = st.empty()
                placeholder.markdown(ui.thinking_indicator(), unsafe_allow_html=True)

                # First show the new question and indicator on a clean run,
                # then call the backend on the next run. This stops elements
                # from the previous layout (e.g. empty-state chips) lingering
                # faded on screen while the answer is generated.
                if prompt_added:
                    st.rerun()

                failure_message = None

                try:
                    result = ask_bankmate_with_sources(
                        question,
                        session_id=conversation["session_id"],
                        request_id=request_id,
                    )
                except Exception as error:  # noqa: BLE001 - recorded as an error message
                    result = None
                    # AssistantError carries a user-safe message (e.g. rate
                    # limits); anything else gets the generic message. Raw
                    # provider errors are logged by the backend, never shown.
                    failure_message = getattr(error, "user_message", None) or ERROR_MESSAGE

                placeholder.empty()

        if result is None:
            state.add_assistant_error(conversation, failure_message or ERROR_MESSAGE)
        else:
            state.add_assistant_reply(conversation, result["answer"], result["sources"])

        # Re-render from state so the reply, sources and retry control
        # all come from the same code path as history.
        st.rerun()
