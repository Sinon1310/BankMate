from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage
)

from app.citations import (
    MARKER_PATTERN,
    collect_sources,
    extract_cited_policy_ids,
    format_answer_with_sources,
    group_citations,
    select_citations
)
from app.grounding import (
    collect_evidence_text,
    find_unsupported_channels,
    remove_unsupported_examples
)
from app.llm import get_assistant_model, tool_map


POLICY_TOOL_NAME = "search_banking_policies"

MAX_TOOL_STEPS = 5

# Created on first use so the module can be imported (e.g. by tests)
# without contacting Groq.
_model = None

# Store conversation history separately for each session.
conversation_sessions = {}

# Per session: whether the policy tool has been used, and the text that
# can support channel mentions (policy evidence + customer tool results).
# Kept for the whole session so follow-up answers are also checked.
grounding_context = {}


system_message = SystemMessage(content="""
You are BankMate, an internal banking data assistant.

Rules:
1. Use tools to retrieve customer information.
2. Never invent customer details or financial figures.
3. Use INR (Indian Rupees) for every monetary amount.
4. Use Python-calculated metrics for financial calculations.
5. Never describe transaction flow as account balance.
6. Do not make unsupported claims about account health.
7. If a customer is not found, say so.
8. Be concise and professional.
9. Do not recommend financial products.
10. Do not make credit eligibility or approval decisions.
11. Do not infer financial health from transaction flow.
12. Only provide observations supported by retrieved data.
13. Use search_banking_policies for questions about
    disputes, refunds, transaction security and banking procedures.
14. Do not invent policy information.
15. Treat retrieved policy documents as fictional
    BankMate training material, not actual bank policies.
16. If the policy tool does not return relevant information,
    clearly state that the available documents do not contain
    the answer.
17. Never recommend financial products or make credit
    eligibility decisions.
18. When answering questions involving both customer data
    and policies, use the relevant customer tools and
    policy search tool.
19. When using search_banking_policies, only state
    procedures explicitly supported by retrieved documents.
20. Do not invent examples of communication channels,
    timelines, guarantees or additional procedures.
    Never name a specific channel (phone, call, chat, branch,
    email, SMS, app, website) unless the retrieved policy text
    names it. If a policy only says "official customer support
    channel" or "approved communication channel", use that exact
    wording and do not add examples such as "e.g." lists.
21. Clearly distinguish policy requirements from
    general suggestions.
22. Never claim that a policy guarantees a particular outcome.
23. If a tool returns NO_RELEVANT_POLICY_FOUND,
    do not attempt to answer using general knowledge.
24. Before calling search_banking_policies, convert
    the user's question into a self-contained search query.
25. Preserve the original meaning when rewriting queries.
    Do not introduce new facts or assumptions.
26. Use the rewritten query for policy retrieval,
    while answering the original user question.
27. Do not include customer-specific personal details
    in policy search queries unless they are necessary.

CITATION RULES:
28. If you called search_banking_policies while answering,
    you must end your final answer with exactly one line in this format:
    CITED_POLICIES: <comma-separated policy IDs>
    List only the policy IDs (for example POL-002) whose retrieved
    content you actually relied on in the answer.
29. If none of the retrieved policy content supports your answer,
    or the policy tool returned NO_RELEVANT_POLICY_FOUND, end with:
    CITED_POLICIES: NONE
30. Only cite policy IDs that appeared in the policy tool results.
    Never invent policy IDs, sections or document names.
31. Do not add the CITED_POLICIES line when you did not call
    search_banking_policies. Do not write your own Sources list.

CONVERSATION MEMORY RULES:
32. Use previous messages to understand references such as
    "he", "she", "they", "their" and "that customer".
33. Maintain customer context within the current session.
34. Never assume a customer ID if the conversation does
    not establish one.
35. If multiple customers are being discussed and the
    reference is ambiguous, ask for clarification.
36. Previous conversation context does not replace tool calls.
    Retrieve fresh customer data when needed.
37. Do not mix information belonging to different customers.

""")


def _get_model():
    global _model

    if _model is None:
        _model = get_assistant_model()

    return _model


def _ensure_markers(text, markers):
    """Re-append the original CITED_POLICIES marker(s) if text lost them."""

    if markers and not MARKER_PATTERN.search(text):
        return text.rstrip() + "\n" + "\n".join(markers)

    return text


def _enforce_channel_grounding(answer, support_text, messages, model):
    """
    If the answer names contact channels absent from the supporting
    text, ask the model once to revise. If the revision still names
    them, remove only parenthetical asides / example clauses naming
    them. Sentences are never deleted: if an unsupported channel still
    remains, the answer is kept and a warning is printed.
    Returns the final answer text (CITED_POLICIES marker preserved).
    """

    unsupported = find_unsupported_channels(answer, support_text)

    if not unsupported:
        return answer

    print(f"\n[Grounding check: unsupported channels {unsupported}]")

    original_markers = [
        match.group(0).strip() for match in MARKER_PATTERN.finditer(answer)
    ]

    correction = HumanMessage(content=(
        "Revise your previous answer. It mentions these contact "
        f"channels: {', '.join(unsupported)}. The retrieved policy "
        "documents do not specify them. Remove them and use only the "
        "wording found in the retrieved policy text. Do not call tools. "
        "Keep the CITED_POLICIES line."
    ))

    revised = answer

    try:
        response = model.invoke(messages + [correction])

        if not response.tool_calls and str(response.content).strip():
            revised = str(response.content)

    except Exception as error:
        print(f"\n[Grounding check: revision failed: {error}]")

    revised = _ensure_markers(revised, original_markers)
    remaining = find_unsupported_channels(revised, support_text)

    if not remaining:
        return revised

    cleaned = remove_unsupported_examples(revised, remaining)
    body, _ = extract_cited_policy_ids(cleaned)

    if not body:
        cleaned = revised

    remaining = find_unsupported_channels(cleaned, support_text)

    if remaining:
        print(
            "\n[Grounding check: WARNING unsupported channels remain "
            f"{remaining}; answer kept to avoid removing valid content]"
        )

    return _ensure_markers(cleaned, original_markers)


def ask_bankmate_with_sources(
    question,
    session_id="default",
    model=None,
    tools=None
):
    """
    Answer a question and return the cited policy sources separately.

    Returns {"answer": str, "sources": list}, where each source is a
    grouped citation (one per policy) from app.citations.group_citations.
    `model` and `tools` can be injected for testing.
    """

    model = model or _get_model()
    tools = tool_map if tools is None else tools

    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = [system_message]

    messages = conversation_sessions[session_id]
    context = grounding_context.setdefault(
        session_id,
        {"policy_tool_used": False, "support_texts": []}
    )
    retrieved_sources = []

    messages.append(HumanMessage(content=question))

    for step in range(MAX_TOOL_STEPS):

        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:

            raw_answer = str(response.content)

            if context["policy_tool_used"]:
                raw_answer = _enforce_channel_grounding(
                    raw_answer,
                    "\n".join(context["support_texts"]),
                    list(messages),
                    model
                )
                # Keep history consistent with what the user sees.
                messages[-1] = AIMessage(content=raw_answer)

            answer, cited_sources = select_citations(
                raw_answer,
                retrieved_sources
            )

            return {
                "answer": answer,
                "sources": group_citations(cited_sources)
            }

        for call in response.tool_calls:

            tool_name = call["name"]
            tool_args = call["args"]
            tool_call_id = call["id"]

            print(f"\n[Tool called: {tool_name}]")

            selected_tool = tools.get(tool_name)

            if selected_tool is None:
                result = '{"error": "Unknown tool"}'

            else:
                result = selected_tool.invoke(tool_args)

                if tool_name == POLICY_TOOL_NAME:
                    context["policy_tool_used"] = True
                    collect_sources(retrieved_sources, result)
                    collect_evidence_text(context["support_texts"], result)
                else:
                    # Customer data can legitimately contain words such
                    # as merchant names that look like channels.
                    context["support_texts"].append(str(result))

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id
                )
            )

    return {
        "answer": "Unable to complete the request within the tool-call limit.",
        "sources": []
    }


def ask_bankmate(question, session_id="default", model=None, tools=None):
    """Answer a question as plain text, with a Sources block if cited."""

    result = ask_bankmate_with_sources(
        question,
        session_id=session_id,
        model=model,
        tools=tools
    )

    return format_answer_with_sources(result["answer"], result["sources"])
