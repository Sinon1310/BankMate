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
    DETAIL_DESCRIPTIONS,
    collect_evidence_text,
    find_unsupported_channels,
    find_unsupported_details,
    remove_unsupported_examples,
    replace_unsupported_detail_sentences
)
from app.llm import get_assistant_model, tool_map


POLICY_TOOL_NAME = "search_banking_policies"

MAX_TOOL_STEPS = 5

# Created on first use so the module can be imported (e.g. by tests)
# without contacting Groq.
_model = None

# Store conversation history separately for each session.
conversation_sessions = {}

# Per session: whether the policy tool has been used, the retrieved
# policy evidence (supports procedural details), and all supporting text
# (policy evidence + customer tool results; supports channel mentions).
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

POLICY GROUNDING RULES:
38. In policy answers, state only what the retrieved policy text says.
    Do not complete a procedure with steps, outcomes, destinations,
    timeframes, fees or notifications that the text does not state.
    For example, do not say where a refunded amount is credited, or
    how long something takes, unless the retrieved text says so.
39. If the user asks about a detail that the retrieved policy text does
    not cover, say explicitly: "The policy does not specify <detail>."
    Do not infer it from general banking practice.
40. Paraphrase the policy closely; when unsure, use the policy's wording.
41. These rules also apply to follow-up questions answered from
    earlier policy results in the conversation.

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


def _enforce_grounding(answer, support_text, evidence_text, messages, model,
                       check_details=True):
    """
    Check an answer that may rely on policy content:
    - contact channels absent from the supporting text, and
    - procedural details (refund destination, timeframe, fees) absent
      from the retrieved policy evidence (only when check_details).

    Any issue triggers one revision request. If problems remain:
    - channels: only example lists / parentheticals naming them are
      removed (sentences kept; a warning is printed);
    - details: each sentence stating one is replaced with an explicit
      "The policy does not specify ..." statement.
    Returns the final answer text (CITED_POLICIES marker preserved).
    """

    channels = find_unsupported_channels(answer, support_text)
    details = find_unsupported_details(answer, evidence_text) if check_details else []

    if not channels and not details:
        return answer

    print(f"\n[Grounding check: unsupported channels {channels} details {details}]")

    original_markers = [
        match.group(0).strip() for match in MARKER_PATTERN.finditer(answer)
    ]

    problems = []
    if channels:
        problems.append(
            "It mentions these contact channels, which the retrieved policy "
            f"documents do not specify: {', '.join(channels)}. Remove them."
        )
    if details:
        described = ", ".join(DETAIL_DESCRIPTIONS[d] for d in details)
        problems.append(
            "It states details the retrieved policy text does not contain "
            f"({described}). Remove those claims. If the user asked about "
            'them, say explicitly "The policy does not specify ..." instead.'
        )

    correction = HumanMessage(content=(
        "Revise your previous answer. " + " ".join(problems) + " "
        "Use only statements found in the retrieved policy text. "
        "Do not call tools. Keep the CITED_POLICIES line."
    ))

    revised = answer

    try:
        response = model.invoke(messages + [correction])

        if not response.tool_calls and str(response.content).strip():
            revised = str(response.content)

    except Exception as error:
        print(f"\n[Grounding check: revision failed: {error}]")

    revised = _ensure_markers(revised, original_markers)

    remaining_details = (
        find_unsupported_details(revised, evidence_text) if check_details else []
    )
    if remaining_details:
        revised = replace_unsupported_detail_sentences(revised, remaining_details)
        print(
            "\n[Grounding check: replaced unsupported details "
            f"{remaining_details} with 'policy does not specify' statements]"
        )

    remaining_channels = find_unsupported_channels(revised, support_text)

    if remaining_channels:
        cleaned = remove_unsupported_examples(revised, remaining_channels)
        body, _ = extract_cited_policy_ids(cleaned)
        revised = cleaned if body else revised

        remaining_channels = find_unsupported_channels(revised, support_text)
        if remaining_channels:
            print(
                "\n[Grounding check: WARNING unsupported channels remain "
                f"{remaining_channels}; answer kept to avoid removing valid content]"
            )

    return _ensure_markers(revised, original_markers)


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
        {"policy_tool_used": False, "support_texts": [], "evidence_texts": []}
    )
    context.setdefault("evidence_texts", [])
    retrieved_sources = []
    policy_tool_this_turn = False
    other_tool_this_turn = False

    messages.append(HumanMessage(content=question))

    for step in range(MAX_TOOL_STEPS):

        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:

            raw_answer = str(response.content)

            if context["policy_tool_used"]:
                # Procedural details are checked on policy turns and on
                # follow-ups answered from memory, but not on turns that
                # only fetched customer data.
                check_details = policy_tool_this_turn or not other_tool_this_turn

                raw_answer = _enforce_grounding(
                    raw_answer,
                    "\n".join(context["support_texts"]),
                    "\n".join(context["evidence_texts"]),
                    list(messages),
                    model,
                    check_details=check_details,
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
                    policy_tool_this_turn = True
                    collect_sources(retrieved_sources, result)
                    collect_evidence_text(context["support_texts"], result)
                    collect_evidence_text(context["evidence_texts"], result)
                else:
                    other_tool_this_turn = True
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
