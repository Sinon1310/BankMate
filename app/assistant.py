import json 
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)

from app.llm import get_assistant_model, tool_map


model = get_assistant_model()

# Store conversation history separately for each session.
conversation_sessions = {}


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
21. Clearly distinguish policy requirements from
    general suggestions.
22. Never claim that a policy guarantees a particular outcome.
23. If a tool returns NO_RELEVANT_POLICY_FOUND,
    do not attempt to answer using general knowledge.
24. Resolve references such as "he", "she", "they",
    "their" and "that customer" using conversation history.

25. Before calling search_banking_policies, convert
    the user's question into a self-contained search query.

26. Preserve the original meaning when rewriting queries.
    Do not introduce new facts or assumptions.

27. Use the rewritten query for policy retrieval,
    while answering the original user question.

28. Do not include customer-specific personal details
    in policy search queries unless they are necessary.

CONVERSATION MEMORY RULES:
24. Use previous messages to understand references such as
    "he", "she", "they", "that customer" and "their transactions".
25. Maintain customer context within the current session.
26. Never assume a customer ID if the conversation does
    not establish one.
27. If multiple customers are being discussed and the
    reference is ambiguous, ask for clarification.
28. Previous conversation context does not replace tool calls.
    Retrieve fresh customer data when needed.
29. Do not mix information belonging to different customers.

""")



def ask_bankmate(question, session_id="default"):

    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = [system_message]

    messages = conversation_sessions[session_id]
    collected_policy_sources = []

    messages.append(HumanMessage(content=question))

    for step in range(5):

        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:

            final_answer = str(response.content)

            if collected_policy_sources:
                citation_lines = []

                for source in collected_policy_sources:
                    citation = (
                        f"- {source.get('policy_id', 'Unknown')}: "
                        f"{source.get('policy_name', 'Unknown')} | "
                        f"Section: {source.get('section', 'Unknown')} | "
                        f"File: {source.get('source_file', 'Unknown')}"
                    )
                    citation_lines.append(citation)

                final_answer += (
                    "\n\nSources (fictional BankMate training policies):\n"
                    + "\n".join(citation_lines)
                )

            return final_answer

        for call in response.tool_calls:

            tool_name = call["name"]
            tool_args = call["args"]
            tool_call_id = call["id"]

            print(f"\n[Tool called: {tool_name}]")

            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:
                result = '{"error": "Unknown tool"}'

            else:
                result = selected_tool.invoke(tool_args)

                if tool_name == "search_banking_policies":
                    try:
                        policy_data = json.loads(result)

                        if policy_data.get("status") == "SUCCESS":

                            for source in policy_data.get("sources", []):

                                source_key = (
                                    source.get("policy_id"),
                                    source.get("section"),
                                    source.get("source_file")
                                )

                                existing_keys = [
                                    (
                                        item.get("policy_id"),
                                        item.get("section"),
                                        item.get("source_file")
                                    )
                                    for item in collected_policy_sources
                                ]

                                if source_key not in existing_keys:
                                    collected_policy_sources.append(source)

                    except (json.JSONDecodeError, TypeError):
                        pass

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id
                )
            )

    return "Unable to complete the request within the tool-call limit."
