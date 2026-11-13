
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)

from app.llm import get_assistant_model, tool_map


model = get_assistant_model()

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
""")


def ask_bankmate(question):

    messages = [
        system_message,
        HumanMessage(content=question)
    ]

    for step in range(5):

        response = model.invoke(messages)

        messages.append(response)

        if not response.tool_calls:
            return response.content

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

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id
                )
            )

    return "Unable to complete the request within the tool-call limit."
