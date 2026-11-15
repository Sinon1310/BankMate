
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.tools import (
    get_customer_profile,
    get_customer_transactions,
    get_customer_metrics,
    search_banking_policies
)

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Verify API key exists
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

tools = [
    get_customer_profile,
    get_customer_transactions,
    get_customer_metrics,
    search_banking_policies
]


def get_assistant_model():
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )
    return model.bind_tools(tools)


tool_map = {
    tool.name: tool
    for tool in tools
}
