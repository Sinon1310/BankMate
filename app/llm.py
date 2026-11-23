
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.tools import (
    get_customer_profile,
    get_customer_transactions,
    get_customer_metrics,
    get_customer_disputes,
    search_banking_policies
)

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PLACEHOLDER_KEYS = {"your_groq_api_key_here", "your_groq_api_key", "changeme"}


def validate_groq_api_key(value):
    """
    Raise ValueError with a clear, value-free message if the Groq API key
    is missing or obviously malformed. The key itself is never included
    in any message or log.
    """

    key = (value or "").strip()

    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file "
            "(see .env.example) or to the deployment's secrets."
        )

    if key.lower() in PLACEHOLDER_KEYS or key.lower().startswith("your_"):
        raise ValueError(
            "GROQ_API_KEY still contains the placeholder from .env.example. "
            "Replace it with your real Groq API key."
        )

    if any(ch.isspace() for ch in key) or key[0] in "'\"" or key[-1] in "'\"":
        raise ValueError(
            "GROQ_API_KEY looks malformed (it contains spaces or quotes). "
            "Paste the key exactly as issued, without quotes."
        )

    return key


# Verify the API key before anything can call the model.
validate_groq_api_key(os.getenv("GROQ_API_KEY"))

logger = logging.getLogger(__name__)


def env_number(name, default, cast=float, minimum=None):
    """
    Read a numeric setting from the environment, falling back to
    `default` (with a logged warning) when it is missing or invalid.
    """

    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return default

    try:
        value = cast(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %r", name, raw, default)
        return default

    if minimum is not None and value < minimum:
        logger.warning("Ignoring %s=%r below minimum %r; using %r", name, raw, minimum, default)
        return default

    return value


# Fail fast instead of hanging the UI on a stalled request.
LLM_TIMEOUT_SECONDS = env_number("BANKMATE_LLM_TIMEOUT_SECONDS", 30.0, float, minimum=1)

# One SDK-level retry (it honours Retry-After on 429s). Kept low on
# purpose: further retries are left to the user ("Try again"), so a
# rate-limited key is not hammered with repeated calls.
LLM_MAX_RETRIES = env_number("BANKMATE_LLM_MAX_RETRIES", 1, int, minimum=0)

tools = [
    get_customer_profile,
    get_customer_transactions,
    get_customer_metrics,
    get_customer_disputes,
    search_banking_policies
]


def get_assistant_model():
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    return model.bind_tools(tools)


tool_map = {
    tool.name: tool
    for tool in tools
}
