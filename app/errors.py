"""
Failure classification for the conversation engine.

Provider and tool exceptions are mapped to an AssistantError carrying a
`kind` (for code and logs) and a friendly `user_message` (safe to show).
Raw provider messages never reach users; they are kept on the exception
(`detail`, and the chained `__cause__`) for developer logs.
"""

RATE_LIMITED = "rate_limited"
TIMEOUT = "timeout"
CONNECTION = "connection"
AUTHENTICATION = "authentication"
SERVICE = "service_error"
TOOL_FAILURE = "tool_failure"
INDEX_UNAVAILABLE = "index_unavailable"
CONFLICT = "conflict"
IN_PROGRESS = "in_progress"
INVALID_INPUT = "invalid_input"
INPUT_TOO_LONG = "input_too_long"
UNEXPECTED = "unexpected"

USER_MESSAGES = {
    RATE_LIMITED: (
        "BankMate is receiving too many requests right now. "
        "Please wait a moment and try again."
    ),
    TIMEOUT: (
        "The AI service took too long to respond. Please try again."
    ),
    CONNECTION: (
        "BankMate couldn't reach the AI service. Check the connection "
        "and try again."
    ),
    AUTHENTICATION: (
        "The AI service isn't configured correctly, so BankMate can't "
        "answer right now. Please contact the administrator."
    ),
    SERVICE: (
        "The AI service had a problem answering. Please try again."
    ),
    TOOL_FAILURE: (
        "A data lookup failed while answering. Please try again."
    ),
    INDEX_UNAVAILABLE: (
        "BankMate's policy index isn't available right now, so policy "
        "questions can't be answered. Customer and transaction questions "
        "still work."
    ),
    CONFLICT: (
        "This conversation was updated by another reply while BankMate "
        "was answering, so this answer wasn't saved. Please try again."
    ),
    IN_PROGRESS: (
        "BankMate is still answering this message. Please wait a moment."
    ),
    INVALID_INPUT: (
        "Please type a question for BankMate."
    ),
    INPUT_TOO_LONG: (
        "That message is too long. Please shorten it to under 2,000 "
        "characters and try again."
    ),
    UNEXPECTED: (
        "Something went wrong while answering. Your conversation is "
        "intact — please try again."
    ),
}


class AssistantError(Exception):
    """A failed assistant response, safe to surface to users."""

    def __init__(self, kind, detail="", retry_after=None):
        self.kind = kind
        self.detail = detail
        self.retry_after = retry_after
        super().__init__(f"{kind}: {detail}" if detail else kind)

    @property
    def user_message(self):
        message = USER_MESSAGES.get(self.kind, USER_MESSAGES[UNEXPECTED])

        if self.kind == RATE_LIMITED and self.retry_after:
            message = (
                "BankMate is receiving too many requests right now. "
                f"Please try again in about {self.retry_after} seconds."
            )

        return message


def _retry_after_seconds(error):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}

    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None

    return max(1, int(round(value))) if value > 0 else None


def classify_provider_error(error):
    """Map an exception raised while calling the LLM to an AssistantError."""

    try:
        import groq
    except ImportError:  # pragma: no cover - groq ships with langchain-groq
        groq = None

    from app.security import redact_secrets

    status = getattr(error, "status_code", None)
    detail = redact_secrets(str(error) or error.__class__.__name__)

    if (groq and isinstance(error, groq.RateLimitError)) or status == 429:
        return AssistantError(RATE_LIMITED, detail, _retry_after_seconds(error))

    if (groq and isinstance(error, groq.APITimeoutError)) or isinstance(error, TimeoutError):
        return AssistantError(TIMEOUT, detail)

    if groq and isinstance(error, groq.APIConnectionError):
        return AssistantError(CONNECTION, detail)

    if (groq and isinstance(error, (groq.AuthenticationError, groq.PermissionDeniedError))) \
            or status in (401, 403):
        return AssistantError(AUTHENTICATION, detail)

    if (groq and isinstance(error, groq.APIStatusError)) or isinstance(status, int):
        return AssistantError(SERVICE, detail)

    return AssistantError(UNEXPECTED, detail)
