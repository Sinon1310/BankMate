
from dotenv import load_dotenv
from app.assistant import ask_bankmate


load_dotenv()

print("\nBANKMATE — Banking Intelligence Assistant")
print("Ask about customers C101, C102 or C103.")
print("Type 'exit' to quit.\n")

# One session for this terminal conversation.
session_id = "terminal_session_1"

while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("BankMate: Goodbye!")
        break

    if not question:
        continue

    try:
        answer = ask_bankmate(
            question,
            session_id=session_id
        )

        print("\nBankMate:", answer, "\n")

    except Exception as error:
        # AssistantError has a user-safe message; never print raw provider
        # errors (details are logged by app.assistant).
        message = getattr(error, "user_message", None) or (
            "Something went wrong while answering. Please try again."
        )
        print(f"\nBankMate: {message}\n")
