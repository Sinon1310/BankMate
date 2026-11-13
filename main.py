from dotenv import load_dotenv
from app.assistant import ask_bankmate

load_dotenv()

print("\n🏦 BANKMATE — Banking Intelligence Assistant")
print("Ask about customers C101, C102 or C103.")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("BankMate: Goodbye!")
        break

    if not question:
        continue

    try:
        answer = ask_bankmate(question)
        print("\nBankMate:", answer, "\n")

    except Exception as error:
        print(f"\nSomething went wrong: {error}\n")