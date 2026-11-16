import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.retriever import search_relevant_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MAX_DISTANCE = 1.4

def get_rag_chain():

    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    You are BankMate, a banking policy assistant.

    Answer the user's question using ONLY the policy context below.

    Rules:
    1. Do not invent policy information.
    2. If the context does not contain the answer, clearly say
       that the available policy documents do not contain it.
    3. Explain the relevant procedure clearly.
    4. Mention the policy document ID when available.
    5. These documents are fictional training policies.
       Never present them as actual bank policies.

    POLICY CONTEXT:
    {context}

    USER QUESTION:
    {question}

    ANSWER:
    """)

    def format_documents(documents):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"{doc.page_content}"
            for doc, distance in documents
        )

    def answer_question(question):

        # Retrieve documents with their similarity distances
        results = search_relevant_documents(
            question,
            k=3,
            max_distance=MAX_DISTANCE
        )

        print(f"\nRelevant documents found: {len(results)}")

        # No relevant documents: do not call the LLM
        if not results:
            return {
                "answer": (
                    "I couldn't find relevant information in the "
                    "available policy documents to answer this question."
                ),
                "sources": []
            }

        # Extract documents from the (document, distance) tuples
        documents = [doc for doc, distance in results]

        context = format_documents(results)

        chain = prompt | model | StrOutputParser()

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": answer,
            "sources": documents
        }

    return answer_question


if __name__ == "__main__":

    rag = get_rag_chain()

    test_questions = [
        "What should a customer do if they notice an unauthorized card transaction?",
        "What happens when a refund is approved?",
        "What is the bank's home loan interest rate?"
    ]

    for question in test_questions:

        result = rag(question)

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("\nBANKMATE:", result["answer"])

        print("\nSOURCES:")

        for source in result["sources"]:
            print("-", source.metadata.get("source"))
