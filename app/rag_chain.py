
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.retriever import get_retriever
from app.llm import get_assistant_model


def get_rag_chain():
    retriever = get_retriever()

    # Use the underlying Groq chat model without tool binding
    from langchain_groq import ChatGroq

    from app.llm import load_dotenv
    import os

    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    You are BankMate, a banking policy assistant.

    Answer the user's question using ONLY the policy context below.

    Rules:
    1. Do not invent policy information.
    2. If the answer is not in the context, say that the
       available policy documents do not contain the answer.
    3. Explain the relevant procedure clearly.
    4. Mention the policy document ID when available.
    5. These documents are fictional training policies.
       Do not present them as actual bank policies.

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
            for doc in documents
        )

    def answer_question(question):
        documents = retriever.invoke(question)

        context = format_documents(documents)

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

    question = "What should a customer do if they notice an unauthorized card transaction?"

    result = rag(question)

    print("\nQUESTION:", question)
    print("\nBANKMATE:", result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print("-", source.metadata.get("source"))
