
from app.vector_store import CHROMA_DIR
from app.embeddings import get_embedding_model

from langchain_chroma import Chroma


def get_retriever():
    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name="bankmate_policies",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever


if __name__ == "__main__":
    retriever = get_retriever()

    question = "What should a customer do if they notice an unauthorized card transaction?"

    results = retriever.invoke(question)

    print("\nQUESTION:", question)
    print("\nRETRIEVED DOCUMENTS:")

    for index, document in enumerate(results, start=1):
        print(f"\n--- RESULT {index} ---")
        print("Source:", document.metadata["source"])
        print("Content:", document.page_content)
