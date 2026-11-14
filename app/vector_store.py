
from pathlib import Path

from langchain_chroma import Chroma

from app.document_loader import (
    load_policy_documents,
    split_documents
)
from app.embeddings import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def create_vector_store():
    documents = load_policy_documents()
    chunks = split_documents(documents)

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="bankmate_policies"
    )

    print(f"\nStored {len(chunks)} chunks in ChromaDB")
    print(f"Vector database location: {CHROMA_DIR}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()
