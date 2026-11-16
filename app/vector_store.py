
from pathlib import Path

import chromadb
from langchain_chroma import Chroma

from app.document_loader import (
    load_policy_documents,
    split_documents
)
from app.embeddings import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "bankmate_policies"


def create_vector_store():

    print("\nLoading policy documents...")

    documents = load_policy_documents()
    chunks = split_documents(documents)

    print("\nGenerating embeddings...")

    embeddings = get_embedding_model()

    # Open the existing persistent Chroma database.
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Remove the previous collection to avoid stale chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted previous collection.")
    except Exception:
        print("No previous collection found. Creating a new one.")

    print("\nCreating new vector store...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME
    )

    print("\nVector store creation successful!")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database location: {CHROMA_DIR}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()
