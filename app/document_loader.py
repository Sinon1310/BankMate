
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = PROJECT_ROOT / "knowledge" / "policies"


def load_policy_documents():
    loader = DirectoryLoader(
        str(POLICY_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} policy documents")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} document chunks")

    return chunks


if __name__ == "__main__":
    documents = load_policy_documents()
    chunks = split_documents(documents)

    for index, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {index + 1} ---")
        print("Source:", chunk.metadata["source"])
        print("Content:", chunk.page_content)
