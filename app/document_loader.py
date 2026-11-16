
from pathlib import Path
import re

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = PROJECT_ROOT / "knowledge" / "policies"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


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


def extract_policy_metadata(document):
    content = document.page_content
    filename = Path(document.metadata.get("source", "")).name

    policy_id_match = re.search(
        r"DOCUMENT ID:\s*(POL-\d+)", content
    )

    version_match = re.search(
        r"VERSION:\s*([\d.]+)", content
    )

    title_match = re.search(
        r"DOCUMENT:\s*(.+)", content
    )

    metadata = {
        "policy_id": (
            policy_id_match.group(1)
            if policy_id_match else "UNKNOWN"
        ),
        "version": (
            version_match.group(1)
            if version_match else "UNKNOWN"
        ),
        "policy_name": (
            title_match.group(1).strip()
            if title_match else filename
        ),
        "source_file": filename
    }

    return metadata


def split_into_sections(content):
    """
    Split policy content using uppercase section headings.
    A heading must occupy its own line and end with a colon.
    """

    heading_pattern = re.compile(
        r"(?m)^([A-Z][A-Z0-9 /&()_-]{2,}):\s*$"
    )

    matches = list(heading_pattern.finditer(content))

    sections = []

    if not matches:
        return [("GENERAL", content.strip())]

    # Preserve any content before the first heading.
    preamble = content[:matches[0].start()].strip()

    if preamble:
        sections.append(("GENERAL", preamble))

    for index, match in enumerate(matches):

        section_name = match.group(1).strip()

        start = match.end()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )

        section_content = content[start:end].strip()

        if section_content:
            sections.append((section_name, section_content))

    return sections


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        add_start_index=True
    )

    all_chunks = []

    for document in documents:

        base_metadata = extract_policy_metadata(document)

        sections = split_into_sections(document.page_content)

        for section_name, section_content in sections:

            section_document = Document(
                page_content=section_content,
                metadata={
                    **document.metadata,
                    **base_metadata,
                    "section": section_name
                }
            )

            section_chunks = splitter.split_documents(
                [section_document]
            )

            for index, chunk in enumerate(section_chunks):

                chunk.metadata["chunk_id"] = (
                    f"{base_metadata['policy_id']}_"
                    f"{section_name.replace(' ', '_')}_"
                    f"{index + 1:03d}"
                )

                all_chunks.append(chunk)

    print(
        f"Created {len(all_chunks)} section-aware document chunks"
    )

    return all_chunks


if __name__ == "__main__":

    documents = load_policy_documents()
    chunks = split_documents(documents)

    for index, chunk in enumerate(chunks):

        print("\n" + "=" * 75)
        print(f"CHUNK {index + 1}")
        print("=" * 75)

        print("Chunk ID:", chunk.metadata["chunk_id"])
        print("Policy ID:", chunk.metadata["policy_id"])
        print("Policy Name:", chunk.metadata["policy_name"])
        print("Version:", chunk.metadata["version"])
        print("Section:", chunk.metadata["section"])
        print("Source:", chunk.metadata["source_file"])
        print("Start Index:", chunk.metadata.get("start_index"))

        print("\nContent:")
        print(chunk.page_content)
