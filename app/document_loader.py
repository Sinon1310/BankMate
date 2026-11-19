
import logging
from pathlib import Path
import re

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = PROJECT_ROOT / "knowledge" / "policies"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Document header lines ("DOCUMENT:", "DOCUMENT ID:", "VERSION:"). They are
# copied into every chunk's metadata, so a section made only of them is
# metadata, not policy content.
HEADER_LINE = re.compile(r"^\s*(DOCUMENT|DOCUMENT ID|VERSION)\s*:", re.IGNORECASE)


def is_header_only(text):
    """True when every non-blank line is a document header line."""

    lines = [line for line in (text or "").splitlines() if line.strip()]
    return bool(lines) and all(HEADER_LINE.match(line) for line in lines)


def _inside(path, directory):
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def load_policy_documents(policy_dir=None):
    """
    Load the policy .txt files. Files that resolve outside the policy
    folder (e.g. symlinks to elsewhere on disk) are skipped, so only
    policy documents can ever enter the index or the policy pages.
    """

    policy_dir = Path(policy_dir or POLICY_DIR)

    loader = DirectoryLoader(
        str(policy_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = []

    for document in loader.load():
        if _inside(Path(document.metadata.get("source", "")), policy_dir):
            documents.append(document)
        else:
            logging.getLogger(__name__).warning(
                "Skipped policy file outside %s: %s",
                policy_dir.name, Path(document.metadata.get("source", "")).name,
            )

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

            # The header block is already in every chunk's metadata;
            # indexing it on its own only crowds out real content.
            if section_name == "GENERAL" and is_header_only(section_content):
                continue

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
