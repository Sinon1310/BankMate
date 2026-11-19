"""
Chunking tests (Stage 5.3): header-only blocks are not indexed, every
meaningful section and line is, and metadata is complete.
"""

import re
from pathlib import Path

from langchain_core.documents import Document

from app.document_loader import (
    HEADER_LINE,
    is_header_only,
    load_policy_documents,
    split_documents,
)


POLICY_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "policies"
HEADING = re.compile(r"(?m)^([A-Z][A-Z0-9 /&()_-]{2,}):\s*$")


def real_chunks():
    return split_documents(load_policy_documents())


def test_header_only_chunks_are_not_indexed():
    chunks = real_chunks()

    assert not [c for c in chunks if c.metadata["section"] == "GENERAL"]
    assert not [c for c in chunks if is_header_only(c.page_content)]
    assert not [c for c in chunks if "DOCUMENT ID:" in c.page_content]


def test_every_policy_section_remains_indexed():
    chunks = real_chunks()

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        policy_id = re.search(r"DOCUMENT ID:\s*(POL-\d+)", text).group(1)
        indexed = {c.metadata["section"] for c in chunks if c.metadata["policy_id"] == policy_id}
        assert indexed == set(HEADING.findall(text)), path.name


def test_no_policy_content_is_lost():
    chunks = real_chunks()

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        policy_id = re.search(r"DOCUMENT ID:\s*(POL-\d+)", text).group(1)
        indexed = "\n".join(c.page_content for c in chunks if c.metadata["policy_id"] == policy_id)

        for line in text.splitlines():
            line = line.strip()
            if not line or HEADER_LINE.match(line) or HEADING.match(line):
                continue
            assert line in indexed, (path.name, line)


def test_policy_metadata_is_complete_and_correct():
    chunks = real_chunks()

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        policy_id = re.search(r"DOCUMENT ID:\s*(POL-\d+)", text).group(1)
        name = re.search(r"DOCUMENT:\s*(.+)", text).group(1).strip()
        version = re.search(r"VERSION:\s*([\d.]+)", text).group(1)

        for chunk in (c for c in chunks if c.metadata["policy_id"] == policy_id):
            assert chunk.metadata["policy_name"] == name
            assert chunk.metadata["version"] == version
            assert chunk.metadata["source_file"] == path.name
            assert chunk.metadata["chunk_id"].startswith(policy_id + "_")


def test_numbered_lists_are_preserved():
    chunks = real_chunks()
    required = next(c for c in chunks if c.metadata["section"] == "REQUIRED INFORMATION")

    for item in ("1. Customer ID.", "2. Card or account reference.", "6. Reason for raising the dispute."):
        assert item in required.page_content


def test_chunk_ids_are_unique_and_deterministic():
    first = [c.metadata["chunk_id"] for c in real_chunks()]
    second = [c.metadata["chunk_id"] for c in real_chunks()]

    assert first == second
    assert len(first) == len(set(first))


def test_preamble_with_real_content_is_kept():
    doc = Document(page_content=(
        "DOCUMENT: SAMPLE POLICY\nDOCUMENT ID: POL-900\nVERSION: 2.0\n"
        "This preamble explains the scope of the sample policy.\n\n"
        "PURPOSE:\nSample purpose text.\n"
    ), metadata={"source": "/x/sample.txt"})

    chunks = split_documents([doc])
    general = [c for c in chunks if c.metadata["section"] == "GENERAL"]

    assert len(general) == 1
    assert "explains the scope" in general[0].page_content
    assert {c.metadata["section"] for c in chunks} == {"GENERAL", "PURPOSE"}


def test_is_header_only():
    assert is_header_only("DOCUMENT: X\nDOCUMENT ID: POL-1\nVERSION: 1.0")
    assert is_header_only("  document: x\n\nversion: 2")
    assert not is_header_only("DOCUMENT: X\nThis line is content.")
    assert not is_header_only("")
