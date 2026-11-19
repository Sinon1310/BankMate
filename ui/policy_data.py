"""
Read-only policy views for the Policy Library.

Reuses the backend unchanged:
- documents: app.document_loader.load_policy_documents
- metadata:  app.document_loader.extract_policy_metadata (ID, name and
             version come from each file's own header; nothing is invented)
- sections:  app.document_loader.split_into_sections (the same section
             boundaries the RAG index uses)
- retrieval: app.tools.search_banking_policies, the exact tool the
             assistant calls (same index, k and distance threshold)

The header block that split_into_sections returns as "GENERAL"
(DOCUMENT / DOCUMENT ID / VERSION lines) is metadata, so it is not shown
as a readable section. The "IMPORTANT" section is the fictional-policy
notice and is kept separate so the UI can always display it.
"""

import json
import re

# Re-exported so UI code can keep using policy_data.display_case.
from app.citations import display_case  # noqa: F401
from app.document_loader import (
    HEADER_LINE,
    extract_policy_metadata,
    is_header_only,
    load_policy_documents,
    split_into_sections,
)


NOTICE_SECTION = "IMPORTANT"

SEARCH_SCOPES = ("All text", "Titles & IDs", "Section headings")


class RetrievalError(Exception):
    """The policy index could not be searched (e.g. ChromaDB missing)."""


# ---------- Catalogue ----------

def build_policy(document):
    metadata = extract_policy_metadata(document)
    sections = []
    notice = None

    for name, content in split_into_sections(document.page_content):
        if name == "GENERAL" and is_header_only(content):
            continue
        if name == NOTICE_SECTION:
            notice = content
            continue
        sections.append({"name": name, "content": content})

    return {
        "policy_id": metadata["policy_id"],
        "policy_name": metadata["policy_name"],
        "version": metadata["version"],
        "source_file": metadata["source_file"],
        "sections": sections,
        "notice": notice,
    }


def build_catalog(documents):
    """Policies sorted by ID."""

    return sorted((build_policy(d) for d in documents), key=lambda p: p["policy_id"])


def load_catalog():
    return build_catalog(load_policy_documents())


def find_policy(catalog, policy_id):
    wanted = (policy_id or "").strip().upper()
    return next((p for p in catalog if p["policy_id"].upper() == wanted), None)


# ---------- Keyword search (titles, IDs, headings, text) ----------

def _snippet(text, needle, width=90):
    flat = " ".join(text.split())
    index = flat.lower().find(needle)

    if index < 0:
        return flat[:width] + ("…" if len(flat) > width else "")

    start = max(0, index - width // 3)
    end = min(len(flat), index + len(needle) + width)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


def search_catalog(catalog, query, scope="All text"):
    """
    Case-insensitive keyword search. Returns [(policy, matches)] where
    matches lists {"section", "snippet"} for section-level hits.
    """

    needle = " ".join((query or "").split()).lower()

    if not needle:
        return [(policy, []) for policy in catalog]

    results = []

    for policy in catalog:
        title_hit = needle in policy["policy_id"].lower() or needle in policy["policy_name"].lower()
        matches = []

        if scope in ("All text", "Section headings"):
            for section in policy["sections"]:
                heading_hit = needle in section["name"].lower()
                text_hit = scope == "All text" and needle in section["content"].lower()

                if heading_hit or text_hit:
                    matches.append({
                        "section": section["name"],
                        "snippet": _snippet(section["content"], needle),
                    })

        if (scope in ("All text", "Titles & IDs") and title_hit) or matches:
            results.append((policy, matches))

    return results


# ---------- Content rendering helpers ----------

LIST_ITEM = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")


def section_blocks(content):
    """
    Split section text into paragraphs and numbered lists, re-joining
    the hard-wrapped lines from the source files.
    Returns [("p", text) | ("ol", [item, ...])].
    """

    blocks = []
    paragraph = []
    items = []

    def flush_paragraph():
        if paragraph:
            blocks.append(("p", " ".join(paragraph)))
            paragraph.clear()

    def flush_items():
        if items:
            blocks.append(("ol", list(items)))
            items.clear()

    for raw in content.splitlines():
        line = raw.strip()

        if not line:
            flush_paragraph()
            flush_items()
            continue

        match = LIST_ITEM.match(line)

        if match:
            flush_paragraph()
            items.append(match.group(2))
        elif items:
            # Continuation of the previous list item.
            items[-1] = items[-1] + " " + line
        else:
            paragraph.append(line)

    flush_paragraph()
    flush_items()
    return blocks


def is_header_passage(item):
    """True for an indexed chunk that is only the document header."""

    return item.get("section") == "GENERAL" and is_header_only(item.get("content") or "")


def passage_section_label(item):
    if is_header_passage(item):
        return "Document header"
    return display_case(item.get("section") or "")


def suggested_questions(policy, limit=3):
    """Questions built only from the policy's own name and headings."""

    name = display_case(policy["policy_name"])
    questions = [f"Summarise the {name}."]

    for section in policy["sections"]:
        if section["name"] == "PURPOSE":
            continue
        questions.append(
            f"What does the {name} say about {display_case(section['name']).lower()}?"
        )
        if len(questions) >= limit:
            break

    return questions


# ---------- Semantic retrieval (same tool as the assistant) ----------

def retrieve_passages(question, search_tool=None):
    """
    Run the assistant's policy search tool and return its result:
    {"status": "SUCCESS" | "NO_RELEVANT_POLICY_FOUND", "evidence": [...]}.
    Raises RetrievalError if the index can't be searched.
    """

    if search_tool is None:
        from app.tools import search_banking_policies as search_tool

    question = " ".join((question or "").split())

    if not question:
        return {"status": "NO_RELEVANT_POLICY_FOUND", "evidence": []}

    try:
        raw = search_tool.invoke({"question": question})
        data = json.loads(raw)
    except Exception as error:  # noqa: BLE001 - surfaced to the user
        raise RetrievalError(str(error) or error.__class__.__name__) from error

    evidence = [
        item for item in (data.get("evidence") or [])
        if isinstance(item, dict) and item.get("policy_id") and item.get("content")
    ]

    status = data.get("status")
    if status == "SUCCESS" and not evidence:
        status = "NO_RELEVANT_POLICY_FOUND"

    return {"status": status or "NO_RELEVANT_POLICY_FOUND", "evidence": evidence}
