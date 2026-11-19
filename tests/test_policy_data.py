"""
Tests for ui/policy_data.py.

The catalogue is checked against the raw policy files (parsed here
independently), so the UI can never show an invented ID, name, version
or section. Retrieval tests use a fake search tool: no ChromaDB needed.
"""

import json
import re
from pathlib import Path

import pytest

from ui import policy_data as pdata


POLICY_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "policies"
HEADING = re.compile(r"(?m)^([A-Z][A-Z0-9 /&()_-]{2,}):\s*$")


def raw_policies():
    """Independent parse of each file's header and headings."""

    parsed = {}

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        policy_id = re.search(r"DOCUMENT ID:\s*(POL-\d+)", text).group(1)
        parsed[policy_id] = {
            "name": re.search(r"DOCUMENT:\s*(.+)", text).group(1).strip(),
            "version": re.search(r"VERSION:\s*([\d.]+)", text).group(1),
            "file": path.name,
            "headings": HEADING.findall(text),
            "text": text,
        }

    return parsed


@pytest.fixture(scope="module")
def catalog():
    return pdata.load_catalog()


# ---------- Catalogue integrity ----------

def test_catalog_matches_files_exactly(catalog):
    raw = raw_policies()

    assert [p["policy_id"] for p in catalog] == sorted(raw)

    for policy in catalog:
        expected = raw[policy["policy_id"]]
        assert policy["policy_name"] == expected["name"]
        assert policy["version"] == expected["version"]
        assert policy["source_file"] == expected["file"]


def test_sections_follow_document_headings_and_exclude_header(catalog):
    raw = raw_policies()

    for policy in catalog:
        headings = [h for h in raw[policy["policy_id"]]["headings"] if h != pdata.NOTICE_SECTION]
        assert [s["name"] for s in policy["sections"]] == headings
        assert all(s["name"] != "GENERAL" for s in policy["sections"])
        assert all(s["content"].strip() for s in policy["sections"])


def test_no_policy_text_is_lost(catalog):
    """Every non-header, non-heading line appears in a section or notice."""

    raw = raw_policies()

    for policy in catalog:
        shown = " ".join([s["content"] for s in policy["sections"]] + [policy["notice"] or ""])

        for line in raw[policy["policy_id"]]["text"].splitlines():
            line = line.strip()
            if not line or HEADING.match(line) or pdata.HEADER_LINE.match(line):
                continue
            assert line in shown, (policy["policy_id"], line)


def test_fictional_notice_is_kept_separately(catalog):
    for policy in catalog:
        assert policy["notice"] and "fictional" in policy["notice"].lower()


def test_find_policy_is_case_insensitive(catalog):
    assert pdata.find_policy(catalog, "pol-002")["policy_id"] == "POL-002"
    assert pdata.find_policy(catalog, "POL-999") is None
    assert pdata.find_policy(catalog, "") is None


# ---------- Keyword search ----------

def ids(results):
    return [policy["policy_id"] for policy, _ in results]


def test_empty_query_returns_all(catalog):
    assert ids(pdata.search_catalog(catalog, "  ")) == [p["policy_id"] for p in catalog]


def test_all_text_search_finds_body_text_with_section_snippet(catalog):
    results = pdata.search_catalog(catalog, "OTP", "All text")

    assert ids(results) == ["POL-003"]
    match = results[0][1][0]
    assert match["section"] == "CUSTOMER PROTECTION"
    assert "OTP" in match["snippet"]


def test_titles_scope_ignores_body_text(catalog):
    assert ids(pdata.search_catalog(catalog, "otp", "Titles & IDs")) == []
    assert ids(pdata.search_catalog(catalog, "pol-003", "Titles & IDs")) == ["POL-003"]
    assert ids(pdata.search_catalog(catalog, "refund", "Titles & IDs")) == ["POL-002"]


def test_heading_scope_matches_section_names(catalog):
    results = pdata.search_catalog(catalog, "investigation", "Section headings")

    assert ids(results) == ["POL-001", "POL-003"]
    assert all(m["section"] == "INVESTIGATION" for _, matches in results for m in matches)


def test_search_with_no_match(catalog):
    assert pdata.search_catalog(catalog, "mortgage", "All text") == []


# ---------- Rendering helpers ----------

def test_section_blocks_rejoin_wrapped_lines_and_lists():
    content = (
        "The customer should provide:\n1. Customer ID.\n2. Card or account\nreference.\n\n"
        "A wrapped\nparagraph."
    )

    assert pdata.section_blocks(content) == [
        ("p", "The customer should provide:"),
        ("ol", ["Customer ID.", "Card or account reference."]),
        ("p", "A wrapped paragraph."),
    ]


def test_suggested_questions_only_use_policy_headings(catalog):
    policy = pdata.find_policy(catalog, "POL-002")
    questions = pdata.suggested_questions(policy)

    assert questions[0] == "Summarise the Transaction Refund Policy."
    headings = {s["name"].lower() for s in policy["sections"]}
    for question in questions[1:]:
        topic = question.rsplit("about ", 1)[1].rstrip("?")
        assert topic in headings


def test_display_case():
    assert pdata.display_case("TRANSACTION REFUND POLICY") == "Transaction Refund Policy"
    assert pdata.display_case("Mixed Case") == "Mixed Case"


# ---------- Retrieval (fake tool) ----------

class FakeTool:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        if self.error:
            raise self.error
        return self.result


def test_retrieve_uses_tool_and_returns_evidence():
    tool = FakeTool(json.dumps({
        "status": "SUCCESS",
        "evidence": [
            {"policy_id": "POL-002", "section": "PROCESSING", "content": "Once approved…", "distance": 0.38},
            {"policy_id": "", "content": "missing id"},
            {"policy_id": "POL-002", "content": ""},
            "not a dict",
        ],
    }))

    result = pdata.retrieve_passages("  refund   approved ", search_tool=tool)

    assert tool.calls == [{"question": "refund approved"}]
    assert result["status"] == "SUCCESS"
    assert [e["section"] for e in result["evidence"]] == ["PROCESSING"]


def test_retrieve_no_relevant_policy():
    tool = FakeTool(json.dumps({"status": "NO_RELEVANT_POLICY_FOUND", "evidence": []}))
    assert pdata.retrieve_passages("home loan", search_tool=tool)["status"] == "NO_RELEVANT_POLICY_FOUND"


def test_retrieve_success_without_usable_evidence_is_no_result():
    tool = FakeTool(json.dumps({"status": "SUCCESS", "evidence": [{"policy_id": "POL-001"}]}))
    assert pdata.retrieve_passages("x", search_tool=tool) == {
        "status": "NO_RELEVANT_POLICY_FOUND", "evidence": []
    }


def test_retrieve_empty_question_skips_tool():
    tool = FakeTool("{}")
    assert pdata.retrieve_passages("   ", search_tool=tool)["evidence"] == []
    assert tool.calls == []


@pytest.mark.parametrize("tool", [
    FakeTool(error=RuntimeError("chroma unavailable")),
    FakeTool(result="not json"),
])
def test_retrieve_failures_raise_retrieval_error(tool):
    with pytest.raises(pdata.RetrievalError):
        pdata.retrieve_passages("refund", search_tool=tool)


def test_header_passages_are_labelled():
    header = {"section": "GENERAL", "content": "DOCUMENT: X\nDOCUMENT ID: POL-001\nVERSION: 1.0"}
    body = {"section": "GENERAL", "content": "Some preamble text."}

    assert pdata.passage_section_label(header) == "Document header"
    assert pdata.passage_section_label(body) == "General"
    assert pdata.passage_section_label({"section": "PROCESSING"}) == "Processing"
