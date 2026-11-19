import json

import pytest

from app.citations import (
    SOURCES_HEADER,
    display_case,
    collect_sources,
    extract_cited_policy_ids,
    format_answer_with_sources,
    format_citation,
    format_sources_block,
    group_citations,
    normalize_source,
    select_citations,
)


def policy_result(*sources, status="SUCCESS"):
    return json.dumps({"status": status, "sources": list(sources)})


DISPUTE = {
    "policy_id": "POL-001",
    "policy_name": "CARD DISPUTE POLICY",
    "version": "1.0",
    "section": "REQUIRED INFORMATION",
    "source_file": "card_dispute_policy.txt",
    "distance": 0.71,
}

REFUND = {
    "policy_id": "POL-002",
    "policy_name": "TRANSACTION REFUND POLICY",
    "version": "1.0",
    "section": "PROCESSING",
    "source_file": "refund_policy.txt",
    "distance": 0.65,
}

SECURITY = {
    "policy_id": "POL-003",
    "policy_name": "TRANSACTION SECURITY POLICY",
    "version": "1.0",
    "section": "REPORTING",
    "source_file": "transaction_security_policy.txt",
    "distance": 1.1,
}


def collected(*sources):
    result = []
    collect_sources(result, policy_result(*sources))
    return result


# --- normalize / collect ---------------------------------------------------

def test_normalize_rejects_missing_or_invalid_policy_id():
    assert normalize_source({"policy_name": "X"}) is None
    assert normalize_source({"policy_id": "Unknown"}) is None
    assert normalize_source({"policy_id": "UNKNOWN"}) is None
    assert normalize_source({"policy_id": "ABC-1"}) is None
    assert normalize_source("POL-001") is None
    assert normalize_source(None) is None


def test_normalize_blanks_missing_optional_fields():
    source = normalize_source({"policy_id": "pol-002", "section": "Unknown"})
    assert source == {
        "policy_id": "POL-002",
        "policy_name": "",
        "version": "",
        "section": "",
        "source_file": "",
    }


def test_collect_deduplicates_sources():
    duplicate = dict(DISPUTE, distance=0.9)
    result = []
    collect_sources(result, policy_result(DISPUTE, duplicate))
    collect_sources(result, policy_result(DISPUTE))
    assert len(result) == 1


def test_collect_ignores_no_policy_found_and_bad_json():
    result = []
    collect_sources(result, policy_result(DISPUTE, status="NO_RELEVANT_POLICY_FOUND"))
    collect_sources(result, "not json")
    collect_sources(result, None)
    collect_sources(result, json.dumps(["list"]))
    collect_sources(result, json.dumps({"status": "SUCCESS", "sources": None}))
    assert result == []


def test_collect_skips_sources_with_invalid_metadata():
    result = collected({"policy_name": "No ID"}, REFUND)
    assert [s["policy_id"] for s in result] == ["POL-002"]


# --- marker extraction -----------------------------------------------------

def test_extract_marker_strips_line_and_parses_ids():
    answer, ids = extract_cited_policy_ids(
        "Refunds are credited.\n\nCITED_POLICIES: POL-002, pol-001, POL-002"
    )
    assert answer == "Refunds are credited."
    assert ids == ["POL-002", "POL-001"]


def test_extract_marker_none():
    answer, ids = extract_cited_policy_ids("Not covered.\nCITED_POLICIES: NONE")
    assert answer == "Not covered."
    assert ids == []


def test_extract_marker_tolerates_markdown_formatting():
    answer, ids = extract_cited_policy_ids("Answer.\n**CITED_POLICIES:** POL-003")
    assert answer == "Answer."
    assert ids == ["POL-003"]


def test_extract_inline_marker_at_end_of_sentence():
    answer, ids = extract_cited_policy_ids(
        "Refunds are credited. CITED_POLICIES: POL-002"
    )
    assert answer == "Refunds are credited."
    assert ids == ["POL-002"]


def test_none_overrides_listed_ids():
    answer, ids = extract_cited_policy_ids("Answer.\nCITED_POLICIES: NONE, POL-002")
    assert answer == "Answer."
    assert ids == []


def test_extract_without_marker_returns_none():
    answer, ids = extract_cited_policy_ids("Customer C101 is Arjun Mehta.")
    assert answer == "Customer C101 is Arjun Mehta."
    assert ids is None


# --- selection -------------------------------------------------------------

def test_relevant_sources_only_cited_policy_kept():
    retrieved = collected(REFUND, DISPUTE, SECURITY)
    answer, cited = select_citations(
        "Approved refunds are processed.\nCITED_POLICIES: POL-002", retrieved
    )
    assert answer == "Approved refunds are processed."
    assert [s["policy_id"] for s in cited] == ["POL-002"]


def test_irrelevant_retrieved_sources_not_attached():
    retrieved = collected(SECURITY)
    _, cited = select_citations(
        "Refunds are credited.\nCITED_POLICIES: NONE", retrieved
    )
    assert cited == []


def test_fabricated_policy_id_is_discarded():
    retrieved = collected(REFUND)
    _, cited = select_citations("Answer.\nCITED_POLICIES: POL-002, POL-999", retrieved)
    assert [s["policy_id"] for s in cited] == ["POL-002"]


def test_cited_id_without_retrieval_gives_no_sources():
    _, cited = select_citations("Answer.\nCITED_POLICIES: POL-001", [])
    assert cited == []


@pytest.mark.parametrize("wording", [
    "The available policy documents do not contain that information.",
    "I'm sorry, but I couldn't find any policy documents on home loan rates.",
    "Home loan interest rates aren't something I can help with here.",
    "That topic is outside BankMate's training material.",
])
def test_no_coverage_signal_blocks_citations_regardless_of_wording(wording):
    # Evidence was retrieved, but the model signals it relied on none.
    retrieved = collected(SECURITY, REFUND)
    answer, cited = select_citations(wording + "\nCITED_POLICIES: NONE", retrieved)
    assert cited == []
    assert answer == wording


def test_missing_marker_gives_no_sources_even_if_ids_named():
    # Fail closed: naming a policy in prose is not a coverage signal.
    retrieved = collected(REFUND, SECURITY)
    _, cited = select_citations("Per POL-002, refunds are credited.", retrieved)
    assert cited == []


def test_marker_without_any_policy_id_gives_no_sources():
    retrieved = collected(REFUND)
    _, cited = select_citations("Answer.\nCITED_POLICIES: the refund policy", retrieved)
    assert cited == []


def test_missing_marker_and_no_named_ids_gives_no_sources():
    retrieved = collected(REFUND, SECURITY)
    _, cited = select_citations("Refunds are credited.", retrieved)
    assert cited == []


# --- grouping / formatting -------------------------------------------------

def test_group_merges_sections_into_one_policy_level_citation():
    second_section = dict(REFUND, section="VALIDATION")
    groups = group_citations(collected(REFUND, second_section, REFUND))
    assert len(groups) == 1
    assert "sections" not in groups[0]
    assert groups[0]["source_files"] == ["refund_policy.txt"]


def test_formatted_citation_does_not_list_retrieved_sections():
    tracking = dict(REFUND, section="TRACKING")
    block = format_sources_block(group_citations(collected(REFUND, tracking)))
    assert "Section" not in block
    assert "Processing" not in block and "Tracking" not in block


def test_format_citation_is_readable():
    line = format_citation(group_citations(collected(REFUND))[0])
    assert line == (
        "POL-002 — Transaction Refund Policy (v1.0) | File: refund_policy.txt"
    )


def test_format_citation_with_missing_metadata():
    groups = group_citations(collected({"policy_id": "POL-004"}))
    assert format_citation(groups[0]) == "POL-004"


def test_sources_block_keeps_fictional_disclaimer():
    block = format_sources_block(group_citations(collected(DISPUTE)))
    assert block.startswith(SOURCES_HEADER)
    assert "fictional BankMate training policies" in block
    assert block.count("\n- ") == 1


def test_no_sources_block_when_no_citations():
    assert format_sources_block([]) == ""
    assert format_answer_with_sources("Answer.", []) == "Answer."


# ---------- Shared display rule ----------

@pytest.mark.parametrize("raw, shown", [
    ("TRANSACTION REFUND POLICY", "Transaction Refund Policy"),
    ("STATUS TRACKING", "Status Tracking"),
    ("Mixed Case", "Mixed Case"),
    ("already title", "already title"),
    ("POL-002", "Pol-002"),
    ("", ""),
    (None, None),
])
def test_display_case(raw, shown):
    assert display_case(raw) == shown


def test_ui_uses_the_single_shared_display_rule():
    import ui.components as components
    import ui.policy_data as policy_data

    assert components.display_case is display_case
    assert policy_data.display_case is display_case
