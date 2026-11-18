
"""
Citation selection and formatting for BankMate policy answers.

The model is asked to end any answer that used search_banking_policies
with a line such as:

    CITED_POLICIES: POL-001, POL-003
    CITED_POLICIES: NONE

The IDs on that line are validated against the source metadata that the
policy tool actually returned during the current turn. The marker is
the only coverage signal: a missing marker or "NONE" yields no citations,
regardless of how the answer is worded. IDs that were never retrieved
are discarded, so citations cannot be fabricated.

Citations are policy-level. Retrieved section names are not shown,
because the model does not report which sections it relied on.
"""

import json
import re


CITATION_MARKER = "CITED_POLICIES"

SOURCES_HEADER = "Sources (fictional BankMate training policies):"

POLICY_ID_PATTERN = re.compile(r"\bPOL-\d+\b", re.IGNORECASE)

# Matches the marker on its own line or at the end of a sentence; the
# marker and everything after it on that line are removed from display.
MARKER_PATTERN = re.compile(
    r"[ \t>*_`-]*\b" + CITATION_MARKER + r"\b[*_`]*[ \t]*:[ \t]*([^\n]*)",
    re.IGNORECASE
)

NONE_PATTERN = re.compile(r"\bNONE\b", re.IGNORECASE)


def _clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("", "unknown", "none", "null"):
        return ""
    return text


def normalize_source(source):
    """
    Return a cleaned source dict, or None if the source has no valid
    policy ID. Missing optional fields become empty strings.
    """

    if not isinstance(source, dict):
        return None

    policy_id = _clean_text(source.get("policy_id")).upper()

    if not POLICY_ID_PATTERN.fullmatch(policy_id):
        return None

    return {
        "policy_id": policy_id,
        "policy_name": _clean_text(source.get("policy_name")),
        "version": _clean_text(source.get("version")),
        "section": _clean_text(source.get("section")),
        "source_file": _clean_text(source.get("source_file")),
    }


def parse_policy_result(tool_result):
    """
    Parse a search_banking_policies result. Returns the dict only for a
    SUCCESS result, otherwise None (bad JSON, wrong type, no match).
    """

    try:
        data = json.loads(tool_result)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict) or data.get("status") != "SUCCESS":
        return None

    return data


def collect_sources(collected, tool_result):
    """
    Merge sources from a search_banking_policies result (JSON string)
    into `collected`, de-duplicating on (policy_id, section, source_file).
    Invalid or unparseable results are ignored.
    """

    data = parse_policy_result(tool_result)

    if data is None:
        return collected

    existing_keys = {
        (item["policy_id"], item["section"], item["source_file"])
        for item in collected
    }

    for raw_source in data.get("sources") or []:
        source = normalize_source(raw_source)

        if source is None:
            continue

        key = (source["policy_id"], source["section"], source["source_file"])

        if key not in existing_keys:
            existing_keys.add(key)
            collected.append(source)

    return collected


def extract_cited_policy_ids(answer):
    """
    Remove CITED_POLICIES lines from the answer.

    Returns (clean_answer, cited_ids), where cited_ids is None when no
    marker was present, otherwise an ordered, de-duplicated list.
    "NONE" anywhere in a marker overrides any listed IDs (empty list).
    """

    matches = list(MARKER_PATTERN.finditer(answer or ""))

    if not matches:
        return (answer or "").strip(), None

    cited_ids = []
    declared_none = False

    for match in matches:
        if NONE_PATTERN.search(match.group(1)):
            declared_none = True

        for policy_id in POLICY_ID_PATTERN.findall(match.group(1)):
            policy_id = policy_id.upper()
            if policy_id not in cited_ids:
                cited_ids.append(policy_id)

    clean_answer = MARKER_PATTERN.sub("", answer)
    clean_answer = re.sub(r"\n{3,}", "\n\n", clean_answer).strip()

    return clean_answer, ([] if declared_none else cited_ids)


def select_citations(answer, retrieved_sources):
    """
    Decide which retrieved sources support the final answer.

    Returns (clean_answer, cited_sources).

    - No retrieved policy evidence      -> no citations.
    - Marker absent or "NONE"           -> no citations (fail closed).
    - Marker lists IDs                  -> retrieved sources whose
                                           policy_id is listed; unknown
                                           IDs are dropped.
    """

    clean_answer, cited_ids = extract_cited_policy_ids(answer)

    if not retrieved_sources or not cited_ids:
        return clean_answer, []

    cited_sources = [
        source for source in retrieved_sources
        if source["policy_id"] in cited_ids
    ]

    return clean_answer, cited_sources


def _display_case(text):
    # Policy names and sections are stored in upper case; title-case
    # them for display while leaving mixed-case text untouched.
    return text.title() if text.isupper() else text


def group_citations(sources):
    """
    Group section-level sources into one policy-level entry each.
    Section names are intentionally dropped (see module docstring).
    """

    grouped = {}

    for source in sources:
        policy_id = source["policy_id"]

        if policy_id not in grouped:
            grouped[policy_id] = {
                "policy_id": policy_id,
                "policy_name": source.get("policy_name", ""),
                "version": source.get("version", ""),
                "source_files": [],
            }

        entry = grouped[policy_id]

        for field in ("policy_name", "version"):
            if not entry[field] and source.get(field):
                entry[field] = source[field]

        source_file = source.get("source_file", "")
        if source_file and source_file not in entry["source_files"]:
            entry["source_files"].append(source_file)

    return list(grouped.values())


def format_citation(citation):
    """One readable line (without bullet) for a grouped citation."""

    title = citation["policy_id"]

    if citation.get("policy_name"):
        title += " — " + _display_case(citation["policy_name"])

    if citation.get("version"):
        title += " (v" + citation["version"] + ")"

    parts = [title]

    if citation.get("source_files"):
        parts.append("File: " + ", ".join(citation["source_files"]))

    return " | ".join(parts)


def format_sources_block(citations):
    if not citations:
        return ""

    lines = ["- " + format_citation(citation) for citation in citations]

    return SOURCES_HEADER + "\n" + "\n".join(lines)


def format_answer_with_sources(answer, citations):
    block = format_sources_block(citations)

    if not block:
        return answer

    return answer + "\n\n" + block
