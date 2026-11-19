"""
Policy Intelligence: browse, search and read the fictional policy
documents, retrieve passages from the RAG index, and hand questions to
the assistant (which answers with validated citations).

Views
-----
- Library (default): keyword search over titles, IDs, headings and text,
  policy cards, and semantic passage search using the assistant's own
  retrieval tool.
- Detail (?policy=POL-xxx, or chosen in-app): metadata, contents,
  readable sections, the fictional-policy notice, and ask actions.

Nothing here invents metadata: IDs, names and versions come from each
document's header via the backend loader.
"""

import logging
import re
from html import escape

import streamlit as st

from ui import components as ui
from ui import index_status
from ui import policy_data as pdata
from ui import state


logger = logging.getLogger(__name__)

POLICY_ID_FORMAT = re.compile(r"^POL-\d{1,6}$")

LAST_QUERY = "policy_search_last_query"
LAST_RESULT = "policy_search_last_result"


@st.cache_data(show_spinner=False)
def _catalog():
    return pdata.load_catalog()


def _open(policy_id):
    state.select_policy(policy_id)
    st.query_params["policy"] = policy_id


def _close():
    state.clear_policy()
    if "policy" in st.query_params:
        del st.query_params["policy"]


def _ask(prompt):
    ui.go_to_assistant(prompt)


def _anchor(policy_id, section_name):
    slug = "".join(ch if ch.isalnum() else "-" for ch in section_name.lower()).strip("-")
    return f"{policy_id.lower()}-{slug}"


def _render_blocks(content):
    parts = []

    for kind, value in pdata.section_blocks(content):
        if kind == "ol":
            parts.append("<ol>" + "".join(f"<li>{escape(item)}</li>" for item in value) + "</ol>")
        else:
            parts.append(f"<p>{escape(value)}</p>")

    return "".join(parts)


def _policy_id_badge(policy_id):
    return f'<span class="bm-source-id">{escape(policy_id)}</span>'


# ---------- Load catalogue ----------

try:
    catalog = _catalog()
    load_error = None
except Exception as error:  # noqa: BLE001 - details go to the server log only
    logger.error("Policy documents could not be loaded: %r", error)
    catalog = []
    load_error = "unavailable"

# Resolve selection: URL first (deep links), then in-app selection.
requested = st.query_params.get("policy")
if requested and pdata.find_policy(catalog, requested):
    state.select_policy(pdata.find_policy(catalog, requested)["policy_id"])

selected = pdata.find_policy(catalog, state.selected_policy()) if state.selected_policy() else None

if selected and st.query_params.get("policy") != selected["policy_id"]:
    st.query_params["policy"] = selected["policy_id"]


# =====================================================================
# Detail view
# =====================================================================

def render_detail(policy):
    name = pdata.display_case(policy["policy_name"])

    if st.button("All policies", icon=":material/arrow_back:", key="policy-back", type="tertiary"):
        _close()
        st.rerun()

    with st.container(key="bm-panel-policy-header"):
        ui.html(
            '<div class="bm-policy-head">'
            f"{_policy_id_badge(policy['policy_id'])}"
            f'<h1 class="bm-policy-title">{escape(name)}</h1>'
            '<div class="bm-profile-meta">'
            f'<span>{ui.icon("history_edu")}Version {escape(policy["version"])}</span>'
            f'<span>{ui.icon("description")}{escape(policy["source_file"])}</span>'
            f'<span>{ui.icon("segment")}{len(policy["sections"])} sections</span>'
            "</div></div>"
        )

        ui.html('<div class="bm-composer-hint bm-ask-hint">Ask BankMate about this policy</div>')

        with st.container(horizontal=True, gap="small", key="bm-chiprow-policy"):
            chosen = None
            for index, question in enumerate(pdata.suggested_questions(policy)):
                if ui.prompt_chip(question, f"policy-q-{index}"):
                    chosen = question

        if chosen:
            _ask(chosen)

        with st.container(key="bm-policy-ask"):
            asked = st.chat_input(
                f"Ask a question about {name}…",
                key="policy-ask-input",
                max_chars=1800,   # room for the policy prefix within the 2,000 limit
            )

        if asked:
            _ask(f"Using the {name} ({policy['policy_id']}): {asked}")

    contents_col, body_col = st.columns([1, 3], gap="large")

    with contents_col:
        links = "".join(
            f'<a href="#{_anchor(policy["policy_id"], s["name"])}">{escape(pdata.display_case(s["name"]))}</a>'
            for s in policy["sections"]
        )
        ui.html(f'<nav class="bm-toc" aria-label="Sections"><div class="bm-toc-label">Contents</div>{links}</nav>')

    with body_col:
        if not policy["sections"]:
            ui.empty_state("article", "No readable sections", "This document has no section headings to display.")

        for section in policy["sections"]:
            ui.html(
                f'<section class="bm-policy-section" id="{_anchor(policy["policy_id"], section["name"])}">'
                f'<h2>{escape(pdata.display_case(section["name"]))}</h2>'
                f"{_render_blocks(section['content'])}</section>"
            )

        notice = policy["notice"] or (
            "Fictional training material created for the BankMate project. "
            "Not an actual bank policy."
        )
        ui.html(
            f'<div class="bm-notice">{ui.icon("info")}<div><strong>Fictional policy</strong>'
            f"<p>{escape(' '.join(notice.split()))}</p></div></div>"
        )


# =====================================================================
# Library view
# =====================================================================

def render_policy_card(policy, matches):
    name = pdata.display_case(policy["policy_name"])
    purpose = next((s["content"] for s in policy["sections"] if s["name"] == "PURPOSE"), "")
    purpose = " ".join(purpose.split())

    with st.container(key=f"bm-card-policy-{policy['policy_id']}"):
        match_html = "".join(
            f'<div class="bm-match"><span>{escape(pdata.display_case(m["section"]))}</span>'
            f"{escape(m['snippet'])}</div>"
            for m in matches[:2]
        )
        ui.html(
            '<div class="bm-policy-card-top">'
            f"{_policy_id_badge(policy['policy_id'])}"
            f'<span class="bm-version">v{escape(policy["version"])}</span></div>'
            f'<div class="bm-card-title">{escape(name)}</div>'
            f'<p class="bm-card-body">{escape(purpose)}</p>'
            f"{match_html}"
            f'<div class="bm-policy-meta">{len(policy["sections"])} sections · {escape(policy["source_file"])}</div>'
        )

        with st.container(horizontal=True, gap="small", key=f"bm-link-policy-{policy['policy_id']}"):
            if st.button("Read policy :material/arrow_forward:", key=f"policy-open-{policy['policy_id']}", type="tertiary"):
                _open(policy["policy_id"])
                st.rerun()
            if st.button("Ask", key=f"policy-ask-{policy['policy_id']}", type="tertiary", icon=":material/forum:"):
                _ask(f"Summarise the {name} ({policy['policy_id']}).")


def render_passages(question, result):
    if result["status"] != "SUCCESS":
        ui.empty_state(
            "search_off",
            "No sufficiently relevant passages",
            "The policy index has nothing close enough to this question. "
            "The assistant would also answer without citing a policy.",
        )
        return

    count = len(result["evidence"])
    ui.html(
        f'<div class="bm-list-meta">{count} passage{"s" if count != 1 else ""} · '
        "closest first · same retrieval the assistant uses</div>"
    )

    for index, item in enumerate(result["evidence"]):
        policy_name = pdata.display_case(item.get("policy_name") or "")
        section = pdata.passage_section_label(item)

        if pdata.is_header_passage(item):
            # Header chunks are metadata lines; keep them on separate lines.
            body_html = "".join(
                f"<p>{escape(line.strip())}</p>"
                for line in item["content"].splitlines() if line.strip()
            )
        else:
            body_html = _render_blocks(item["content"])
        distance = item.get("distance")
        distance_html = (
            f'<span class="bm-distance" title="Vector distance — lower is closer">distance {float(distance):.2f}</span>'
            if distance is not None else ""
        )

        with st.container(key=f"bm-panel-passage-{index}"):
            ui.html(
                '<div class="bm-passage-head">'
                f"{_policy_id_badge(item['policy_id'])}"
                f'<span class="bm-passage-title">{escape(policy_name)} · {escape(section)}</span>'
                f"{distance_html}</div>"
                f'<div class="bm-passage-body">{body_html}</div>'
            )

            if pdata.find_policy(catalog, item["policy_id"]):
                with st.container(key=f"bm-link-passage-{index}"):
                    if st.button(
                        "Open policy :material/arrow_forward:",
                        key=f"passage-open-{index}",
                        type="tertiary",
                    ):
                        _open(item["policy_id"])
                        st.rerun()

    if st.button(
        "Ask BankMate this question",
        icon=":material/forum:",
        key="passage-ask",
        type="primary",
    ):
        _ask(question)


def render_library():
    ui.page_header(
        "Policy Intelligence",
        "Browse the fictional BankMate training policies, read them section "
        "by section, and search the same index the assistant cites from.",
    )

    if load_error is not None:
        st.error("The policy documents could not be loaded. Details are in the server logs.",
                 icon=":material/error:")
        return

    if not catalog:
        ui.empty_state("folder_off", "No policies", "knowledge/policies contains no policy documents.")
        return

    # ---------- Keyword search ----------
    with st.container(key="bm-panel-filters"):
        query_col, scope_col = st.columns([3, 1.3], gap="medium", vertical_alignment="bottom")
        with query_col:
            query = st.text_input(
                "Find a policy",
                placeholder="Search titles, IDs, headings or text",
                icon=":material/search:",
                key="policy-filter",
                max_chars=200,
            )
        with scope_col:
            scope = st.selectbox("Search in", pdata.SEARCH_SCOPES, key="policy-scope")

    results = pdata.search_catalog(catalog, query, scope)

    if not results:
        ui.empty_state(
            "search_off",
            "No policies match",
            f"Nothing in {scope.lower()} matches “{query.strip()}”. Try another term or search all text.",
        )
    else:
        noun = "policy" if len(results) == 1 else "policies"
        ui.html(f'<div class="bm-list-meta">{len(results)} {noun}</div>')

        columns = st.columns(3, gap="medium")
        for index, (policy, matches) in enumerate(results):
            with columns[index % 3]:
                render_policy_card(policy, matches)

    # ---------- Semantic passage search ----------
    ui.section_label("Search the policy index")
    index_status.show_index_banner()

    with st.container(key="bm-panel-retrieval"):
        ui.html(
            '<p class="bm-panel-subtitle bm-retrieval-intro">Finds the closest passages '
            "using the assistant's retrieval (same embeddings, index and relevance "
            "threshold). Results show what would be available to cite.</p>"
        )

        input_col, button_col = st.columns([4, 1], gap="small", vertical_alignment="bottom")
        with input_col:
            question = st.text_input(
                "Question",
                placeholder="e.g. What details are needed to raise a card dispute?",
                key="policy-retrieval-input",
                label_visibility="collapsed",
                max_chars=500,
            )
        with button_col:
            clicked = st.button("Search", key="policy-retrieval-search", use_container_width=True)

        question = " ".join((question or "").split())
        needs_search = question and (clicked or question != st.session_state.get(LAST_QUERY))

        if needs_search:
            try:
                with st.spinner("Searching the policy index… the first search loads the embedding model."):
                    st.session_state[LAST_RESULT] = pdata.retrieve_passages(question)
                st.session_state[LAST_QUERY] = question
            except pdata.RetrievalError as error:
                logger.error("Policy passage search failed: %s", error)
                st.session_state[LAST_QUERY] = None
                st.session_state[LAST_RESULT] = None
                st.error(
                    "The policy index could not be searched. If the vector store is "
                    "missing, rebuild it with `python -m app.vector_store`. "
                    "Details are in the server logs.",
                    icon=":material/error:",
                )

        if question and st.session_state.get(LAST_QUERY) == question and st.session_state.get(LAST_RESULT):
            render_passages(question, st.session_state[LAST_RESULT])
        elif not question:
            ui.html('<div class="bm-hint">Type a question and press Enter or Search.</div>')


# =====================================================================

if selected is not None:
    render_detail(selected)
else:
    if requested and load_error is None and catalog:
        # Never echo the raw URL value: it is attacker-controllable and
        # st.warning renders markdown (links, images).
        shown = requested if POLICY_ID_FORMAT.match(requested) else "the requested ID"
        st.warning(f"No policy matches {shown}. Showing all policies.", icon=":material/info:")
        if "policy" in st.query_params:
            del st.query_params["policy"]
    render_library()
