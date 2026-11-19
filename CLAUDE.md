# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BankMate is a LangChain + Groq tool-calling assistant over synthetic banking data (CSV) and fictional policy documents (RAG via ChromaDB). All data is synthetic; policies must always be presented as fictional "BankMate training material".

## Commands

Run everything from the repo root with the venv active (`source .venv/bin/activate`). Modules use `app.`-prefixed imports, so run them with `python -m`, not as file paths.

```bash
pip install -r requirements.txt
python -m app.vector_store        # (Re)build chroma_db/ from knowledge/policies/*.txt — required after editing policies or chunking logic
python main.py                    # Terminal chat (type 'exit' to quit)
streamlit run streamlit_app.py    # Web UI; restart after editing ui/ modules (no watchdog, so module changes may not reload)
python -m app.evaluate_rag        # Retrieval eval: Recall@3 / Precision@3 / false-positive rate vs data/rag_evaluation.csv
python -m app.compare_retrieval   # Similarity vs MMR retriever comparison on the same dataset
pip install -r requirements-dev.txt  # adds pytest
python -m pytest                  # Offline tests (fake model + fake tools; no Groq calls, no ChromaDB)
python -m pytest tests/test_citations.py::test_format_citation_is_readable  # single test
python -m pytest tests/test_ui_app.py  # Streamlit AppTest UI tests (fake backend via monkeypatch)
```

Several modules have `__main__` smoke tests (`app.document_loader` prints chunks, `app.retriever` prints raw distances, `app.rag_chain`, `app.embeddings`). There is no linter or build step.

`GROQ_API_KEY` must be set in `.env`; `app/llm.py` raises at import time if it's missing, so anything importing `app.assistant` needs it (`tests/conftest.py` sets a dummy key for this reason). The Groq model itself is created lazily on first question.

## Architecture

**Request flow:** `main.py` / `streamlit_app.py` → `app/assistant.py:ask_bankmate(question, session_id)` → Groq model (`openai/gpt-oss-20b`, bound to tools in `app/llm.py`) → tools in `app/tools.py` → back to the model, up to 5 tool-call rounds.

- **assistant.py** owns the long system prompt (behavioral rules: INR only, no invented figures/policies, no credit decisions, handle `NO_RELEVANT_POLICY_FOUND`, pronoun resolution) and an in-process `conversation_sessions` dict keyed by `session_id` (lost on restart). `ask_bankmate_with_sources()` returns `{"answer", "sources"}` (used by Streamlit); `ask_bankmate()` returns the same as text with a Sources block (used by the CLI). Both accept injected `model`/`tools` for tests.
- **citations.py** (pure, no I/O): the system prompt tells the model to end policy answers with `CITED_POLICIES: POL-xxx, ...` or `CITED_POLICIES: NONE`. That line is the only coverage signal: it is stripped and its IDs are intersected with sources actually returned by `search_banking_policies` in the current turn; a missing marker or `NONE` yields no citations (fail closed). Citations are policy-level only — section names are deliberately not shown because the model doesn't report which sections it used.
- **grounding.py**: once a session has used the policy tool, answers are checked by `assistant._enforce_grounding` for (a) contact channels (phone/chat/branch/email/app/…) absent from the session's support text (policy evidence + customer tool results) and (b) procedural details — refund destination, timeframe, fees (`DETAIL_PATTERNS`) — absent from the retrieved policy evidence only (`grounding_context[...]["evidence_texts"]`). Detail checks run on policy turns and on follow-ups answered from memory, not on turns that only called customer tools. Any issue triggers one revision request asking for "The policy does not specify …". If it persists: channels → only example clauses removed, sentence kept, console warning; details → the sentence is replaced by an explicit "The policy does not specify …" statement. Clauses that say the policy does *not* specify something ("does not specify/mention/state…") are exempt from both checks, per clause, so grounded "not specified" answers don't trigger revisions. System-prompt rules 38–41 (POLICY GROUNDING RULES) carry the same instruction. The tool's JSON shape (`status`, `sources[]` with `policy_id/policy_name/version/section/source_file`) is a contract with this module.
- **llm.py** defines the `tools` list and `tool_map`; adding a tool means writing it in `tools.py` with `@tool` and registering it in both places here.
- **tools.py**: customer tools read the in-memory CSVs loaded at import time by `data_loader.py`; metrics use `Decimal` and return strings. `search_banking_policies` calls `retriever.search_relevant_documents(k=3, max_distance=1.2)`.
- **RAG ingestion** (`document_loader.py` → `vector_store.py`): each policy file's header (`DOCUMENT:`, `DOCUMENT ID: POL-xxx`, `VERSION:`) is regex-parsed into metadata; content is split on uppercase lines ending in `:` into sections, then each section is chunked (500/80). Chunks carry `policy_id`, `policy_name`, `version`, `section`, `source_file`, `chunk_id`. New policy files must follow this header/heading format or metadata becomes `UNKNOWN`/`GENERAL`. `vector_store.py` deletes and recreates the `bankmate_policies` collection on each run.
- **retriever.py** opens the persisted Chroma store (`all-MiniLM-L6-v2`, normalized, CPU) and filters by distance — lower is more similar. Note the thresholds are not centralized: tools/rag_chain use 1.2, `search_relevant_documents` defaults to 1.3, `evaluate_rag.py` uses 1.4. `search_by_policy` supports metadata filtering by policy ID/section.
- **rag_chain.py** is a standalone single-shot RAG chain (not used by the assistant).
- Evaluation scripts map filenames → policy IDs via a hardcoded dict; update it when adding policy files. Rows with `expected_policy_id=NONE` are out-of-scope questions that should retrieve nothing.

## Web UI (`ui/`, `streamlit_app.py`, `.streamlit/config.toml`)

- `streamlit_app.py` sets page config, applies `ui.theme`, builds `st.navigation` from `ui/navigation.PAGES`, then runs the page script in `ui/views/`. Register a page in `navigation.PAGES` only once it is implemented — the sidebar, mobile nav and tests all read that list, and the product rule is no fake navigation or inert controls.
- Theme colours/fonts live in `.streamlit/config.toml`; `ui/theme.py` holds matching `TOKENS` plus a scoped CSS layer that targets keyed containers (`.st-key-<key>`, e.g. `bm-card-*`, `bm-chip-*`, `bm-msg-user-*`) and stable `data-testid` hooks. Keep the two in sync.
- `ui/state.py` is the conversation store: several conversations per browser session, each with its own `session_id` passed to `ask_bankmate_with_sources` (this is what isolates backend memory/grounding context). Cross-page prompts go through `queue_prompt(..., new_conversation=True)` / `pop_pending_prompt()`; `components.go_to_assistant()` wraps this.
- `ui/views/assistant.py` imports `app.assistant` inside a try/except so other pages work without `GROQ_API_KEY`. It records the question, reruns, then calls the backend on the next run (avoids stale elements lingering during the call). It only displays citations the backend returned.
- `ui/data.py` is the only data access for the Explorer/Analytics pages: records via `app.data_loader`, per-customer metrics via the `get_customer_metrics` tool (so the UI matches the assistant), and filtered aggregates with the same Decimal/type rule (tests cross-check them). Always present net flow as credits minus debits, never as a balance. `ui/charts.py` follows the dataviz rules (≤24px bars, 4px rounded ends, hairline grid, text-coloured labels, legend only for 2+ series); credit/debit colours and `chartCategoricalColors` order were validated for colour-blind separation — re-validate if changed.
- `ui/policy_data.py` builds the policy catalogue from `app.document_loader` (`load_policy_documents`, `extract_policy_metadata`, `split_into_sections`) — never hardcode policy metadata. The header-only `GENERAL` block is dropped and `IMPORTANT` is kept as the fictional notice. Passage search calls `app.tools.search_banking_policies` (the assistant's exact retrieval), imported at call time so tests can monkeypatch it; failures raise `RetrievalError` for the UI's error state. `app/embeddings.get_embedding_model` is `lru_cache`d (one model per process): the first search takes ~5s, later ones ~10ms. Chroma clients are deliberately not cached, so a rebuilt vector store is picked up.
- `ui/views/policies.py` has a library view and a detail view selected by `?policy=POL-xxx` or `state.select_policy()`. Cross-page navigation to a policy uses `components.go_to_policy()` (session state + `st.switch_page`), not URL links — a full reload would start a new Streamlit session and lose conversations. The assistant renders an "Open POL-xxx" button under each cited source.
- Mobile nav (<768px): only the current page's pill shows its label; others are icon-only with the label visually hidden but kept for screen readers (CSS in `ui/theme.py`). Keep page `short` labels in `navigation.PAGES` short.
- AppTest gotcha (Streamlit 1.50): a single-select `segmented_control` must be set as a one-item list via `at.get("button_group")` before every rerun, or AppTest crashes iterating its string value (see `set_type` in `tests/test_ui_app.py`).
- AppTest gotcha: right after a click that calls `st.rerun()` inside the script, `at.button`/`at.markdown` can still include elements from the previous layout (verified not to happen in the browser). Assert on content specific to the new view, or call `at.run()` once more before listing elements.
- UI tests monkeypatch `app.assistant.ask_bankmate_with_sources`; the page re-imports it each run, so the fake is picked up. `page_text()` in the tests skips the injected `<style>` block.
