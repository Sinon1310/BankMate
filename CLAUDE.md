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
streamlit run streamlit_app.py    # Web UI (streamlit is installed in .venv but NOT listed in requirements.txt)
python -m app.evaluate_rag        # Retrieval eval: Recall@3 / Precision@3 / false-positive rate vs data/rag_evaluation.csv
python -m app.compare_retrieval   # Similarity vs MMR retriever comparison on the same dataset
pip install -r requirements-dev.txt  # adds pytest
python -m pytest                  # Offline tests (fake model + fake tools; no Groq calls, no ChromaDB)
python -m pytest tests/test_citations.py::test_format_citation_is_readable  # single test
```

Several modules have `__main__` smoke tests (`app.document_loader` prints chunks, `app.retriever` prints raw distances, `app.rag_chain`, `app.embeddings`). There is no linter or build step.

`GROQ_API_KEY` must be set in `.env`; `app/llm.py` raises at import time if it's missing, so anything importing `app.assistant` needs it (`tests/conftest.py` sets a dummy key for this reason). The Groq model itself is created lazily on first question.

## Architecture

**Request flow:** `main.py` / `streamlit_app.py` → `app/assistant.py:ask_bankmate(question, session_id)` → Groq model (`openai/gpt-oss-20b`, bound to tools in `app/llm.py`) → tools in `app/tools.py` → back to the model, up to 5 tool-call rounds.

- **assistant.py** owns the long system prompt (behavioral rules: INR only, no invented figures/policies, no credit decisions, handle `NO_RELEVANT_POLICY_FOUND`, pronoun resolution) and an in-process `conversation_sessions` dict keyed by `session_id` (lost on restart). `ask_bankmate_with_sources()` returns `{"answer", "sources"}` (used by Streamlit); `ask_bankmate()` returns the same as text with a Sources block (used by the CLI). Both accept injected `model`/`tools` for tests.
- **citations.py** (pure, no I/O): the system prompt tells the model to end policy answers with `CITED_POLICIES: POL-xxx, ...` or `CITED_POLICIES: NONE`. That line is the only coverage signal: it is stripped and its IDs are intersected with sources actually returned by `search_banking_policies` in the current turn; a missing marker or `NONE` yields no citations (fail closed). Citations are policy-level only — section names are deliberately not shown because the model doesn't report which sections it used.
- **grounding.py**: once a session has used the policy tool, every answer's contact channels (phone/chat/branch/email/app/…) are checked against that session's support text (policy evidence + customer tool results, kept in `assistant.grounding_context`). Unsupported ones trigger one revision request; if that fails, only parenthetical/"e.g."/"such as"/"including" example clauses naming them are removed. Sentences are never deleted — anything left is kept and a console warning is printed (`assistant._enforce_channel_grounding`). The tool's JSON shape (`status`, `sources[]` with `policy_id/policy_name/version/section/source_file`) is a contract with this module.
- **llm.py** defines the `tools` list and `tool_map`; adding a tool means writing it in `tools.py` with `@tool` and registering it in both places here.
- **tools.py**: customer tools read the in-memory CSVs loaded at import time by `data_loader.py`; metrics use `Decimal` and return strings. `search_banking_policies` calls `retriever.search_relevant_documents(k=3, max_distance=1.2)`.
- **RAG ingestion** (`document_loader.py` → `vector_store.py`): each policy file's header (`DOCUMENT:`, `DOCUMENT ID: POL-xxx`, `VERSION:`) is regex-parsed into metadata; content is split on uppercase lines ending in `:` into sections, then each section is chunked (500/80). Chunks carry `policy_id`, `policy_name`, `version`, `section`, `source_file`, `chunk_id`. New policy files must follow this header/heading format or metadata becomes `UNKNOWN`/`GENERAL`. `vector_store.py` deletes and recreates the `bankmate_policies` collection on each run.
- **retriever.py** opens the persisted Chroma store (`all-MiniLM-L6-v2`, normalized, CPU) and filters by distance — lower is more similar. Note the thresholds are not centralized: tools/rag_chain use 1.2, `search_relevant_documents` defaults to 1.3, `evaluate_rag.py` uses 1.4. `search_by_policy` supports metadata filtering by policy ID/section.
- **rag_chain.py** is a standalone single-shot RAG chain (not used by the assistant).
- Evaluation scripts map filenames → policy IDs via a hardcoded dict; update it when adding policy files. Rows with `expected_policy_id=NONE` are out-of-scope questions that should retrieve nothing.
