"""
BankMate assistant evaluation (Phase 1: offline).

Runs evaluation cases through the real assistant loop
(app.assistant.ask_bankmate_with_sources) with an injected model and
tool set, records every model call and tool execution, and scores the
results deterministically. Offline mode never calls Groq, never loads
the embedding model and never opens ChromaDB.

    python -m evals.run                      # offline run -> evals/results/*.json
    python -m evals.compare A.json B.json    # compare two saved runs
"""
