import os
import sys
from pathlib import Path

# Force a dummy key so importing app.llm never needs (or uses) a real
# Groq credential. Citation tests inject a fake model instead.
os.environ["GROQ_API_KEY"] = "test-dummy-key"

# UI tests must not touch the real ChromaDB index or load the embedding
# model at app startup; index lifecycle is tested directly.
os.environ["BANKMATE_INDEX_AUTOINIT"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tests use the legacy fixture (the original sample records in the v1
# schema), not data/, so they do not change when data/ is regenerated.
os.environ["BANKMATE_DATA_DIR"] = str(PROJECT_ROOT / "tests" / "fixtures" / "data")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
