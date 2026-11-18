import os
import sys
from pathlib import Path

# Force a dummy key so importing app.llm never needs (or uses) a real
# Groq credential. Citation tests inject a fake model instead.
os.environ["GROQ_API_KEY"] = "test-dummy-key"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
