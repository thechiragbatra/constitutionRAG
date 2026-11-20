"""Entry point for Render deployment.

Re-exports the FastAPI app from backend/main.py so `uvicorn main:app` works.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app  # noqa: E402
