"""Vercel serverless backend — thin wrapper around backend.main.

Exports the FastAPI `app` from backend/main.py so Vercel's @vercel/python
runtime can serve it. All route logic lives in backend/main.py (single source
of truth). Heavy ML deps are imported lazily so the bundle stays under
Vercel's 500 MB limit.

Endpoints (served at /api/* by vercel.json routing):
  POST   /api/ask
  GET    /api/article/{n}
  GET    /api/config
  GET    /api/health
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app  # noqa: E402

handler = app
