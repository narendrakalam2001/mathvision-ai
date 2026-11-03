#!/usr/bin/env python3
"""Convenience launcher for the FastAPI server.

Usage:
    python scripts/run_api.py                 # http://localhost:8000
    python scripts/run_api.py --port 9000
    PORT=8080 python scripts/run_api.py        # Render/Heroku-style $PORT
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/<name>.py` from any directory: put the project root on sys.path
# (Python only adds the scripts/ folder itself, so `import src` would fail otherwise).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MathVision AI FastAPI server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only)")
    args = parser.parse_args()

    uvicorn.run("serving.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
