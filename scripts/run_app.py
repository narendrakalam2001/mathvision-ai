#!/usr/bin/env python3
"""Convenience launcher for the Streamlit frontend.

Usage:
    python scripts/run_app.py
    python scripts/run_app.py --port 8502
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MathVision AI Streamlit app.")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)],
        check=True,
    )


if __name__ == "__main__":
    main()
