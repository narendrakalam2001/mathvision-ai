"""Every CLI script must start from ANY working directory.

`python scripts/foo.py` puts only ``scripts/`` on ``sys.path``, so a script that
does ``from src... import`` crashes with ``ModuleNotFoundError`` unless it
bootstraps the project root. pytest hides this (``pythonpath = .``), so this
test launches each script in a real subprocess from a different directory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PACKAGES = ("src", "services", "serving", "frontend", "scripts")
SCRIPTS = sorted(p.name for p in (ROOT / "scripts").glob("*.py") if p.name != "__init__.py" and p.name != "run_app.py")


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help_runs_from_another_directory(script: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        missing = [line for line in result.stderr.splitlines() if "No module named" in line]
        # a missing THIRD-PARTY package (e.g. torch on a slim CI box) is not this test's concern
        third_party_only = missing and not any(f"'{pkg}'" in missing[-1] or f"'{pkg}." in missing[-1]
                                               for pkg in PROJECT_PACKAGES)
        if third_party_only:
            pytest.skip(f"optional dependency missing: {missing[-1].strip()}")
        pytest.fail(f"{script} --help failed:\n{result.stderr[-600:]}")


def test_run_app_points_at_an_existing_frontend() -> None:
    assert (ROOT / "frontend" / "app.py").is_file()
    assert "frontend" in (ROOT / "scripts" / "run_app.py").read_text(encoding="utf-8")
