"""Guard: the code base must stay importable on Python 3.11 (the version CI and deployment use).

Python 3.12 relaxed f-string rules (PEP 701), so code that works on a 3.12
dev machine can be a SyntaxError on 3.11. This test finds those constructs
statically, so it fails on any interpreter.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "services", "serving", "frontend", "scripts", "tests")


def _py312_only_fstrings(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:                 # on 3.11 the offending file does not even parse
        return [f"{path.name}:{exc.lineno} SyntaxError on this interpreter: {exc.msg}"]
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        whole = ast.get_source_segment(source, node) or ""
        prefix = whole.lstrip("fFrRbB")
        quote = next((q for q in ('"""', "'''", '"', "'") if prefix.startswith(q)), '"')
        for part in node.values:
            if not isinstance(part, ast.FormattedValue):
                continue
            expr = ast.get_source_segment(source, part.value) or ""
            if "\\" in expr:
                problems.append(f"{path.name}:{part.value.lineno} backslash inside f-string expression")
            elif len(quote) == 1 and quote in expr:
                problems.append(f"{path.name}:{part.value.lineno} f-string expression reuses the outer quote")
    return problems


def test_no_python_312_only_fstring_syntax() -> None:
    problems: list[str] = []
    for folder in SOURCE_DIRS:
        for path in (ROOT / folder).rglob("*.py"):
            problems.extend(_py312_only_fstrings(path))
    assert not problems, "Python 3.12-only f-string syntax:\n" + "\n".join(problems)


def test_the_scanner_itself_detects_the_bug(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("kind = 'dot'\nx = f\"{'\\\\cdot' if kind == 'dot' else 'y'}\"\n", encoding="utf-8")
    assert _py312_only_fstrings(bad), "scanner failed to flag a backslash inside an f-string expression"
