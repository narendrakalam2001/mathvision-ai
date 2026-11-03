"""LaTeX -> SymPy conversion (feeds Module 3, the math solver).

Converts the validated LaTeX string produced by :mod:`src.ocr_engine`
into SymPy expressions/equations that :mod:`src.math_solver` (Day 2)
can operate on. Wraps SymPy's ``parse_latex`` (via ``antlr4``) with:

    * pre-cleaning of common OCR artifacts (stray spaces, smart quotes,
      Hindi-typed operator words that occasionally leak into OCR output)
    * splitting on ``=`` to build proper ``sympy.Eq`` objects for equations
    * splitting multi-equation systems (e.g. simultaneous equations
      separated by ``;`` or newlines)
    * clear, typed exceptions so the API layer can return a friendly
      "Hume samajh nahi aaya, please edit the LaTeX" message

Example:
    >>> from src.latex_parser import LatexParser
    >>> parser = LatexParser()
    >>> parser.parse_equation(r"x^2 - 5x + 6 = 0")
    Eq(x**2 - 5*x + 6, 0)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import sympy
from sympy import Eq, Symbol, sympify
from sympy.parsing.latex import parse_latex

from src.vector_ops import VectorError, VectorParser, VectorProblem

logger = logging.getLogger(__name__)


class LatexParseError(Exception):
    """Raised when a LaTeX string cannot be converted to a SymPy expression."""


# Symbols very commonly present in NCERT/JEE problems; declared up-front
# so `x`, `y`, `theta`, etc. are treated consistently as symbols rather
# than being accidentally re-bound.
COMMON_SYMBOLS: dict[str, Symbol] = {
    name: Symbol(name)
    for name in [
        "x", "y", "z", "t", "n", "k", "a", "b", "c",
        "theta", "phi", "alpha", "beta", "gamma",
    ]
}

_PRECLEAN_RULES: list[tuple[str, str]] = [
    (r"[“”]", '"'),
    (r"[‘’]", "'"),
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\cdot", r"*"),
    (r"\\times", r"*"),
    (r"\\div", r"/"),
    (r"\s*\\,\s*", " "),   # thin space macro
    (r"\\!", ""),
    (r"\u200b", ""),        # zero-width space sometimes injected by OCR
]


def _read_arg(text: str, i: int) -> tuple[Optional[str], int]:
    """Read one LaTeX macro argument starting at ``text[i]``.

    Accepts a ``{...}`` group (balanced), a control sequence such as
    ``\\alpha``, or a single character -- the three forms CROHME-style
    output uses (``\\frac 1 x``, ``\\frac {a} 2``).

    Returns:
        ``(argument_without_outer_braces, index_after_argument)`` or
        ``(None, i)`` when no argument is present.
    """
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        return None, i
    if text[i] == "{":
        depth = 0
        for j in range(i, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j], j + 1
        return None, i
    if text[i] == "\\":
        match = re.match(r"\\[A-Za-z]+", text[i:])
        if match:
            return match.group(0), i + match.end()
        return text[i : i + 2], i + 2
    return text[i], i + 1


def _brace_macro(text: str, macro: str, num_args: int) -> str:
    """Rewrite ``macro a b`` as ``macro{a}{b}`` (recursively) for brace-less arguments."""
    out: list[str] = []
    i = 0
    while True:
        k = text.find(macro, i)
        if k < 0:
            out.append(text[i:])
            break
        after = k + len(macro)
        if after < len(text) and text[after].isalpha():      # e.g. \fractal -> not our macro
            out.append(text[i:after])
            i = after
            continue
        out.append(text[i:k])
        args: list[str] = []
        j = after
        if macro == "\\sqrt" and text[after:].lstrip().startswith("["):   # \sqrt[3]{x}
            out.append(text[k:after])
            i = after
            continue
        for _ in range(num_args):
            arg, j = _read_arg(text, j)
            if arg is None:
                break
            args.append(_brace_macro(arg, macro, num_args))
        if len(args) != num_args:
            out.append(text[k:after])
            i = after
            continue
        out.append(macro + "".join("{" + a + "}" for a in args))
        i = j
    return "".join(out)


_MATRIX_RE = re.compile(r"\\begin\{([pbvBV]?matrix)\}(.*?)\\end\{\1\}", re.S)


def normalize_crohme(latex: str) -> str:
    """Convert CROHME-style LaTeX (what the fine-tuned OCR emits) to standard LaTeX.

    Handles ``\\frac 1 x`` -> ``\\frac{1}{x}``, ``\\sqrt 2`` -> ``\\sqrt{2}``,
    ``\\begin {pmatrix}`` -> ``\\begin{pmatrix}``, ``\\mbox { G }`` -> ``G`` and
    ``\\dfrac``/``\\tfrac`` -> ``\\frac``.

    Args:
        latex: LaTeX as produced by the OCR model.

    Returns:
        Standard LaTeX that SymPy's parser understands.
    """
    text = latex
    text = re.sub(r"\\[dt]frac", r"\\frac", text)
    text = re.sub(r"\\(begin|end)\s*\{\s*([A-Za-z*]+)\s*\}", r"\\\1{\2}", text)
    text = re.sub(r"\\(?:mbox|mathrm|text|textrm)\s*\{\s*([^{}]*?)\s*\}", r"\1", text)
    text = _brace_macro(text, "\\frac", 2)
    text = _brace_macro(text, "\\sqrt", 1)
    return re.sub(r"[ \t]+", " ", text).strip()


def _split_top_level(text: str, separators: str = ",;") -> list[str]:
    """Split ``text`` on separator characters that are not inside brackets/braces."""
    parts: list[str] = []
    depth, start = 0, 0
    for idx, ch in enumerate(text):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth = max(0, depth - 1)
        elif ch in separators and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return [p for p in parts if p.strip()]


def _preclean(latex: str) -> str:
    """Normalize common OCR/typography artifacts before parsing.

    Args:
        latex: Raw LaTeX string (post OCR symbol-correction).

    Returns:
        Cleaned LaTeX string, still valid LaTeX.
    """
    cleaned = latex.strip()
    for pattern, replacement in _PRECLEAN_RULES:
        cleaned = re.sub(pattern, replacement, cleaned)
    return normalize_crohme(cleaned)


@dataclass
class ParsedProblem:
    """Result of parsing a (possibly multi-equation) LaTeX problem.

    Attributes:
        equations: List of ``sympy.Eq`` objects (length 1 for a single
            equation, >1 for simultaneous-equation systems).
        expression: For non-equation inputs (e.g. a bare expression to
            differentiate/integrate/simplify), the parsed SymPy
            expression; ``None`` for equation systems.
        symbols: Free symbols found across all equations/expressions.
        raw_latex: The original (precleaned) LaTeX input.
        vector: Parsed vector problem (dot/cross/magnitude/...), or ``None``.
    """

    equations: list[Eq]
    expression: Optional[sympy.Expr]
    symbols: set[Symbol]
    raw_latex: str
    vector: Optional["VectorProblem"] = None


class LatexParser:
    """Converts OCR'd LaTeX into SymPy objects for the solver pipeline."""

    def __init__(self, extra_symbols: Optional[dict[str, Symbol]] = None) -> None:
        """Create a parser.

        Args:
            extra_symbols: Additional name->Symbol mappings to recognize
                beyond :data:`COMMON_SYMBOLS` (e.g. domain-specific
                variables used in a particular problem set).
        """
        self.symbol_table = dict(COMMON_SYMBOLS)
        if extra_symbols:
            self.symbol_table.update(extra_symbols)

    @staticmethod
    def _canonical(expr: sympy.Expr) -> sympy.Expr:
        """Rebuild an expression tree so SymPy re-evaluates nested ``Add``/``Mul`` nodes.

        ``parse_latex`` returns unevaluated trees such as ``Add(Add(x**2, -x), -90)``
        for ``x^2 - x - 90``; ``Poly`` rejects them, which silently turned many
        quadratics into "unknown". Rebuilding flattens them without evaluating
        integrals, derivatives or limits (unlike ``doit``).
        """
        if not getattr(expr, "args", None):
            return expr
        return expr.func(*[LatexParser._canonical(arg) for arg in expr.args])

    @staticmethod
    def _euler_constant(expr: sympy.Expr) -> sympy.Expr:
        """Treat the symbol ``e`` as Euler's number (``e^{x}`` is common in NCERT/JEE).

        SymPy's LaTeX parser reads a bare ``e`` as an ordinary variable, which
        makes ``\\int e^{x} dx`` come out as a Piecewise in ``log(e)``.
        """
        e_symbol = Symbol("e")
        if hasattr(expr, "has") and expr.has(e_symbol):
            return expr.subs(e_symbol, sympy.E)
        return expr

    def _latex_to_sympy(self, latex_fragment: str) -> sympy.Expr:
        """Parse a single (equation-free) LaTeX fragment into a SymPy expression.

        Tries ``sympy.parsing.latex.parse_latex`` first (handles full
        LaTeX syntax like ``\\frac``, ``\\sqrt``, ``\\int``); falls back
        to plain ``sympify`` for simple algebraic strings that slipped
        through without LaTeX macros.

        Raises:
            LatexParseError: If both parsing strategies fail.
        """
        fragment = fragment_stripped = latex_fragment.strip()
        if not fragment_stripped:
            raise LatexParseError("Empty expression.")

        try:
            return self._canonical(self._euler_constant(parse_latex(fragment)))
        except Exception as latex_exc:  # noqa: BLE001
            logger.debug("parse_latex failed for %r: %s", fragment, latex_exc)

        try:
            # Fallback for plain algebraic notation, e.g. "2*x + 3"
            plain = fragment.replace("^", "**")
            return sympify(plain, locals=self.symbol_table)
        except Exception as exc:  # noqa: BLE001
            raise LatexParseError(f"Could not parse expression: {latex_fragment!r}") from exc

    def parse_equation(self, latex: str) -> Eq:
        """Parse a single-equation LaTeX string (must contain exactly one '=').

        Args:
            latex: e.g. ``r"x^2 - 5x + 6 = 0"``.

        Returns:
            A ``sympy.Eq`` object.

        Raises:
            LatexParseError: If the string doesn't contain exactly one
                '=' sign, or either side fails to parse.
        """
        cleaned = _preclean(latex)
        parts = cleaned.split("=")
        if len(parts) != 2:
            raise LatexParseError(
                f"Expected exactly one '=' in equation, found {len(parts) - 1}: {latex!r}"
            )

        lhs = self._latex_to_sympy(parts[0])
        rhs = self._latex_to_sympy(parts[1])
        return Eq(lhs, rhs)

    def parse_system(self, latex: str) -> list[Eq]:
        """Parse a system of simultaneous equations.

        Equations may be separated by newlines, semicolons, or the
        LaTeX ``\\\\`` line break (common when OCR reads a 2-line
        system of equations as one block).

        Args:
            latex: Multi-equation LaTeX block.

        Returns:
            List of ``sympy.Eq`` objects, one per equation.
        """
        cleaned = _preclean(latex)
        raw_lines: list[str] = []
        for chunk in re.split(r"\\\\|\n", cleaned):
            raw_lines.extend(_split_top_level(chunk, ",;"))
        equations = [self.parse_equation(line) for line in raw_lines if line.strip()]
        if not equations:
            raise LatexParseError(f"No equations found in system: {latex!r}")
        return equations

    def _extract_matrix(self, cleaned: str) -> Optional[sympy.Matrix]:
        """Build a SymPy matrix from ``\\begin{pmatrix}...\\end{pmatrix}`` blocks.

        Several blocks are combined using the text between them: ``+``/``-``
        add/subtract, anything else (adjacency, ``*``) is a matrix product.

        Returns:
            The resulting matrix, or ``None`` if the input has no matrix block.

        Raises:
            LatexParseError: If a block is malformed or shapes are incompatible.
        """
        found = list(_MATRIX_RE.finditer(cleaned))
        if not found:
            return None
        try:
            matrices = []
            for block in found:
                rows = [r for r in re.split(r"\\\\", block.group(2)) if r.strip()]
                data = [[self._latex_to_sympy(cell) for cell in row.split("&")] for row in rows]
                matrices.append(sympy.Matrix(data))
            result = matrices[0]
            for prev, block, mat in zip(found, found[1:], matrices[1:]):
                operator = cleaned[prev.end() : block.start()].strip()
                if operator == "+":
                    result = result + mat
                elif operator == "-":
                    result = result - mat
                else:
                    result = result * mat
            return result
        except LatexParseError:
            raise
        except Exception as exc:  # noqa: BLE001 - ShapeError, ValueError, ...
            raise LatexParseError(f"Could not build matrix from {cleaned!r}: {exc}") from exc

    def parse(self, latex: str) -> ParsedProblem:
        """Parse arbitrary OCR'd LaTeX into a :class:`ParsedProblem`.

        Automatically decides whether the input is a single equation,
        a system of equations, or a bare expression (for
        differentiation/integration/simplification tasks).

        Args:
            latex: Raw LaTeX from the OCR engine.

        Returns:
            A populated :class:`ParsedProblem`.

        Raises:
            LatexParseError: If parsing fails entirely.
        """
        cleaned = _preclean(latex)
        if not cleaned:
            raise LatexParseError("Received empty LaTeX input.")

        try:
            vector = VectorParser(self._latex_to_sympy).parse(latex)
        except VectorError as exc:
            raise LatexParseError(str(exc)) from exc
        except LatexParseError:
            vector = None                    # not a vector after all: try the other shapes
        if vector is not None:
            return ParsedProblem(
                equations=[], expression=None, symbols=set(vector.free_symbols),
                raw_latex=cleaned, vector=vector,
            )

        matrix = self._extract_matrix(cleaned)
        if matrix is not None:
            return ParsedProblem(
                equations=[], expression=matrix, symbols=set(matrix.free_symbols), raw_latex=cleaned
            )

        num_equals = cleaned.count("=")
        has_system_separator = bool(re.search(r"\\\\|;|\n", cleaned))

        equations: list[Eq] = []
        expression: Optional[sympy.Expr] = None

        if num_equals >= 2 or (num_equals >= 1 and has_system_separator):
            equations = self.parse_system(cleaned)
        elif num_equals == 1:
            equations = [self.parse_equation(cleaned)]
        else:
            expression = self._latex_to_sympy(cleaned)

        symbols: set[Symbol] = set()
        for eq in equations:
            symbols |= eq.free_symbols
        if expression is not None:
            symbols |= expression.free_symbols

        return ParsedProblem(
            equations=equations,
            expression=expression,
            symbols=symbols,
            raw_latex=cleaned,
        )


if __name__ == "__main__":
    parser = LatexParser()

    samples = [
        r"x^2 - 5x + 6 = 0",
        r"\frac{-b \pm \sqrt{b^2-4ac}}{2a}",
        r"2x + y = 5 \\ x - y = 1",
        r"\int_0^1 x^2 dx",
    ]
    for sample in samples:
        try:
            parsed = parser.parse(sample)
            print(f"{sample!r} -> equations={parsed.equations} expr={parsed.expression}")
        except LatexParseError as err:
            print(f"{sample!r} -> FAILED: {err}")
