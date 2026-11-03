"""Vector problems: parsing (Class 11-12 NCERT) and exact solving with SymPy.

Supported operations, each with worked steps:

    * dot product            ``\\vec{a} \\cdot \\vec{b}``      -> scalar (+ angle remark)
    * cross product          ``\\vec{a} \\times \\vec{b}``     -> vector (+ parallelogram area)
    * magnitude / unit       ``|\\vec{a}|``, a lone vector    -> length and unit vector
    * addition / subtraction ``\\vec{a} + \\vec{b}``
    * angle between          ``Find the angle between a and b``

Accepted vector notations (also what the OCR model produces):

    * ``\\begin{pmatrix} 1 \\\\ 2 \\\\ 3 \\end{pmatrix}`` / ``bmatrix`` / one-row ``matrix``
    * tuples ``(1, 2, 3)``, ``\\langle 1, 2, 3 \\rangle``, ``[1, 2, 3]``
    * unit-vector form ``2\\hat{i} - \\hat{j} + 3\\hat{k}`` (``\\mathbf{i}`` or, in an
      operation context, plain ``2i - j + 3k``)
    * named vectors with definitions:
      ``Find \\vec{a} \\cdot \\vec{b} for \\vec{a} = (1,2,3), \\vec{b} = (4,5,6)``

Example:
    >>> from src.vector_ops import VectorParser, solve_vector
    >>> problem = VectorParser(parse_component).parse(r"(1, 2, 3) \\cdot (4, 5, 6)")
    >>> solve_vector(problem).answer
    '\\\\vec{a} \\\\cdot \\\\vec{b} = 32'
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import sympy
from sympy import Matrix, Rational, latex

logger = logging.getLogger(__name__)

ComponentParser = Callable[[str], sympy.Expr]


class VectorError(Exception):
    """Raised when a vector problem is malformed (e.g. mismatched dimensions)."""


@dataclass
class VectorProblem:
    """A parsed vector problem.

    Attributes:
        operation: ``dot``, ``cross``, ``magnitude``, ``add``, ``subtract`` or ``angle``.
        vectors: Operands as column matrices.
        labels: LaTeX names for the operands (``\\vec{a}``, ``\\vec{b}``).
        source: The cleaned LaTeX the problem was parsed from.
    """

    operation: str
    vectors: list[Matrix]
    labels: list[str]
    source: str = ""

    @property
    def free_symbols(self) -> set:
        """Free symbols across all operands."""
        out: set = set()
        for vec in self.vectors:
            out |= vec.free_symbols
        return out


@dataclass
class VectorSolution:
    """Result of :func:`solve_vector`.

    Attributes:
        answer: Final answer as LaTeX.
        value: Exact SymPy value (scalar or column Matrix).
        numeric: Float approximation(s) for display, when available.
        steps: ``(title, content, why)`` triples for the step list.
        details: Extra data (magnitudes, unit vector, angle, ...).
    """

    answer: str
    value: Any
    numeric: Optional[list[float]]
    steps: list[tuple[str, str, str]]
    details: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
_NAME_RE = re.compile(r"\\(?:vec|overrightarrow|mathbf|boldsymbol|bar)\s*\{\s*([A-Za-z])\s*\}")
_UNIT_RE = re.compile(r"\\(?:hat|mathbf|vec|boldsymbol)\s*\{?\s*(?:\\hat\s*\{?\s*)?([ijk])\s*\}*")
_MATRIX_ENV = re.compile(r"\\begin\{([pbvBV]?matrix)\}(.*?)\\end\{\1\}", re.S)
_IJK_TOKEN = re.compile(r"([+-]?)(\d+(?:\.\d+)?(?:/\d+)?|\\frac\{\d+\}\{\d+\})?\*?(#[ijk]|[ijk])")
_SPLIT_WORDS = re.compile(r"(?i)\b(?:for|where|given|if|when)\b")
_OPEN, _CLOSE = "([{", ")]}"


def _depth0_positions(text: str, token: str) -> list[int]:
    """Indices where ``token`` starts outside any bracket/brace."""
    found: list[int] = []
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(token, i):
            found.append(i)
            i += len(token)
            continue
        i += 1
    return found


def _matching_close(text: str, start: int) -> int:
    """Index of the bracket that closes the one opening at ``text[start]`` (-1 if unbalanced)."""
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] in _OPEN:
            depth += 1
        elif text[idx] in _CLOSE:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _split_depth0(text: str, seps: tuple[str, ...]) -> list[str]:
    """Split on separator characters that are not inside brackets."""
    parts: list[str] = []
    depth, start = 0, 0
    for idx, ch in enumerate(text):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        elif depth == 0 and ch in seps:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


class VectorParser:
    """Turns LaTeX that describes a vector problem into a :class:`VectorProblem`."""

    def __init__(self, parse_component: ComponentParser) -> None:
        """Create a parser.

        Args:
            parse_component: Callable converting one LaTeX fragment (a vector
                component such as ``-2`` or ``\\frac{1}{2}``) to a SymPy expression.
        """
        self._component = parse_component

    # -- public ---------------------------------------------------------
    def parse(self, latex_text: str) -> Optional[VectorProblem]:
        """Parse ``latex_text``.

        Returns:
            A :class:`VectorProblem`, or ``None`` when the input is not a vector
            problem (so the caller can try other interpretations, e.g. matrices).

        Raises:
            VectorError: If it *is* a vector problem but is malformed.
        """
        text = self._canonical(latex_text)
        if not text:
            return None
        lead, core, defs = self._split_sections(text)

        # "Find the angle between a and b"
        if "angle" in lead.lower():
            names = [p.strip() for p in re.split(r"\band\b|,", core) if p.strip()]
            operands = [self._operand(n, defs) for n in names]
            if len(operands) == 2 and all(o is not None for o in operands):
                return self._build("angle", operands, names, defs, text)

        for token, op in (("\\cdot", "dot"), ("\\times", "cross")):
            positions = _depth0_positions(core, token)
            if len(positions) == 1:
                left, right = core[: positions[0]], core[positions[0] + len(token):]
                a, b = self._operand(left, defs), self._operand(right, defs)
                if a is not None and b is not None:
                    return self._build(op, [a, b], [left, right], defs, text)
                return None

        inner = self._strip_bars(core)
        if inner is not None:
            single = self._operand(inner, defs)
            if single is not None:
                return self._build("magnitude", [single], [inner], defs, text)

        explicit_context = bool(re.search(r"(?i)magnitude|length|modulus|norm|unit", lead))
        single = self._operand(core, defs)
        if single is not None and (explicit_context or self._looks_explicit(core)):
            return self._build("magnitude", [single], [core], defs, text)

        for sign, op in (("+", "add"), ("-", "subtract")):
            for pos in _depth0_positions(core, sign):
                if pos == 0:
                    continue
                a, b = self._operand(core[:pos], defs), self._operand(core[pos + 1:], defs)
                if a is not None and b is not None:
                    return self._build(op, [a, b], [core[:pos], core[pos + 1:]], defs, text)
        return None

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _canonical(raw: str) -> str:
        """Standardise vector macros and strip text wrappers."""
        text = raw.strip()
        text = re.sub(r"\\(?:left|right)\s*", "", text)
        text = re.sub(r"\\(?:mbox|text|textrm|mathrm)\s*\{\s*([^{}]*?)\s*\}", r" \1 ", text)
        text = re.sub(r"\\begin\s*\{\s*([A-Za-z*]+)\s*\}", r"\\begin{\1}", text)
        text = re.sub(r"\\end\s*\{\s*([A-Za-z*]+)\s*\}", r"\\end{\1}", text)
        text = _UNIT_RE.sub(lambda m: f"#{m.group(1)}", text)
        text = _NAME_RE.sub(lambda m: f"@{m.group(1)}", text)
        text = text.replace("\\|", "|").replace("\\lvert", "|").replace("\\rvert", "|")
        return re.sub(r"[ \t]+", " ", text).strip().rstrip("?.")

    def _split_sections(self, text: str) -> tuple[str, str, dict[str, Matrix]]:
        """Split ``Find X for a = ..., b = ...`` into lead words, expression and definitions."""
        match = _SPLIT_WORDS.search(text)
        expr_part, defs_part = (text[: match.start()], text[match.end():]) if match else (text, "")
        lead_match = re.match(r"^\s*([A-Za-z ]*)", expr_part)
        lead = lead_match.group(1) if lead_match else ""
        core = expr_part[lead_match.end():].strip() if lead_match else expr_part.strip()
        if not core and not defs_part:       # e.g. bare "a" -> nothing to parse
            core = lead.strip()
        defs: dict[str, Matrix] = {}
        inline = re.match(r"^@?([A-Za-z])\s*=\s*(.+)$", core)
        if inline and not defs_part:
            vector = self._operand(inline.group(2), {})
            if vector is not None:
                defs[inline.group(1)] = vector
                core = "@" + inline.group(1)
        for item in _split_depth0(re.sub(r"(?i)\band\b", ",", defs_part), (",", ";")):
            m = re.match(r"^@?([A-Za-z])\s*=\s*(.+)$", item)
            if not m:
                continue
            vector = self._operand(m.group(2), {})
            if vector is not None:
                defs[m.group(1)] = vector
        return lead, core, defs

    @staticmethod
    def _strip_bars(core: str) -> Optional[str]:
        core = core.strip()
        if len(core) > 2 and core[0] == "|" and core[-1] == "|" and core.count("|") == 2:
            return core[1:-1].strip()
        if core.startswith("||") and core.endswith("||") and len(core) > 4:
            return core[2:-2].strip()
        return None

    @staticmethod
    def _looks_explicit(core: str) -> bool:
        """True when ``core`` uses unmistakable vector notation (marker, tuple, bars)."""
        return bool(re.search(r"[@#]|\\langle|^\s*\(.*,.*\)\s*$|^\s*\[.*,.*\]\s*$", core))

    def _build(self, op: str, operands: list[Matrix], raw_names: list[str],
               defs: dict[str, Matrix], source: str) -> VectorProblem:
        labels: list[str] = []
        for idx, name in enumerate(raw_names):
            m = re.match(r"^\s*@?([A-Za-z])\s*$", name)
            default = "\\vec{" + "ab"[idx % 2] + "}"
            labels.append("\\vec{" + m.group(1) + "}" if m else default)
        if len(operands) == 2 and labels[0] == labels[1]:
            labels[1] = "\\vec{b}" if labels[0] != "\\vec{b}" else "\\vec{a}"
        return VectorProblem(operation=op, vectors=operands, labels=labels, source=source)

    # -- operands ---------------------------------------------------------
    def _operand(self, text: str, defs: dict[str, Matrix]) -> Optional[Matrix]:
        """Parse one operand to a column vector, or ``None`` if it is not a vector."""
        text = text.strip()
        if not text:
            return None
        name = re.match(r"^@([A-Za-z])$", text)
        if name:
            return defs.get(name.group(1))

        env = _MATRIX_ENV.fullmatch(text) or _MATRIX_ENV.fullmatch(text.strip("()[] "))
        if env:
            return self._from_matrix_env(env.group(2))

        inner = self._single_bracket_content(text)
        if inner is not None and "," in inner:
            comps = _split_depth0(inner, (",",))
            if len(comps) >= 2:
                return Matrix([self._component(c) for c in comps])

        unwrapped = text
        while True:                          # strip redundant (...) around an ijk expression
            content = self._single_bracket_content(unwrapped, parens_only=True)
            if content is None or _depth0_positions(content, ",") != []:
                break
            unwrapped = content.strip()
        return self._from_ijk(unwrapped)

    @staticmethod
    def _single_bracket_content(text: str, parens_only: bool = False) -> Optional[str]:
        """Content of ``text`` if ONE bracket pair spans all of it (``(a, b)`` yes, ``(a) + (b)`` no)."""
        text = text.strip()
        if text.startswith("\\langle") and text.endswith("\\rangle"):
            return text[len("\\langle"): -len("\\rangle")].strip()
        if not text or text[0] not in ("(" if parens_only else "([<"):
            return None
        if text[0] == "<":
            return text[1:-1].strip() if text.endswith(">") and ">" not in text[:-1] else None
        end = _matching_close(text, 0)
        return text[1:end].strip() if end == len(text) - 1 else None

    def _from_matrix_env(self, body: str) -> Optional[Matrix]:
        rows = [r for r in re.split(r"\\\\", body) if r.strip()]
        grid = [[self._component(c) for c in row.split("&")] for row in rows]
        flat = [c for row in grid for c in row]
        if len(flat) < 2 or not (len(grid) == 1 or all(len(r) == 1 for r in grid)):
            return None                      # a real matrix (m x n), not a vector
        return Matrix(flat)

    def _from_ijk(self, text: str) -> Optional[Matrix]:
        compact = text.replace(" ", "")
        if not compact or not re.search(r"[#ijk]", compact):
            return None
        coeffs = {"i": sympy.Integer(0), "j": sympy.Integer(0), "k": sympy.Integer(0)}
        pos = 0
        while pos < len(compact):
            m = _IJK_TOKEN.match(compact, pos)
            if not m or m.end() == pos:
                return None
            sign, num, axis = m.groups()
            if num is None:
                value = sympy.Integer(1)
            elif num.startswith("\\frac"):
                p, q = re.findall(r"\d+", num)
                value = Rational(int(p), int(q))
            else:
                value = Rational(num)
            coeffs[axis.lstrip("#")] += -value if sign == "-" else value
            pos = m.end()
        return Matrix([coeffs["i"], coeffs["j"], coeffs["k"]])


# ----------------------------------------------------------------------
# Solving
# ----------------------------------------------------------------------
def _ijk(vec: Matrix) -> str:
    """Format a vector as ``2\\hat{i} - \\hat{j} + 3\\hat{k}`` (or a tuple beyond 3 components)."""
    if len(vec) > 3:
        return "\\left(" + ", ".join(latex(c) for c in vec) + "\\right)"
    pieces: list[tuple[bool, str]] = []
    for comp, axis in zip(vec, ("i", "j", "k")):
        comp = sympy.simplify(comp)
        if comp == 0:
            continue
        negative = bool(comp.is_number and comp < 0)
        mag = sympy.Abs(comp) if comp.is_number else comp
        coeff = "" if mag == 1 else (latex(mag) if comp.is_number else f"\\left({latex(comp)}\\right)")
        pieces.append((negative, f"{coeff}\\hat{{{axis}}}"))
    if not pieces:
        return "\\vec{0}"
    out = ("-" if pieces[0][0] else "") + pieces[0][1]
    for negative, body in pieces[1:]:
        out += (" - " if negative else " + ") + body
    return out


def _sub(x: sympy.Expr, y: sympy.Expr) -> str:
    """``x - y`` with a negative ``y`` parenthesised: ``8 - (-1)`` instead of ``8 - -1``."""
    right = latex(y)
    return f"{latex(x)} - ({right})" if right.startswith("-") else f"{latex(x)} - {right}"


def _tup(vec: Matrix) -> str:
    return "(" + ", ".join(latex(c) for c in vec) + ")"


def _numeric(value: Any) -> Optional[float]:
    try:
        return float(sympy.N(value))
    except (TypeError, ValueError):
        return None


def _norm(vec: Matrix) -> sympy.Expr:
    return sympy.sqrt(sympy.simplify(sum(c ** 2 for c in vec)))


def solve_vector(problem: VectorProblem) -> VectorSolution:
    """Solve a :class:`VectorProblem` exactly.

    Args:
        problem: Parsed problem.

    Returns:
        A :class:`VectorSolution` with the answer, exact value and worked steps.

    Raises:
        VectorError: On mismatched dimensions, a zero vector where a direction is
            needed, or a cross product in an unsupported dimension.
    """
    op, vecs, labels = problem.operation, problem.vectors, problem.labels
    if op != "magnitude" and len({len(v) for v in vecs}) != 1:
        raise VectorError("The two vectors must have the same number of components.")
    handler = {
        "dot": _solve_dot, "cross": _solve_cross, "magnitude": _solve_magnitude,
        "add": _solve_sum, "subtract": _solve_sum, "angle": _solve_angle,
    }.get(op)
    if handler is None:
        raise VectorError(f"Unsupported vector operation: {op}")
    return handler(problem)


def _identify(problem: VectorProblem, what: str) -> tuple[str, str, str]:
    listing = ";  ".join(f"{lab} = {_tup(v)}" for lab, v in zip(problem.labels, problem.vectors))
    return ("Identify the Vectors", f"{listing}.  Task: {what}.",
            "Write the components down first; every vector formula works component by component.")


def _solve_dot(problem: VectorProblem) -> VectorSolution:
    a, b = problem.vectors
    la, lb = problem.labels
    products = [x * y for x, y in zip(a, b)]
    total = sympy.simplify(sum(products))
    subst = " + ".join(f"({latex(x)})({latex(y)})" for x, y in zip(a, b))
    parts = " + ".join(latex(p) for p in products)
    steps = [
        _identify(problem, "dot (scalar) product"),
        ("Dot Product Formula", f"{la} \\cdot {lb} = a_1 b_1 + a_2 b_2 + a_3 b_3" + (" + \\dots" if len(a) > 3 else ""),
         "Multiply matching components and add the results; the answer is a number, not a vector."),
        ("Substitute", f"{la} \\cdot {lb} = {subst}", "Put the components of both vectors into the formula."),
        ("Calculate", f"= {parts} = {latex(total)}", "Multiply each pair, then add."),
    ]
    details: dict[str, Any] = {"dot": total}
    if total.is_number:
        meaning = ("zero, so the vectors are perpendicular" if total == 0 else
                   "positive, so the angle between them is acute" if total > 0 else
                   "negative, so the angle between them is obtuse")
        steps.append(("What does it mean?", f"The dot product is {meaning}.",
                      "The sign of a·b tells us about the angle: cos θ has the same sign."))
    answer = f"{la} \\cdot {lb} = {latex(total)}"
    return VectorSolution(answer, total, [_numeric(total)] if _numeric(total) is not None else None, steps, details)


def _solve_cross(problem: VectorProblem) -> VectorSolution:
    a, b = problem.vectors
    la, lb = problem.labels
    if len(a) == 2:                           # planar vectors: z-component only
        z = sympy.simplify(a[0] * b[1] - a[1] * b[0])
        steps = [
            _identify(problem, "cross product of two plane vectors"),
            ("2-D Cross Product", f"{la} \\times {lb} = (a_1 b_2 - a_2 b_1)\\,\\hat{{k}}",
             "In the plane the cross product points along the z-axis."),
            ("Calculate", f"= ({latex(a[0])})({latex(b[1])}) - ({latex(a[1])})({latex(b[0])}) = {latex(z)}\\,\\hat{{k}}",
             "Cross-multiply the components and subtract."),
        ]
        return VectorSolution(f"{la} \\times {lb} = {latex(z)}\\,\\hat{{k}}", z,
                              [_numeric(z)] if _numeric(z) is not None else None, steps, {"area": sympy.Abs(z)})
    if len(a) != 3:
        raise VectorError("Cross product needs 3-component (or 2-component) vectors.")
    cross = Matrix([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]).applyfunc(sympy.simplify)
    area = sympy.simplify(_norm(cross))
    det = (f"\\begin{{vmatrix}} \\hat{{i}} & \\hat{{j}} & \\hat{{k}} \\\\ {latex(a[0])} & {latex(a[1])} & {latex(a[2])} \\\\ "
           f"{latex(b[0])} & {latex(b[1])} & {latex(b[2])} \\end{{vmatrix}}")
    steps = [
        _identify(problem, "cross (vector) product"),
        ("Cross Product as a Determinant", f"{la} \\times {lb} = {det}",
         "Expanding this 3×3 determinant along the first row gives the three components."),
        ("Expand", f"= \\hat{{i}}({_sub(a[1] * b[2], a[2] * b[1])}) - \\hat{{j}}({_sub(a[0] * b[2], a[2] * b[0])})"
                   f" + \\hat{{k}}({_sub(a[0] * b[1], a[1] * b[0])})",
         "Each component is a 2×2 minor; the j-term carries a minus sign."),
        ("Result", f"= {_ijk(cross)}", "Simplify the three brackets."),
        ("Area of the Parallelogram", f"|{la} \\times {lb}| = {latex(area)}",
         "The length of a×b equals the area of the parallelogram spanned by the two vectors."),
    ]
    return VectorSolution(f"{la} \\times {lb} = {_ijk(cross)}", cross, None, steps, {"area": area})


def _solve_magnitude(problem: VectorProblem) -> VectorSolution:
    (a,) = problem.vectors
    (la,) = problem.labels
    squares = " + ".join(f"({latex(c)})^2" for c in a)
    total = sympy.simplify(sum(c ** 2 for c in a))
    mag = sympy.simplify(sympy.sqrt(total))
    steps = [
        _identify(problem, "magnitude (length) and unit vector"),
        ("Magnitude Formula", f"|{la}| = \\sqrt{{a_1^2 + a_2^2 + a_3^2}}",
         "The length of a vector comes from Pythagoras' theorem applied to its components."),
        ("Substitute", f"|{la}| = \\sqrt{{{squares}}}", "Square every component."),
        ("Calculate", f"= \\sqrt{{{latex(total)}}} = {latex(mag)}", "Add the squares and take the square root."),
    ]
    details: dict[str, Any] = {"magnitude": mag}
    answer = f"|{la}| = {latex(mag)}"
    num = _numeric(mag)
    if num is not None and not (mag.is_Integer or mag.is_Rational):
        answer += f" \\approx {num:.5g}"
    if mag != 0:
        unit = (a / mag).applyfunc(sympy.simplify)
        details["unit_vector"] = unit
        steps.append(("Unit Vector", f"\\hat{{a}} = \\frac{{{la}}}{{|{la}|}} = {_tup(unit)}",
                      "Dividing a vector by its length gives a vector of length 1 in the same direction."))
    return VectorSolution(answer, mag, [num] if num is not None else None, steps, details)


def _solve_sum(problem: VectorProblem) -> VectorSolution:
    a, b = problem.vectors
    la, lb = problem.labels
    plus = problem.operation == "add"
    sign = "+" if plus else "-"
    result = (a + b if plus else a - b).applyfunc(sympy.simplify)
    comps = ", ".join(f"{latex(x)} {sign} ({latex(y)})" for x, y in zip(a, b))
    steps = [
        _identify(problem, "vector addition" if plus else "vector subtraction"),
        ("Add Component by Component" if plus else "Subtract Component by Component",
         f"{la} {sign} {lb} = ({comps})", "Vectors are added or subtracted one component at a time."),
        ("Result", f"= {_tup(result)}" + (f" = {_ijk(result)}" if len(result) <= 3 else ""), "Simplify each component."),
    ]
    return VectorSolution(f"{la} {sign} {lb} = {_tup(result)}", result, None, steps, {})


def _solve_angle(problem: VectorProblem) -> VectorSolution:
    a, b = problem.vectors
    la, lb = problem.labels
    dot = sympy.simplify(sum(x * y for x, y in zip(a, b)))
    ma, mb = sympy.simplify(_norm(a)), sympy.simplify(_norm(b))
    if ma == 0 or mb == 0:
        raise VectorError("The angle is undefined when one of the vectors is the zero vector.")
    cos_t = sympy.simplify(dot / (ma * mb))
    theta = sympy.acos(cos_t)
    deg = _numeric(theta * 180 / sympy.pi)
    steps = [
        _identify(problem, "angle between the vectors"),
        ("Angle Formula", f"\\cos\\theta = \\frac{{{la} \\cdot {lb}}}{{|{la}||{lb}|}}",
         "Rearranging a·b = |a||b|cos θ gives the cosine of the angle."),
        ("Dot Product and Lengths", f"{la} \\cdot {lb} = {latex(dot)},\\;|{la}| = {latex(ma)},\\;|{lb}| = {latex(mb)}",
         "Compute the three quantities the formula needs."),
        ("cos θ", f"\\cos\\theta = \\frac{{{latex(dot)}}}{{{latex(ma)}\\cdot {latex(mb)}}} = {latex(cos_t)}", "Substitute and simplify."),
        ("θ", f"\\theta = \\cos^{{-1}}\\left({latex(cos_t)}\\right)" + (f" \\approx {deg:.2f}^\\circ" if deg is not None else ""),
         "Take the inverse cosine to get the angle."),
    ]
    theta_tex = latex(theta)
    if "acos" in theta_tex:
        theta_tex = f"\\cos^{{-1}}\\left({latex(cos_t)}\\right)"
    answer = f"\\theta = {theta_tex}" + (f" \\approx {deg:.2f}^\\circ" if deg is not None else "")
    return VectorSolution(answer, theta, [deg] if deg is not None else None, steps,
                          {"cos_theta": cos_t, "dot": dot, "magnitudes": (ma, mb)})
