"""Math problem-type classification (Module 3A).

Given a parsed problem (from :mod:`src.latex_parser`) or its raw
LaTeX, determines which category it belongs to so
:mod:`src.math_solver` (Day 2) can route it to the right SymPy routine
(or to the WolframAlpha/Gemini fallbacks).

Two complementary strategies are combined:

    1. **Structural/rule-based classification** (primary, fast, free):
       inspects the SymPy expression tree — presence of ``Derivative``,
       ``Integral``, ``MatrixSymbol``, trig functions, number of
       equations/unknowns, polynomial degree, etc. This alone correctly
       classifies the overwhelming majority of NCERT/JEE-style problems
       and requires no network call.

    2. **Keyword/NLP fallback**: when the structural signal is
       ambiguous (e.g. a bare expression with no equation, or a word
       problem containing both text and an equation), falls back to
       keyword matching on any accompanying problem text.

Example:
    >>> from src.latex_parser import LatexParser
    >>> from src.problem_classifier import ProblemClassifier
    >>> parsed = LatexParser().parse("x^2 - 5x + 6 = 0")
    >>> ProblemClassifier().classify(parsed).problem_type
    <ProblemType.QUADRATIC_EQUATION: 'quadratic_equation'>
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import sympy
from sympy import Derivative, Eq, Integral, Limit, MatrixBase, Poly

from src.latex_parser import ParsedProblem

logger = logging.getLogger(__name__)


class ProblemType(str, Enum):
    """All problem categories MathVision AI knows how to route."""

    LINEAR_EQUATION = "linear_equation"
    QUADRATIC_EQUATION = "quadratic_equation"
    POLYNOMIAL = "polynomial"
    SIMULTANEOUS_EQUATIONS = "simultaneous_equations"
    TRIGONOMETRY = "trigonometry"
    DIFFERENTIATION = "differentiation"
    INTEGRATION = "integration"
    LIMIT = "limit"
    MATRIX = "matrix"
    PROBABILITY_STATISTICS = "probability_statistics"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    WORD_PROBLEM = "word_problem"
    VECTOR = "vector"
    ARITHMETIC = "arithmetic"      # numeric expression with no variables, e.g. \sqrt{49}
    EXPRESSION = "expression"      # expression with variables and no '=', e.g. simplify
    UNKNOWN = "unknown"


_TRIG_FUNCS = (sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc)

_KEYWORD_MAP: dict[ProblemType, tuple[str, ...]] = {
    ProblemType.PROBABILITY_STATISTICS: (
        "probability", "mean", "median", "mode", "variance", "standard deviation",
        "sambhavna", "ausat",
    ),
    ProblemType.GEOMETRY: (
        "area", "perimeter", "volume", "triangle", "circle", "radius", "diameter",
        "kshetrafal", "parimaap",
    ),
    ProblemType.NUMBER_THEORY: (
        "prime", "divisor", "factor of", "gcd", "hcf", "lcm", "remainder",
    ),
    ProblemType.WORD_PROBLEM: (
        "how many", "find the number of", "a train", "a man", "speed", "distance",
        "kitna", "kitne",
    ),
}


@dataclass
class ClassificationResult:
    """Output of the classifier.

    Attributes:
        problem_type: Best-guess :class:`ProblemType`.
        confidence: Heuristic confidence in [0, 1].
        signals: Human-readable notes on which structural/keyword
            signals contributed to the decision (useful for debugging
            and for the "why this classification" explanation).
    """

    problem_type: ProblemType
    confidence: float
    signals: list[str] = field(default_factory=list)


class ProblemClassifier:
    """Rule-based + keyword-fallback classifier for math problems."""

    def classify(
        self, parsed: ParsedProblem, problem_text: Optional[str] = None
    ) -> ClassificationResult:
        """Classify a parsed problem.

        Args:
            parsed: Output of :class:`src.latex_parser.LatexParser`.
            problem_text: Optional accompanying free text (e.g. OCR'd
                word-problem narrative, or text typed by the student),
                used only when structural signals are ambiguous.

        Returns:
            A :class:`ClassificationResult`.
        """
        signals: list[str] = []

        if parsed.vector is not None:
            signals.append(f"Vector operation: {parsed.vector.operation}")
            return ClassificationResult(ProblemType.VECTOR, 0.95, signals)

        # --- Simultaneous equations: structural, unambiguous ------------
        if len(parsed.equations) >= 2:
            signals.append(f"{len(parsed.equations)} equations detected")
            return ClassificationResult(ProblemType.SIMULTANEOUS_EQUATIONS, 0.97, signals)

        target = parsed.equations[0] if parsed.equations else parsed.expression
        if target is None:
            return ClassificationResult(ProblemType.UNKNOWN, 0.0, ["No parsable content"])

        expr_tree = target.lhs - target.rhs if isinstance(target, Eq) else target

        # --- Calculus signals: unambiguous once an operator is present --
        if expr_tree.has(Integral):
            signals.append("Integral operator present")
            return ClassificationResult(ProblemType.INTEGRATION, 0.98, signals)
        if expr_tree.has(Derivative):
            signals.append("Derivative operator present")
            return ClassificationResult(ProblemType.DIFFERENTIATION, 0.98, signals)
        if expr_tree.has(Limit):
            signals.append("Limit operator present")
            return ClassificationResult(ProblemType.LIMIT, 0.98, signals)

        # --- Matrix -------------------------------------------------------
        if isinstance(target, MatrixBase) or expr_tree.has(MatrixBase):
            signals.append("Matrix object present")
            return ClassificationResult(ProblemType.MATRIX, 0.95, signals)

        # --- Trigonometry ---------------------------------------------------
        if any(expr_tree.has(fn) for fn in _TRIG_FUNCS):
            signals.append("Trigonometric function present")
            return ClassificationResult(ProblemType.TRIGONOMETRY, 0.93, signals)

        # --- Bare expression (no '='): evaluate or simplify ---------------
        if not isinstance(target, Eq):
            if not expr_tree.free_symbols:
                signals.append("No variables - numeric expression")
                return ClassificationResult(ProblemType.ARITHMETIC, 0.9, signals)
            signals.append("Expression without '=' - simplify")
            return ClassificationResult(ProblemType.EXPRESSION, 0.85, signals)

        # --- Polynomial degree-based classification (equations only) ------
        if isinstance(target, Eq):
            free_syms = list(expr_tree.free_symbols)
            if len(free_syms) == 1:
                symbol = free_syms[0]
                try:
                    degree = Poly(expr_tree, symbol).degree()
                    signals.append(f"Single-variable polynomial, degree={degree}")
                    if degree == 1:
                        return ClassificationResult(ProblemType.LINEAR_EQUATION, 0.95, signals)
                    if degree == 2:
                        return ClassificationResult(ProblemType.QUADRATIC_EQUATION, 0.95, signals)
                    if degree >= 3:
                        return ClassificationResult(ProblemType.POLYNOMIAL, 0.92, signals)
                except sympy.PolynomialError:
                    signals.append("Not a polynomial in the classical sense")
            elif len(free_syms) == 2:
                try:
                    degree = Poly(expr_tree, *free_syms).total_degree()
                    signals.append(f"Relation between two variables (graphable), degree={degree}")
                    kind = ProblemType.LINEAR_EQUATION if degree == 1 else ProblemType.POLYNOMIAL
                    return ClassificationResult(kind, 0.9, signals)
                except sympy.PolynomialError:
                    signals.append("Two-variable relation is not polynomial")
            elif len(free_syms) == 0:
                signals.append("No free symbols — constant equation")

        # --- Keyword fallback (word problems, stats, geometry, etc.) -----
        if problem_text:
            keyword_result = self._classify_by_keywords(problem_text)
            if keyword_result is not None:
                signals.append("Classified via keyword match on problem text")
                return ClassificationResult(keyword_result, 0.6, signals)

        signals.append("No structural or keyword signal matched")
        return ClassificationResult(ProblemType.UNKNOWN, 0.2, signals)

    @staticmethod
    def _classify_by_keywords(text: str) -> Optional[ProblemType]:
        """Match free text against :data:`_KEYWORD_MAP`.

        Args:
            text: Free-form problem text.

        Returns:
            The first matching :class:`ProblemType`, or ``None``.
        """
        lowered = text.lower()
        for ptype, keywords in _KEYWORD_MAP.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in keywords):
                return ptype
        return None


if __name__ == "__main__":
    from src.latex_parser import LatexParser

    latex_parser = LatexParser()
    classifier = ProblemClassifier()

    test_cases = [
        r"x^2 - 5x + 6 = 0",
        r"2x + 3 = 7",
        r"\int_0^1 x^2 dx",
        r"\sin(x) + \cos(x) = 1",
        r"2x + y = 5 \\ x - y = 1",
    ]
    for case in test_cases:
        parsed_problem = latex_parser.parse(case)
        result = classifier.classify(parsed_problem)
        print(f"{case!r} -> {result.problem_type.value} (confidence={result.confidence})")
