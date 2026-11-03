"""Unit tests for src.math_solver (SymPy engine only — no network calls)."""

from __future__ import annotations

import pytest

from src.latex_parser import LatexParser
from src.math_solver import MathSolver, SolverError, WolframQuotaTracker
from src.problem_classifier import ProblemClassifier, ProblemType


@pytest.fixture()
def parser() -> LatexParser:
    return LatexParser()


@pytest.fixture()
def classifier() -> ProblemClassifier:
    return ProblemClassifier()


@pytest.fixture()
def solver() -> MathSolver:
    # No Wolfram/Gemini keys configured -> exercises the pure-SymPy path.
    return MathSolver(wolfram_app_id="", gemini_api_key="")


class TestAlgebraicSolving:
    def test_linear_equation(self, parser, classifier, solver) -> None:
        parsed = parser.parse(r"2x + 3 = 7")
        ptype = classifier.classify(parsed).problem_type
        result = solver.solve(parsed, ptype)
        assert result.engine_used == "sympy"
        assert "x = 2" in result.answer

    def test_quadratic_equation(self, parser, classifier, solver) -> None:
        parsed = parser.parse(r"x^2 - 5x + 6 = 0")
        ptype = classifier.classify(parsed).problem_type
        result = solver.solve(parsed, ptype)
        assert "2" in result.answer and "3" in result.answer
        assert result.intermediate["discriminant"] == 1

    def test_simultaneous_equations(self, parser, classifier, solver) -> None:
        parsed = parser.parse(r"2x + y = 5 \\ x - y = 1")
        ptype = classifier.classify(parsed).problem_type
        assert ptype == ProblemType.SIMULTANEOUS_EQUATIONS
        result = solver.solve(parsed, ptype)
        assert "x = 2" in result.answer
        assert "y = 1" in result.answer


class TestCalculus:
    def test_differentiation(self, parser, classifier, solver) -> None:
        parsed = parser.parse(r"\frac{d}{dx} x^3")
        ptype = classifier.classify(parsed).problem_type
        result = solver.solve(parsed, ptype)
        assert "3" in result.answer  # 3x^2

    def test_integration(self, parser, classifier, solver) -> None:
        parsed = parser.parse(r"\int x^2 dx")
        ptype = classifier.classify(parsed).problem_type
        result = solver.solve(parsed, ptype)
        assert "+ C" in result.answer


class TestMatrix:
    def test_determinant(self, parser, classifier, solver) -> None:
        from sympy import Matrix
        from src.latex_parser import ParsedProblem

        matrix = Matrix([[1, 2], [3, 4]])
        parsed = ParsedProblem(equations=[], expression=matrix, symbols=set(), raw_latex="matrix")
        result = solver.solve(parsed, ProblemType.MATRIX)
        assert result.intermediate["determinant"] == -2


class TestErrorHandling:
    def test_unsupported_type_raises(self, parser, classifier, solver) -> None:
        from src.latex_parser import ParsedProblem
        import sympy

        # A bare unsolved symbol (no equation/derivative/etc.) with an
        # UNKNOWN classification should fail gracefully.
        parsed = ParsedProblem(equations=[], expression=None, symbols=set(), raw_latex="")
        with pytest.raises(SolverError):
            solver.solve(parsed, ProblemType.UNKNOWN)


class TestWolframQuotaTracker:
    def test_quota_decrements(self, tmp_path) -> None:
        tracker = WolframQuotaTracker(quota=2, storage_path=tmp_path / "quota.json")
        assert tracker.remaining() == 2
        tracker.record_call()
        assert tracker.remaining() == 1
        tracker.record_call()
        assert tracker.remaining() == 0

    def test_quota_exhausted_raises(self, tmp_path) -> None:
        from src.math_solver import QuotaExceededError

        tracker = WolframQuotaTracker(quota=1, storage_path=tmp_path / "quota.json")
        tracker.record_call()
        with pytest.raises(QuotaExceededError):
            tracker.record_call()
