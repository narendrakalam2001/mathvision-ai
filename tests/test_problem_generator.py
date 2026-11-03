"""Unit tests for src.problem_generator."""

from __future__ import annotations

import sympy
import pytest

from src.problem_generator import ProblemGenerator
from src.problem_classifier import ProblemType
from src.latex_parser import LatexParser


@pytest.fixture()
def generator() -> ProblemGenerator:
    return ProblemGenerator(seed=42)


class TestGenerateForType:
    def test_quadratic_answer_is_correct(self, generator: ProblemGenerator) -> None:
        problem = generator.generate_for_type(ProblemType.QUADRATIC_EQUATION)
        parsed = LatexParser().parse(problem.latex_problem)
        solutions = sympy.solve(parsed.equations[0])
        # The generator's own answer string should mention every true root.
        for sol in solutions:
            assert str(sol) in problem.latex_answer or sympy.latex(sol) in problem.latex_answer

    def test_linear_difficulty_is_low(self, generator: ProblemGenerator) -> None:
        problem = generator.generate_for_type(ProblemType.LINEAR_EQUATION)
        assert problem.difficulty == 1

    def test_unsupported_type_raises(self, generator: ProblemGenerator) -> None:
        with pytest.raises(ValueError):
            generator.generate_for_type(ProblemType.WORD_PROBLEM)


class TestGenerateSimilar:
    def test_topic_keyword_routes_correctly(self, generator: ProblemGenerator) -> None:
        problem = generator.generate_similar("quadratic")
        assert problem.topic == "Quadratic Equations"

    def test_latex_reference_routes_correctly(self, generator: ProblemGenerator) -> None:
        problem = generator.generate_similar(r"\int x^3 dx")
        assert problem.topic == "Integration"

    def test_unknown_reference_falls_back_gracefully(self, generator: ProblemGenerator) -> None:
        problem = generator.generate_similar("completely unrelated gibberish topic")
        assert problem.latex_problem  # still produces *something* valid

    def test_none_reference_returns_a_problem(self, generator: ProblemGenerator) -> None:
        problem = generator.generate_similar(None)
        assert problem.latex_problem
