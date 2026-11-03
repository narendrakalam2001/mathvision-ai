"""Unit tests for src.step_extractor."""

from __future__ import annotations

import pytest

from src.latex_parser import LatexParser
from src.math_solver import MathSolver
from src.problem_classifier import ProblemClassifier
from src.step_extractor import StepExtractor


@pytest.fixture()
def pipeline():
    parser = LatexParser()
    classifier = ProblemClassifier()
    solver = MathSolver(wolfram_app_id="", gemini_api_key="")
    extractor = StepExtractor()
    return parser, classifier, solver, extractor


def test_quadratic_steps_include_verification(pipeline) -> None:
    parser, classifier, solver, extractor = pipeline
    parsed = parser.parse(r"x^2 - 5x + 6 = 0")
    ptype = classifier.classify(parsed).problem_type
    result = solver.solve(parsed, ptype)
    sequence = extractor.extract(ptype, result)

    assert sequence.final_answer
    assert any(step.verified for step in sequence.steps)
    assert sequence.steps[0].title == "Identify Equation Type"
    # Step numbers should be sequential starting at 1
    assert [s.number for s in sequence.steps] == list(range(1, len(sequence.steps) + 1))


def test_linear_steps_produced(pipeline) -> None:
    parser, classifier, solver, extractor = pipeline
    parsed = parser.parse(r"2x + 3 = 7")
    ptype = classifier.classify(parsed).problem_type
    result = solver.solve(parsed, ptype)
    sequence = extractor.extract(ptype, result)

    assert len(sequence.steps) >= 3
    assert "2" in sequence.final_answer


def test_derivative_steps_produced(pipeline) -> None:
    parser, classifier, solver, extractor = pipeline
    parsed = parser.parse(r"\frac{d}{dx} x^3")
    ptype = classifier.classify(parsed).problem_type
    result = solver.solve(parsed, ptype)
    sequence = extractor.extract(ptype, result)

    assert len(sequence.steps) == 3
    assert sequence.difficulty >= 1


def test_every_step_has_a_why(pipeline) -> None:
    parser, classifier, solver, extractor = pipeline
    parsed = parser.parse(r"x^2 - 5x + 6 = 0")
    ptype = classifier.classify(parsed).problem_type
    result = solver.solve(parsed, ptype)
    sequence = extractor.extract(ptype, result)

    for step in sequence.steps:
        assert step.why, f"Step {step.number} ({step.title}) is missing a 'why' explanation"


def test_estimate_difficulty_bounds(pipeline) -> None:
    from src.problem_classifier import ProblemType

    _, _, _, extractor = pipeline
    for ptype in ProblemType:
        difficulty = extractor.estimate_difficulty(ptype, num_steps=3)
        assert 1 <= difficulty <= 3
