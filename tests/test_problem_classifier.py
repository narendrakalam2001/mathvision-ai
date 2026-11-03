"""Unit tests for src.problem_classifier."""

from __future__ import annotations

import pytest

from src.latex_parser import LatexParser
from src.problem_classifier import ProblemClassifier, ProblemType


@pytest.fixture()
def parser() -> LatexParser:
    return LatexParser()


@pytest.fixture()
def classifier() -> ProblemClassifier:
    return ProblemClassifier()


@pytest.mark.parametrize(
    "latex_str,expected_type",
    [
        (r"2x + 3 = 7", ProblemType.LINEAR_EQUATION),
        (r"x^2 - 5x + 6 = 0", ProblemType.QUADRATIC_EQUATION),
        (r"x^3 - 6x^2 + 11x - 6 = 0", ProblemType.POLYNOMIAL),
        (r"\sin(x) + \cos(x) = 1", ProblemType.TRIGONOMETRY),
        (r"2x + y = 5 \\ x - y = 1", ProblemType.SIMULTANEOUS_EQUATIONS),
    ],
)
def test_classification(
    parser: LatexParser,
    classifier: ProblemClassifier,
    latex_str: str,
    expected_type: ProblemType,
) -> None:
    parsed = parser.parse(latex_str)
    result = classifier.classify(parsed)
    assert result.problem_type == expected_type
    assert 0.0 <= result.confidence <= 1.0


def test_differentiation_detected(parser: LatexParser, classifier: ProblemClassifier) -> None:
    parsed = parser.parse(r"\frac{d}{dx} x^2")
    result = classifier.classify(parsed)
    assert result.problem_type == ProblemType.DIFFERENTIATION


def test_integration_detected(parser: LatexParser, classifier: ProblemClassifier) -> None:
    parsed = parser.parse(r"\int x^2 dx")
    result = classifier.classify(parsed)
    assert result.problem_type == ProblemType.INTEGRATION


def test_keyword_fallback_probability(parser: LatexParser, classifier: ProblemClassifier) -> None:
    parsed = parser.parse(r"x = 5")  # ambiguous linear form
    result = classifier.classify(parsed, problem_text="Find the probability of drawing a red card")
    # Structural signal (linear equation) still wins unless truly ambiguous;
    # this test documents current precedence rather than asserting keyword override.
    assert result.problem_type in {ProblemType.LINEAR_EQUATION, ProblemType.PROBABILITY_STATISTICS}


def test_unknown_for_no_content(classifier: ProblemClassifier) -> None:
    from src.latex_parser import ParsedProblem

    empty = ParsedProblem(equations=[], expression=None, symbols=set(), raw_latex="")
    result = classifier.classify(empty)
    assert result.problem_type == ProblemType.UNKNOWN
