"""Unit tests for src.latex_parser."""

from __future__ import annotations

import pytest
import sympy

from src.latex_parser import LatexParseError, LatexParser


@pytest.fixture()
def parser() -> LatexParser:
    return LatexParser()


class TestParseEquation:
    def test_simple_quadratic(self, parser: LatexParser) -> None:
        eq = parser.parse_equation(r"x^2 - 5x + 6 = 0")
        assert isinstance(eq, sympy.Eq)
        solutions = sympy.solve(eq)
        assert set(solutions) == {2, 3}

    def test_linear_equation(self, parser: LatexParser) -> None:
        eq = parser.parse_equation(r"2x + 3 = 7")
        solutions = sympy.solve(eq)
        assert solutions == [2]

    def test_missing_equals_raises(self, parser: LatexParser) -> None:
        with pytest.raises(LatexParseError):
            parser.parse_equation(r"x^2 - 5x + 6")

    def test_too_many_equals_raises(self, parser: LatexParser) -> None:
        with pytest.raises(LatexParseError):
            parser.parse_equation(r"x = y = 5")


class TestParseSystem:
    def test_two_equation_system(self, parser: LatexParser) -> None:
        equations = parser.parse_system(r"2x + y = 5 \\ x - y = 1")
        assert len(equations) == 2
        solution = sympy.solve(equations)
        assert solution[sympy.Symbol("x")] == 2
        assert solution[sympy.Symbol("y")] == 1

    def test_empty_system_raises(self, parser: LatexParser) -> None:
        with pytest.raises(LatexParseError):
            parser.parse_system("   ")


class TestParse:
    def test_bare_expression_returns_expression(self, parser: LatexParser) -> None:
        result = parser.parse(r"x^2 + 1")
        assert result.expression is not None
        assert not result.equations

    def test_single_equation_detected(self, parser: LatexParser) -> None:
        result = parser.parse(r"x^2 - 5x + 6 = 0")
        assert len(result.equations) == 1
        assert result.expression is None

    def test_system_detected(self, parser: LatexParser) -> None:
        result = parser.parse(r"2x + y = 5 \\ x - y = 1")
        assert len(result.equations) == 2

    def test_symbols_extracted(self, parser: LatexParser) -> None:
        result = parser.parse(r"x^2 - 5x + 6 = 0")
        assert sympy.Symbol("x") in result.symbols

    def test_empty_input_raises(self, parser: LatexParser) -> None:
        with pytest.raises(LatexParseError):
            parser.parse("")

    def test_ocr_artifact_cleanup(self, parser: LatexParser) -> None:
        # Em-dash / smart quotes / cdot should not break parsing.
        result = parser.parse(r"2 \cdot x + 3 = 7")
        assert len(result.equations) == 1
