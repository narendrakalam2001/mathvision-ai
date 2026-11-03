"""Regression tests: CROHME-style OCR output must flow through parse -> classify -> solve.

The fine-tuned TrOCR emits CROHME-flavoured LaTeX (``\\frac 1 x``,
``\\begin {pmatrix}``, ``\\mbox { G }``). These tests pin down that every such
shape is understood end-to-end, without any network or OCR model.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.latex_parser import LatexParseError, LatexParser, normalize_crohme
from src.ocr_engine import OCRResult
from services.math_service import MathService, MathServiceError


@pytest.fixture(scope="module")
def service() -> MathService:
    return MathService()


class TestNormalizeCrohme:
    def test_braceless_frac(self) -> None:
        assert normalize_crohme(r"\frac 1 x") == r"\frac{1}{x}"

    def test_nested_frac_with_group(self) -> None:
        assert normalize_crohme(r"\frac 2 {\frac {a} {b}}") == r"\frac{2}{\frac{a}{b}}"

    def test_already_braced_frac_is_untouched(self) -> None:
        assert normalize_crohme(r"\frac{1}{x}") == r"\frac{1}{x}"

    def test_begin_end_spacing(self) -> None:
        assert normalize_crohme(r"\begin {pmatrix} 1 \end {pmatrix}") == r"\begin{pmatrix} 1 \end{pmatrix}"

    def test_mbox_removed(self) -> None:
        assert normalize_crohme(r"\mbox { G }") == "G"

    def test_sqrt_without_braces(self) -> None:
        assert normalize_crohme(r"\sqrt 2") == r"\sqrt{2}"


class TestParserShapes:
    def test_matrix_product_of_adjacent_blocks(self) -> None:
        parsed = LatexParser().parse(
            r"\begin {pmatrix} 3 & 4 \\ 1 & 2 \end {pmatrix} \begin {pmatrix} 1 & - 2 \\ - 1 & 3 \end {pmatrix}"
        )
        assert parsed.expression.tolist() == [[-1, 6], [-1, 4]]

    def test_comma_separated_system(self) -> None:
        parsed = LatexParser().parse(r"2 x + 3 y = 12 , x - y = 1")
        assert len(parsed.equations) == 2

    def test_euler_constant_is_not_a_symbol(self) -> None:
        parsed = LatexParser().parse(r"e ^ { 0 } + 1")
        assert not parsed.symbols

    def test_malformed_matrix_raises_parse_error(self) -> None:
        with pytest.raises(LatexParseError):
            LatexParser().parse(r"\begin {pmatrix} 1 & - \end {pmatrix}")


@pytest.mark.parametrize(
    ("latex", "expected_type", "answer_part"),
    [
        (r"x^2 - 5 x + 6 = 0", "quadratic_equation", "x = 2"),
        (r"y = 2 x + 3", "linear_equation", "y = 2 x + 3"),
        (r"y = x ^ { 2 } - 4 x + 3", "polynomial", "x^{2}"),
        (r"\sqrt { 49 }", "arithmetic", "7"),
        (r"\sin ^ { 2 } x + \cos ^ { 2 } x", "trigonometry", "1"),
        (r"\frac 1 x", "expression", r"\frac{1}{x}"),
        (r"2 x + 3 y = 12 , x - y = 1", "simultaneous_equations", "x = 3"),
        (r"\frac { d } { d x } ( x ^ { 3 } )", "differentiation", "3 x^{2}"),
        (r"\int e ^ { x } d x", "integration", "e^{x} + C"),
        (r"\begin {pmatrix} 3 & 4 \\ 1 & 2 \end {pmatrix}", "matrix", r"\det = 2"),
    ],
)
def test_crohme_style_problem_is_solved(service: MathService, latex: str, expected_type: str, answer_part: str) -> None:
    result = service.solve_latex(latex)
    assert result["problem_type"] == expected_type
    assert answer_part in result["answer"]
    assert result["steps"], "every solved problem needs at least one step"


class TestGraphs:
    def test_line_gets_a_graph(self, service: MathService) -> None:
        assert service.solve_latex(r"y = 2 x + 3")["graph_base64"]

    def test_arithmetic_has_no_graph(self, service: MathService) -> None:
        assert service.solve_latex(r"\sqrt { 49 }")["graph_base64"] is None

    def test_line_steps_report_slope_and_intercept(self, service: MathService) -> None:
        content = " ".join(s["content"] for s in service.solve_latex(r"y = 2 x + 3")["steps"])
        assert "slope" in content


class _FakeEngine:
    """Returns one scripted OCR result per call (no model needed)."""

    def __init__(self, latexes: list[str]) -> None:
        self._latexes = list(latexes)

    def recognize(self, image: np.ndarray) -> OCRResult:
        assert image.shape == (96, 384), "OCR must receive the normalised training-style canvas"
        latex = self._latexes.pop(0)
        return OCRResult(latex=latex, raw_latex=latex, confidence=0.95, engine_used="trocr", is_valid=True)


def _two_line_page() -> np.ndarray:
    page = np.full((600, 1000, 3), 255, np.uint8)
    cv2.putText(page, "x+2=5", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
    cv2.putText(page, "y=3x", (80, 460), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
    return page


class TestSolveImage:
    def test_all_lines_recognised_first_is_solved(self) -> None:
        svc = MathService()
        svc._ocr_engine = _FakeEngine([r"x + 2 = 5", r"y = 3 x"])
        result = svc.solve_image(_two_line_page())
        assert [ln["latex"] for ln in result["ocr"]["lines"]] == ["x + 2 = 5", "y = 3 x"]
        assert result["ocr"]["selected_line"] == 0
        assert result["answer"] == "x = 3"

    def test_line_index_selects_second_line(self) -> None:
        svc = MathService()
        svc._ocr_engine = _FakeEngine([r"x + 2 = 5", r"y = 3 x"])
        result = svc.solve_image(_two_line_page(), line_index=1)
        assert result["ocr"]["selected_line"] == 1
        assert result["problem_type"] == "linear_equation"
        assert "y = 3 x" in result["answer"]

    def test_blank_image_asks_for_a_retake(self) -> None:
        svc = MathService()
        svc._ocr_engine = _FakeEngine([])
        result = svc.solve_image(np.full((400, 600, 3), 255, np.uint8))
        assert result["retake_recommended"] is True


@pytest.mark.parametrize("latex", [
    r"x^{2} - x - 90 = 0", r"x^{2} + 3 x - 10 = 0", r"x^{2} + 8 x + 16 = 0", r"2 x^{2} - 7 x + 3 = 0",
])
def test_quadratics_with_negative_terms_are_classified(service: MathService, latex: str) -> None:
    """parse_latex yields nested unevaluated Adds for these; they once classified as 'unknown'."""
    result = service.solve_latex(latex)
    assert result["problem_type"] == "quadratic_equation"
    assert result["answer"].startswith("x =")


def test_unsupported_vectors_fail_gracefully(service: MathService) -> None:
    with pytest.raises(MathServiceError) as info:
        service.solve_latex(r"\text{Find } \vec{a} \cdot \vec{b}")
    assert "could not be solved" in str(info.value) or "could not understand" in str(info.value)
