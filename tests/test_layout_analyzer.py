"""Unit tests for src.layout_analyzer (synthetic images, no datasets needed)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.layout_analyzer import CANVAS_H, CANVAS_W, LayoutAnalyzer


@pytest.fixture()
def analyzer() -> LayoutAnalyzer:
    return LayoutAnalyzer()


def _page(*texts: str, height: int = 400, width: int = 900) -> np.ndarray:
    """White page with one rendered text line per entry."""
    img = np.full((height, width), 255, np.uint8)
    for i, text in enumerate(texts):
        cv2.putText(img, text, (30, 90 + i * 150), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 6)
    return img


class TestLineSplitting:
    def test_blank_page_has_no_lines(self, analyzer: LayoutAnalyzer) -> None:
        assert analyzer.analyze(np.full((300, 300), 255, np.uint8)) == []

    def test_single_line(self, analyzer: LayoutAnalyzer) -> None:
        assert len(analyzer.analyze(_page("x+2=5"))) == 1

    def test_two_lines_split_top_to_bottom(self, analyzer: LayoutAnalyzer) -> None:
        lines = analyzer.analyze(_page("x+2=5", "y=3x"))
        assert len(lines) == 2
        assert lines[0].bbox[1] < lines[1].bbox[1]
        assert [l.index for l in lines] == [0, 1]

    def test_speck_noise_is_ignored(self, analyzer: LayoutAnalyzer) -> None:
        page = _page("x+2=5")
        page[350, 100] = 0
        page[352, 500] = 0
        assert len(analyzer.analyze(page)) == 1

    def test_max_lines_cap(self, analyzer: LayoutAnalyzer) -> None:
        assert len(analyzer.analyze(_page("1+1", "2+2"), max_lines=1)) == 1

    def test_bgr_input_accepted(self, analyzer: LayoutAnalyzer) -> None:
        bgr = cv2.cvtColor(_page("x+2=5"), cv2.COLOR_GRAY2BGR)
        assert len(analyzer.analyze(bgr)) == 1


class TestNormalization:
    def test_output_shape_and_dtype(self, analyzer: LayoutAnalyzer) -> None:
        line = analyzer.analyze(_page("x+2=5"))[0]
        assert line.image.shape == (CANVAS_H, CANVAS_W)
        assert line.image.dtype == np.uint8

    def test_black_ink_on_white_background(self, analyzer: LayoutAnalyzer) -> None:
        img = analyzer.analyze(_page("x+2=5"))[0].image
        assert img[0, 0] == 255 and (img == 0).any()
        assert (img == 255).mean() > 0.5

    def test_ink_stays_inside_padding(self, analyzer: LayoutAnalyzer) -> None:
        ys, xs = np.where(analyzer.analyze(_page("x+2=5"))[0].image == 0)
        assert ys.min() >= 5 and ys.max() <= CANVAS_H - 5
        assert xs.min() >= 5 and xs.max() <= CANVAS_W - 5

    def test_very_wide_expression_still_fits(self, analyzer: LayoutAnalyzer) -> None:
        img = analyzer.analyze(_page("1+2+3+4+5+6+7+8+9", width=1800))[0].image
        assert img.shape == (CANVAS_H, CANVAS_W) and (img == 0).any()
