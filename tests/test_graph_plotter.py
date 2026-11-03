"""Unit tests for src.graph_plotter."""

from __future__ import annotations

import base64

import pytest
import sympy

from src.graph_plotter import GraphPlotter, GraphPlottingError

x, y = sympy.symbols("x y")


@pytest.fixture()
def plotter() -> GraphPlotter:
    return GraphPlotter(figsize=(4, 3), dpi=60)  # small/fast for tests


class TestPlotFunction:
    def test_returns_valid_data_uri(self, plotter: GraphPlotter) -> None:
        result = plotter.plot_function(x**2 - 5 * x + 6, x, roots=[2, 3])
        assert result.startswith("data:image/png;base64,")

    def test_base64_payload_decodes(self, plotter: GraphPlotter) -> None:
        result = plotter.plot_function(x**2, x)
        payload = result.split(",", 1)[1]
        decoded = base64.b64decode(payload)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_multivariable_expression_raises(self, plotter: GraphPlotter) -> None:
        with pytest.raises(GraphPlottingError):
            plotter.plot_function(x + y, x)

    def test_custom_range_respected(self, plotter: GraphPlotter) -> None:
        result = plotter.plot_function(sympy.sin(x), x, x_range=(-5, 5))
        assert result.startswith("data:image/png;base64,")


class TestPlotProbabilityBar:
    def test_returns_valid_data_uri(self, plotter: GraphPlotter) -> None:
        result = plotter.plot_probability_bar(["A", "B", "C"], [0.2, 0.5, 0.3])
        assert result.startswith("data:image/png;base64,")
