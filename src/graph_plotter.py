"""Graph plotting utilities (feeds Screen 3's graph panel and MathBot's
``plot_graph`` tool).

Renders SymPy expressions/equations as Matplotlib figures and returns
them as base64-encoded PNGs, ready to embed directly in a Streamlit
``st.image`` call or a FastAPI JSON response (``data:image/png;base64,...``).

Supports:
    * 2D function plots (``y = f(x)``), e.g. parabolas, trig curves
    * Root/x-intercept markers (auto-computed via SymPy)
    * Simple statistics bar charts (for probability/stats problems)

Example:
    >>> import sympy
    >>> from src.graph_plotter import GraphPlotter
    >>> x = sympy.Symbol("x")
    >>> plotter = GraphPlotter()
    >>> b64 = plotter.plot_function(x**2 - 5*x + 6, x, roots=[2, 3])
    >>> b64.startswith("data:image/png;base64,")
    True
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless rendering — safe for servers/notebooks alike

import matplotlib.pyplot as plt
import numpy as np
import sympy

from src.math_text import pretty_math

logger = logging.getLogger(__name__)

# MathVision AI brand palette (matches the frontend spec).
COLOR_PRIMARY = "#1A237E"
COLOR_SECONDARY = "#FF6F00"
COLOR_SUCCESS = "#2E7D32"
COLOR_BACKGROUND = "#FFFFFF"
COLOR_CARD = "#E3F2FD"


def _math_title(expr_latex: str) -> str:
    """Return a matplotlib title that renders ``y = <expr>`` as real math.

    Uses mathtext (``$...$``); if matplotlib cannot parse the LaTeX, falls back to a
    readable Unicode string so the plot never fails because of its title.
    """
    candidate = f"$y = {expr_latex}$"
    try:
        from matplotlib.mathtext import MathTextParser

        MathTextParser("path").parse(candidate)
        return candidate
    except Exception:  # noqa: BLE001 - any parse problem -> plain text
        return f"y = {pretty_math(expr_latex)}"


class GraphPlottingError(Exception):
    """Raised when an expression cannot be plotted (e.g. too many free variables)."""


@dataclass
class PlotResult:
    """Output of a plotting call.

    Attributes:
        base64_png: A ``data:image/png;base64,...`` string ready for
            direct embedding in HTML/Streamlit/JSON.
        x_range: The (min, max) x-range actually plotted.
        notes: Any warnings generated while auto-selecting the plot range.
    """

    base64_png: str
    x_range: tuple[float, float]
    notes: list[str]


class GraphPlotter:
    """Renders SymPy expressions as branded Matplotlib figures."""

    def __init__(self, figsize: tuple[float, float] = (7.0, 4.5), dpi: int = 120) -> None:
        """Create a plotter.

        Args:
            figsize: Matplotlib figure size in inches.
            dpi: Resolution of the rendered PNG.
        """
        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------------------
    def _figure_to_base64(self, fig: "plt.Figure") -> str:
        """Serialize a Matplotlib figure to a base64 PNG data URI."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=COLOR_BACKGROUND)
        plt.close(fig)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _auto_range(self, roots: Optional[Sequence[float]]) -> tuple[float, float]:
        """Pick a sensible x-range, centering on any known roots."""
        if roots:
            finite_roots = [r for r in roots if np.isfinite(r)]
            if finite_roots:
                span = max(finite_roots) - min(finite_roots)
                pad = max(span * 0.6, 2.0)
                return (min(finite_roots) - pad, max(finite_roots) + pad)
        return (-10.0, 10.0)

    # ------------------------------------------------------------------
    def plot_function(
        self,
        expr: sympy.Expr,
        symbol: sympy.Symbol,
        roots: Optional[Sequence] = None,
        x_range: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        num_points: int = 400,
    ) -> str:
        """Plot ``y = expr`` over a real range and return a base64 PNG.

        Args:
            expr: The SymPy expression to plot (function of ``symbol``).
            symbol: The independent variable.
            roots: Optional known roots to mark and to auto-center the
                x-range around.
            x_range: Explicit (min, max) x-range; overrides auto-selection.
            title: Optional plot title.
            num_points: Number of sample points across the x-range.

        Returns:
            A ``data:image/png;base64,...`` string.

        Raises:
            GraphPlottingError: If the expression has more than one free
                variable, or cannot be numerically evaluated.
        """
        free_syms = expr.free_symbols
        if len(free_syms) > 1:
            raise GraphPlottingError(
                f"Cannot plot a 2D graph of an expression with {len(free_syms)} free variables: {free_syms}"
            )

        notes: list[str] = []
        numeric_roots = self._safe_floats(roots) if roots else []
        x_min, x_max = x_range or self._auto_range(numeric_roots)

        func = sympy.lambdify(symbol, expr, modules=["numpy"])
        xs = np.linspace(x_min, x_max, num_points)

        try:
            ys = func(xs)
            ys = np.asarray(ys, dtype=float)
        except Exception as exc:  # noqa: BLE001
            raise GraphPlottingError(f"Could not numerically evaluate expression: {exc}") from exc

        # Clip extreme values (e.g. near asymptotes) so the plot stays readable.
        finite_mask = np.isfinite(ys)
        if finite_mask.sum() < len(ys) * 0.5:
            notes.append("Function has many undefined/infinite points in this range.")
        if finite_mask.any():
            y_abs_cap = np.percentile(np.abs(ys[finite_mask]), 98) * 3 or 1.0
            ys = np.clip(ys, -y_abs_cap, y_abs_cap)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(xs, ys, color=COLOR_PRIMARY, linewidth=2.2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_facecolor(COLOR_CARD)
        ax.grid(True, alpha=0.3)

        if numeric_roots:
            ax.scatter(
                numeric_roots,
                [0] * len(numeric_roots),
                color=COLOR_SECONDARY,
                zorder=5,
                s=60,
                label="Roots",
            )
            for r in numeric_roots:
                ax.annotate(f"{r:.2f}", (r, 0), textcoords="offset points", xytext=(0, 10), ha="center")
            ax.legend()

        ax.set_xlabel(str(symbol))
        ax.set_ylabel(f"f({symbol})")
        ax.set_title(pretty_math(title) if title else _math_title(sympy.latex(expr)), fontsize=12, color=COLOR_PRIMARY)

        b64 = self._figure_to_base64(fig)
        return b64

    # ------------------------------------------------------------------
    def plot_probability_bar(
        self, labels: Sequence[str], values: Sequence[float], title: str = "Probability Distribution"
    ) -> str:
        """Render a simple bar chart for probability/statistics problems.

        Args:
            labels: Category labels (x-axis).
            values: Corresponding probabilities/frequencies (y-axis).
            title: Chart title.

        Returns:
            A ``data:image/png;base64,...`` string.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS] * (len(labels) // 3 + 1)
        ax.bar(labels, values, color=colors[: len(labels)])
        ax.set_facecolor(COLOR_CARD)
        ax.set_title(pretty_math(title), fontsize=12, color=COLOR_PRIMARY)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=20, ha="right")
        return self._figure_to_base64(fig)

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_floats(values: Sequence) -> list[float]:
        floats = []
        for v in values:
            try:
                floats.append(float(v))
            except (TypeError, ValueError):
                continue
        return floats


if __name__ == "__main__":
    x = sympy.Symbol("x")
    plotter = GraphPlotter()
    result = plotter.plot_function(x**2 - 5 * x + 6, x, roots=[2, 3], title="x^2 - 5x + 6 = 0")
    print(f"Generated PNG data URI, length={len(result)} chars")
