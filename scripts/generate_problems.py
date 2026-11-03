#!/usr/bin/env python3
"""Generate a synthetic NCERT-aligned math problem dataset.

Produces ~5,000 problems spanning Class 8-12 NCERT topics (Algebra,
Trigonometry, Calculus, Coordinate Geometry, Statistics, Probability,
Matrices, Vectors). Each problem is:

    1. Generated + solved symbolically with SymPy (guaranteeing a
       verified, exact ground-truth answer)
    2. Rendered as a handwritten-style PNG image with PIL, to be mixed
       into OCR training/eval data
    3. Written to ``data/ncert_problems/manifest.csv`` alongside its
       LaTeX, topic, class level and worked SymPy answer

Usage:
    python scripts/generate_problems.py --count 5000
    python scripts/generate_problems.py --count 50 --no-images  # fast smoke test
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/<name>.py` from any directory: put the project root on sys.path
# (Python only adds the scripts/ folder itself, so `import src` would fail otherwise).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import sympy
from sympy import Symbol, diff, integrate, latex, simplify, solve
from sympy.abc import x, y

from src.config import settings

logger = logging.getLogger(__name__)

Generator = Callable[[random.Random], "GeneratedProblem"]


@dataclass
class GeneratedProblem:
    """A single synthetic problem plus its verified answer.

    Attributes:
        topic: NCERT topic label, e.g. "Quadratic Equations".
        class_level: e.g. "Class 10".
        latex_problem: The problem statement rendered as LaTeX.
        latex_answer: The worked answer rendered as LaTeX.
    """

    topic: str
    class_level: str
    latex_problem: str
    latex_answer: str


def _gen_linear_equation(rng: random.Random) -> GeneratedProblem:
    a, b, c = rng.randint(2, 9), rng.randint(-20, 20), rng.randint(-20, 20)
    expr = sympy.Eq(a * x + b, c)
    solution = solve(expr, x)
    return GeneratedProblem(
        topic="Linear Equations",
        class_level="Class 8",
        latex_problem=latex(expr),
        latex_answer=f"x = {latex(solution[0])}",
    )


def _gen_quadratic_equation(rng: random.Random) -> GeneratedProblem:
    # Generate from known roots to guarantee "nice" integer solutions
    # most of the time, matching typical NCERT textbook problems.
    r1, r2 = rng.randint(-10, 10), rng.randint(-10, 10)
    a = rng.choice([1, 1, 1, 2])  # bias toward monic
    expr = sympy.expand(a * (x - r1) * (x - r2))
    equation = sympy.Eq(expr, 0)
    solutions = solve(equation, x)
    answer = " or ".join(f"x = {latex(s)}" for s in solutions)
    return GeneratedProblem(
        topic="Quadratic Equations",
        class_level="Class 10",
        latex_problem=latex(equation),
        latex_answer=answer,
    )


def _gen_trigonometry(rng: random.Random) -> GeneratedProblem:
    theta = Symbol("theta")
    identity_pairs = [
        (sympy.sin(theta) ** 2 + sympy.cos(theta) ** 2, 1),
        (1 + sympy.tan(theta) ** 2, sympy.sec(theta) ** 2),
        (1 + sympy.cot(theta) ** 2, sympy.csc(theta) ** 2),
    ]
    lhs, rhs = rng.choice(identity_pairs)
    equation = sympy.Eq(lhs, rhs)
    verified = simplify(lhs - rhs) == 0
    return GeneratedProblem(
        topic="Trigonometric Identities",
        class_level="Class 11",
        latex_problem=f"\\text{{Prove: }} {latex(equation)}",
        latex_answer="Identity verified (LHS - RHS simplifies to 0)" if verified else "Not an identity",
    )


def _gen_differentiation(rng: random.Random) -> GeneratedProblem:
    power = rng.randint(2, 6)
    coeff = rng.randint(1, 9)
    expr = coeff * x ** power + rng.randint(-10, 10) * x
    derivative = diff(expr, x)
    return GeneratedProblem(
        topic="Differentiation",
        class_level="Class 12",
        latex_problem=f"\\frac{{d}}{{dx}}\\left({latex(expr)}\\right)",
        latex_answer=latex(derivative),
    )


def _gen_integration(rng: random.Random) -> GeneratedProblem:
    power = rng.randint(1, 5)
    coeff = rng.randint(1, 9)
    expr = coeff * x ** power
    integral = integrate(expr, x)
    return GeneratedProblem(
        topic="Integration",
        class_level="Class 12",
        latex_problem=f"\\int {latex(expr)} \\, dx",
        latex_answer=f"{latex(integral)} + C",
    )


def _gen_coordinate_geometry(rng: random.Random) -> GeneratedProblem:
    x1, y1 = rng.randint(-10, 10), rng.randint(-10, 10)
    x2, y2 = rng.randint(-10, 10), rng.randint(-10, 10)
    distance_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    distance = sympy.sqrt(distance_sq)
    return GeneratedProblem(
        topic="Coordinate Geometry",
        class_level="Class 10",
        latex_problem=(
            f"\\text{{Find the distance between }} ({x1}, {y1}) \\text{{ and }} ({x2}, {y2})"
        ),
        latex_answer=f"d = {latex(simplify(distance))}",
    )


def _gen_matrix(rng: random.Random) -> GeneratedProblem:
    values = [[rng.randint(-5, 5) for _ in range(2)] for _ in range(2)]
    matrix = sympy.Matrix(values)
    determinant = matrix.det()
    return GeneratedProblem(
        topic="Matrices",
        class_level="Class 12",
        latex_problem=f"\\text{{Find }} \\det{latex(matrix)}",
        latex_answer=str(determinant),
    )


def _gen_probability(rng: random.Random) -> GeneratedProblem:
    total = rng.randint(10, 52)
    favorable = rng.randint(1, total)
    prob = sympy.Rational(favorable, total)
    return GeneratedProblem(
        topic="Probability",
        class_level="Class 10",
        latex_problem=(
            f"\\text{{A bag has {total} cards, {favorable} are favorable. "
            f"Find P(favorable).}}"
        ),
        latex_answer=f"P = {latex(prob)}",
    )


def _gen_vectors(rng: random.Random) -> GeneratedProblem:
    a_vec = sympy.Matrix([rng.randint(-5, 5) for _ in range(3)])
    b_vec = sympy.Matrix([rng.randint(-5, 5) for _ in range(3)])
    dot = a_vec.dot(b_vec)
    return GeneratedProblem(
        topic="Vectors",
        class_level="Class 12",
        latex_problem=(
            f"\\text{{Find }} \\vec{{a}} \\cdot \\vec{{b}} \\text{{ for }} "
            f"\\vec{{a}}={latex(a_vec.T)}, \\vec{{b}}={latex(b_vec.T)}"
        ),
        latex_answer=str(dot),
    )


GENERATORS: list[Generator] = [
    _gen_linear_equation,
    _gen_quadratic_equation,
    _gen_trigonometry,
    _gen_differentiation,
    _gen_integration,
    _gen_coordinate_geometry,
    _gen_matrix,
    _gen_probability,
    _gen_vectors,
]


def render_handwritten_style_image(text: str, output_path: Path, seed: int) -> None:
    """Render ``text`` as a PNG that mimics handwritten pen-on-paper style.

    Uses PIL with slight per-character jitter and rotation to
    approximate handwriting variability, since we don't have real ink
    for every synthetic sample. A monospace fallback font is used if no
    system TTF is available, so this always works in minimal containers.

    Args:
        text: Plain-text (not LaTeX-rendered) version of the problem,
            since PIL cannot render TeX macros directly.
        output_path: Destination PNG path.
        seed: RNG seed for jitter, for reproducibility.
    """
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed)
    width, height = 800, 200
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)

    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 28)
            break
    if font is None:
        font = ImageFont.load_default()

    x_cursor, y_cursor = 20, 70
    for ch in text:
        jitter_y = rng.randint(-3, 3)
        draw.text((x_cursor, y_cursor + jitter_y), ch, font=font, fill=0)
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            char_width = bbox[2] - bbox[0] + 2
        except Exception:  # noqa: BLE001
            char_width = 16
        x_cursor += max(char_width, 8)
        if x_cursor > width - 40:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _latex_to_plain(text: str) -> str:
    """Very rough LaTeX -> readable-text stripping for image rendering."""
    replacements = {
        "\\frac": "", "\\sqrt": "sqrt", "\\left": "", "\\right": "",
        "\\text": "", "\\int": "integral", "\\cdot": "*", "{": "", "}": "",
        "\\pm": "+/-", "\\vec": "",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def generate_dataset(count: int, with_images: bool = True, seed: int = 42) -> Path:
    """Generate ``count`` synthetic problems and write the manifest CSV.

    Args:
        count: Number of problems to generate (spread evenly across topics).
        with_images: Whether to also render handwritten-style PNGs.
        seed: RNG seed for reproducibility.

    Returns:
        Path to the written ``manifest.csv``.
    """
    rng = random.Random(seed)
    output_dir = settings.ncert_problems_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    rows = []
    for i in range(count):
        generator = GENERATORS[i % len(GENERATORS)]
        try:
            problem = generator(rng)
        except Exception:  # noqa: BLE001
            logger.warning("Generator %s failed on sample %d", generator.__name__, i, exc_info=True)
            continue

        image_filename = f"problem_{i:05d}.png"
        if with_images:
            render_handwritten_style_image(
                _latex_to_plain(problem.latex_problem),
                images_dir / image_filename,
                seed=seed + i,
            )

        rows.append(
            {
                "filename": image_filename if with_images else "",
                "topic": problem.topic,
                "class_level": problem.class_level,
                "latex": problem.latex_problem,
                "answer_latex": problem.latex_answer,
            }
        )

        if (i + 1) % 500 == 0:
            logger.info("Generated %d/%d problems", i + 1, count)

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "topic", "class_level", "latex", "answer_latex"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d problems to %s", len(rows), manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NCERT problem dataset.")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--no-images", action="store_true", help="Skip PNG rendering (CSV only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_dataset(args.count, with_images=not args.no_images, seed=args.seed)


if __name__ == "__main__":
    main()
