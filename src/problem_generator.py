"""Practice problem generator (feeds Screen 3's "Practice Similar" button
and MathBot's ``similar_problems`` tool).

Wraps the same SymPy-verified generation approach used offline in
``scripts/generate_problems.py`` (for bulk synthetic dataset creation)
as a lightweight, on-demand, importable API: given either a topic name
or an existing problem's LaTeX, produce one new, verified problem of
matching difficulty.

Every generated problem is solved with SymPy at generation time, so
"Show Answer" is always guaranteed correct — no LLM hallucination risk
for the practice-problem answer key.

Example:
    >>> from src.problem_generator import ProblemGenerator
    >>> generator = ProblemGenerator()
    >>> problem = generator.generate_similar("x^2 - 5x + 6 = 0")
    >>> problem.topic
    'Quadratic Equations'
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Optional

import sympy
from sympy import diff, integrate, latex, solve
from sympy.abc import x

from src.latex_parser import LatexParser, LatexParseError
from src.problem_classifier import ProblemClassifier, ProblemType

logger = logging.getLogger(__name__)


@dataclass
class PracticeProblem:
    """A single generated-and-solved practice problem.

    Attributes:
        topic: Human-readable topic label.
        class_level: NCERT class level, e.g. "Class 10".
        latex_problem: The problem statement.
        latex_answer: The verified worked answer.
        difficulty: 1-3 star rating.
    """

    topic: str
    class_level: str
    latex_problem: str
    latex_answer: str
    difficulty: int = 2


Generator = Callable[[random.Random], PracticeProblem]


def _gen_linear(rng: random.Random) -> PracticeProblem:
    a, b, c = rng.randint(2, 9), rng.randint(-20, 20), rng.randint(-20, 20)
    equation = sympy.Eq(a * x + b, c)
    solution = solve(equation, x)
    return PracticeProblem(
        topic="Linear Equations",
        class_level="Class 8",
        latex_problem=latex(equation),
        latex_answer=f"x = {latex(solution[0])}",
        difficulty=1,
    )


def _gen_quadratic(rng: random.Random) -> PracticeProblem:
    r1, r2 = rng.randint(-10, 10), rng.randint(-10, 10)
    a = rng.choice([1, 1, 1, 2])
    expr = sympy.expand(a * (x - r1) * (x - r2))
    equation = sympy.Eq(expr, 0)
    solutions = solve(equation, x)
    answer = " or ".join(f"x = {latex(s)}" for s in solutions)
    return PracticeProblem(
        topic="Quadratic Equations",
        class_level="Class 10",
        latex_problem=latex(equation),
        latex_answer=answer,
        difficulty=2,
    )


def _gen_polynomial(rng: random.Random) -> PracticeProblem:
    roots = [rng.randint(-6, 6) for _ in range(3)]
    expr = sympy.expand(sympy.prod([(x - r) for r in roots]))
    equation = sympy.Eq(expr, 0)
    solutions = solve(equation, x)
    answer = " or ".join(f"x = {latex(s)}" for s in solutions)
    return PracticeProblem(
        topic="Polynomials",
        class_level="Class 10",
        latex_problem=latex(equation),
        latex_answer=answer,
        difficulty=3,
    )


def _gen_trigonometry(rng: random.Random) -> PracticeProblem:
    theta = sympy.Symbol("theta")
    identity_pairs = [
        (sympy.sin(theta) ** 2 + sympy.cos(theta) ** 2, 1),
        (1 + sympy.tan(theta) ** 2, sympy.sec(theta) ** 2),
    ]
    lhs, rhs = rng.choice(identity_pairs)
    equation = sympy.Eq(lhs, rhs)
    return PracticeProblem(
        topic="Trigonometric Identities",
        class_level="Class 11",
        latex_problem=f"\\text{{Prove: }} {latex(equation)}",
        latex_answer="Identity holds (LHS - RHS simplifies to 0).",
        difficulty=2,
    )


def _gen_differentiation(rng: random.Random) -> PracticeProblem:
    power = rng.randint(2, 6)
    coeff = rng.randint(1, 9)
    expr = coeff * x ** power + rng.randint(-10, 10) * x
    derivative = diff(expr, x)
    return PracticeProblem(
        topic="Differentiation",
        class_level="Class 12",
        latex_problem=f"\\frac{{d}}{{dx}}\\left({latex(expr)}\\right)",
        latex_answer=latex(derivative),
        difficulty=2,
    )


def _gen_integration(rng: random.Random) -> PracticeProblem:
    power = rng.randint(1, 5)
    coeff = rng.randint(1, 9)
    expr = coeff * x ** power
    result = integrate(expr, x)
    return PracticeProblem(
        topic="Integration",
        class_level="Class 12",
        latex_problem=f"\\int {latex(expr)} \\, dx",
        latex_answer=f"{latex(result)} + C",
        difficulty=2,
    )


def _gen_vector(rng: random.Random) -> PracticeProblem:
    """Dot, cross, magnitude or angle problem on small integer vectors (verified by the solver)."""
    from src.vector_ops import Matrix, VectorProblem, solve_vector

    n = 3
    a = Matrix([rng.randint(-4, 5) for _ in range(n)])
    b = Matrix([rng.randint(-4, 5) for _ in range(n)])
    kind = rng.choice(["dot", "cross", "magnitude", "angle"])
    while a.norm() == 0 or b.norm() == 0:
        a, b = Matrix([rng.randint(-4, 5) for _ in range(n)]), Matrix([rng.randint(-4, 5) for _ in range(n)])
    fmt = lambda v: "(" + ", ".join(str(c) for c in v) + ")"
    if kind == "magnitude":
        statement, operands = f"|{fmt(a)}|", [a]
    elif kind == "angle":
        statement = (f"\\text{{Find the angle between }} \\vec{{a}} \\text{{ and }} \\vec{{b}} "
                     f"\\text{{ for }} \\vec{{a}}={fmt(a)}, \\vec{{b}}={fmt(b)}")
        operands = [a, b]
    else:
        operator = "\\cdot" if kind == "dot" else "\\times"
        statement, operands = f"{fmt(a)} {operator} {fmt(b)}", [a, b]
    solution = solve_vector(VectorProblem(kind, operands, ["\\vec{a}", "\\vec{b}"][: len(operands)]))
    return PracticeProblem(
        topic="Vectors", class_level="Class 12", latex_problem=statement,
        latex_answer=solution.answer, difficulty=2,
    )


_TOPIC_GENERATORS: dict[ProblemType, Generator] = {
    ProblemType.VECTOR: _gen_vector,
    ProblemType.LINEAR_EQUATION: _gen_linear,
    ProblemType.QUADRATIC_EQUATION: _gen_quadratic,
    ProblemType.POLYNOMIAL: _gen_polynomial,
    ProblemType.TRIGONOMETRY: _gen_trigonometry,
    ProblemType.DIFFERENTIATION: _gen_differentiation,
    ProblemType.INTEGRATION: _gen_integration,
}

_ALL_GENERATORS: list[Generator] = list(_TOPIC_GENERATORS.values())

_TOPIC_NAME_HINTS: dict[str, ProblemType] = {
    "linear": ProblemType.LINEAR_EQUATION,
    "quadratic": ProblemType.QUADRATIC_EQUATION,
    "polynomial": ProblemType.POLYNOMIAL,
    "trig": ProblemType.TRIGONOMETRY,
    "trigonometry": ProblemType.TRIGONOMETRY,
    "differentiat": ProblemType.DIFFERENTIATION,
    "derivative": ProblemType.DIFFERENTIATION,
    "integrat": ProblemType.INTEGRATION,
    "vector": ProblemType.VECTOR,
    "dot product": ProblemType.VECTOR,
    "cross product": ProblemType.VECTOR,
}


class ProblemGenerator:
    """Generates a single verified practice problem, matched to a topic or existing problem."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Create a generator.

        Args:
            seed: Optional RNG seed for reproducibility (mainly for tests).
        """
        self._rng = random.Random(seed)
        self._latex_parser = LatexParser()
        self._classifier = ProblemClassifier()

    def generate_for_type(self, problem_type: ProblemType) -> PracticeProblem:
        """Generate a fresh problem of a specific :class:`ProblemType`.

        Args:
            problem_type: The desired problem type.

        Returns:
            A verified :class:`PracticeProblem`.

        Raises:
            ValueError: If no generator exists for that type yet.
        """
        generator = _TOPIC_GENERATORS.get(problem_type)
        if generator is None:
            raise ValueError(f"No practice-problem generator available for {problem_type.value} yet.")
        return generator(self._rng)

    def generate_similar(self, reference: Optional[str]) -> PracticeProblem:
        """Generate a problem similar in topic/difficulty to a reference.

        Args:
            reference: Either a topic keyword (e.g. "quadratic") or the
                LaTeX of an existing problem the student just solved.
                If ``None`` or unrecognized, a random topic is chosen.

        Returns:
            A verified :class:`PracticeProblem`.
        """
        problem_type = self._infer_type(reference) if reference else None
        if problem_type is not None:
            generator = _TOPIC_GENERATORS.get(problem_type)
            if generator is not None:
                return generator(self._rng)

        logger.info("Falling back to a random practice-problem topic for reference=%r", reference)
        return self._rng.choice(_ALL_GENERATORS)(self._rng)

    def _infer_type(self, reference: str) -> Optional[ProblemType]:
        """Infer a :class:`ProblemType` from a topic keyword or LaTeX string."""
        lowered = reference.lower()
        for hint, ptype in _TOPIC_NAME_HINTS.items():
            if hint in lowered:
                return ptype

        try:
            parsed = self._latex_parser.parse(reference)
            return self._classifier.classify(parsed).problem_type
        except LatexParseError:
            return None


if __name__ == "__main__":
    generator = ProblemGenerator(seed=1)
    for topic in ["quadratic", "differentiation", "x^2 - 5x + 6 = 0", "unknown topic xyz"]:
        problem = generator.generate_similar(topic)
        print(f"{topic!r} -> [{problem.topic}] {problem.latex_problem} => {problem.latex_answer}")
