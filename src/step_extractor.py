"""Human-readable step extraction (Module 3E).

Converts the ``intermediate`` data SymPy left behind in a
:class:`src.math_solver.SolverResult` into the numbered, "why this
step" annotated sequence the frontend renders in Screen 3 (Solution
View), e.g.:

    Step 1: Identify equation type — Quadratic
    Step 2: Write in standard form: x^2 - 5x + 6 = 0
    Step 3: Find factors: (x-2)(x-3) = 0
    Step 4: Apply zero product property
    Step 5: x - 2 = 0 -> x = 2
    Step 6: x - 3 = 0 -> x = 3
    Step 7: Verify: (2)^2 - 5(2) + 6 = 0 [check]
    Answer: x = 2 or x = 3

Each :class:`Step` carries a short "why" annotation, matching the
"💡 Why this step?" UI element in the product spec. This module has no
external dependencies beyond SymPy, so it is fully unit-testable
offline.

Example:
    >>> from src.latex_parser import LatexParser
    >>> from src.problem_classifier import ProblemClassifier
    >>> from src.math_solver import MathSolver
    >>> from src.step_extractor import StepExtractor
    >>> parsed = LatexParser().parse("x^2 - 5x + 6 = 0")
    >>> ptype = ProblemClassifier().classify(parsed).problem_type
    >>> result = MathSolver().solve(parsed, ptype)
    >>> steps = StepExtractor().extract(ptype, result)
    >>> steps[0].title
    'Identify Equation Type'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import sympy
from sympy import latex

from src.math_solver import SolverResult
from src.problem_classifier import ProblemType

logger = logging.getLogger(__name__)


@dataclass
class Step:
    """A single, student-facing solution step.

    Attributes:
        number: 1-indexed step number.
        title: Short heading, e.g. "Factorization Method".
        content: The worked math for this step (LaTeX-friendly text).
        why: One-sentence rationale ("Why this step?" annotation).
        verified: Whether this step includes a self-check (e.g. plugging
            the root back into the original equation).
    """

    number: int
    title: str
    content: str
    why: str
    verified: bool = False


@dataclass
class StepSequence:
    """The full step-by-step solution plus the final answer.

    Attributes:
        steps: Ordered list of :class:`Step`.
        final_answer: The headline answer string shown in the "Answer" box.
        difficulty: A 1-3 star rating (see :meth:`StepExtractor.estimate_difficulty`).
    """

    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    difficulty: int = 1


class StepExtractor:
    """Builds a :class:`StepSequence` from a solver result, per problem type."""

    def extract(self, problem_type: ProblemType, result: SolverResult) -> StepSequence:
        """Dispatch to a type-specific step builder.

        Args:
            problem_type: The classified problem type.
            result: The :class:`~src.math_solver.SolverResult`.

        Returns:
            A :class:`StepSequence`. Falls back to a minimal 1-step
            sequence (just the final answer) for engines/types without
            a dedicated step builder (e.g. WolframAlpha/Gemini results,
            which don't expose SymPy intermediate steps).
        """
        if result.engine_used != "sympy":
            return self._fallback_sequence(result)

        if "vector" in result.intermediate:
            data = result.intermediate["vector"]
            steps = [Step(i + 1, title, content, why) for i, (title, content, why) in enumerate(data["steps"])]
            return StepSequence(steps=steps, final_answer=result.answer, difficulty=2)
        if result.intermediate.get("two_variable"):
            builder_two = self._steps_for_two_variable
            try:
                return builder_two(result)
            except Exception:  # noqa: BLE001
                logger.warning("Two-variable step extraction failed; using fallback", exc_info=True)
                return self._fallback_sequence(result)
        if "simplified" in result.intermediate:
            try:
                return self._steps_for_expression(result)
            except Exception:  # noqa: BLE001
                logger.warning("Expression step extraction failed; using fallback", exc_info=True)
                return self._fallback_sequence(result)

        builders = {
            ProblemType.LINEAR_EQUATION: self._steps_for_linear,
            ProblemType.QUADRATIC_EQUATION: self._steps_for_quadratic,
            ProblemType.POLYNOMIAL: self._steps_for_polynomial,
            ProblemType.SIMULTANEOUS_EQUATIONS: self._steps_for_system,
            ProblemType.DIFFERENTIATION: self._steps_for_derivative,
            ProblemType.INTEGRATION: self._steps_for_integral,
            ProblemType.MATRIX: self._steps_for_matrix,
        }
        builder = builders.get(problem_type)
        if builder is None:
            return self._fallback_sequence(result)

        try:
            return builder(result)
        except Exception:  # noqa: BLE001
            logger.warning("Step extraction failed for %s; using fallback", problem_type, exc_info=True)
            return self._fallback_sequence(result)

    # ------------------------------------------------------------------
    def _fallback_sequence(self, result: SolverResult) -> StepSequence:
        step = Step(
            number=1,
            title="Solution",
            content=result.answer,
            why=f"Computed via {result.engine_used}.",
        )
        return StepSequence(steps=[step], final_answer=result.answer, difficulty=2)

    # ------------------------------------------------------------------
    def _steps_for_linear(self, result: SolverResult) -> StepSequence:
        equation = result.intermediate["equation"]
        symbol = result.intermediate["symbol"]
        solutions = result.intermediate["solutions"]

        steps = [
            Step(
                number=1,
                title="Identify Equation Type",
                content=f"{latex(equation)} is a Linear Equation in {symbol}.",
                why="Recognizing the equation type tells us which method to apply.",
            ),
            Step(
                number=2,
                title="Isolate the Variable",
                content=f"Move all {symbol}-terms to one side and constants to the other.",
                why="Isolating the variable lets us solve for it directly.",
            ),
            Step(
                number=3,
                title="Solve",
                content=f"{symbol} = {latex(solutions[0])}",
                why="Dividing both sides by the coefficient gives the value of the variable.",
            ),
        ]
        verify_step = self._verification_step(len(steps) + 1, equation, symbol, solutions[0])
        steps.append(verify_step)

        return StepSequence(
            steps=steps,
            final_answer=f"{symbol} = {latex(solutions[0])}",
            difficulty=1,
        )

    def _steps_for_quadratic(self, result: SolverResult) -> StepSequence:
        equation = result.intermediate["equation"]
        symbol = result.intermediate["symbol"]
        solutions = result.intermediate["solutions"]
        factored = result.intermediate.get("factored")
        discriminant = result.intermediate.get("discriminant")

        steps = [
            Step(
                number=1,
                title="Identify Equation Type",
                content=(
                    f"{latex(equation)} is a Quadratic Equation. "
                    f"Standard form: a{symbol}^2 + b{symbol} + c = 0"
                ),
                why="Standard form helps us apply the quadratic formula or factorization.",
            ),
        ]

        step_num = 2
        if discriminant is not None:
            steps.append(
                Step(
                    number=step_num,
                    title="Compute the Discriminant",
                    content=f"D = b^2 - 4ac = {latex(discriminant)}",
                    why=(
                        "The discriminant tells us how many real solutions exist: "
                        "D>0 gives 2 real roots, D=0 gives 1 repeated root, D<0 gives no real roots."
                    ),
                )
            )
            step_num += 1

        if factored is not None and factored != sympy.expand(equation.lhs - equation.rhs):
            steps.append(
                Step(
                    number=step_num,
                    title="Factorize",
                    content=f"{latex(factored)} = 0",
                    why="Factoring lets us use the zero-product property to split into simpler equations.",
                )
            )
            step_num += 1
            steps.append(
                Step(
                    number=step_num,
                    title="Apply Zero Product Property",
                    content="If A * B = 0, then A = 0 or B = 0.",
                    why="A product is zero only if at least one factor is zero.",
                )
            )
            step_num += 1

        for sol in solutions:
            steps.append(
                Step(
                    number=step_num,
                    title="Solve for the Root",
                    content=f"{symbol} = {latex(sol)}",
                    why="Solving each factor/branch gives one root of the equation.",
                )
            )
            step_num += 1

        verify_content = " ".join(
            self._verification_text(equation, symbol, sol) for sol in solutions
        )
        steps.append(
            Step(
                number=step_num,
                title="Verify",
                content=verify_content,
                why="Substituting each root back confirms the solution satisfies the original equation.",
                verified=True,
            )
        )

        answer = " or ".join(f"{symbol} = {latex(sol)}" for sol in solutions)
        return StepSequence(steps=steps, final_answer=answer, difficulty=2)

    def _steps_for_polynomial(self, result: SolverResult) -> StepSequence:
        equation = result.intermediate["equation"]
        symbol = result.intermediate["symbol"]
        solutions = result.intermediate["solutions"]
        factored = result.intermediate.get("factored")

        steps = [
            Step(
                number=1,
                title="Identify Equation Type",
                content=f"{latex(equation)} is a higher-degree Polynomial Equation.",
                why="Higher-degree polynomials are usually solved by factoring or the rational root theorem.",
            )
        ]
        if factored is not None:
            steps.append(
                Step(
                    number=2,
                    title="Factorize",
                    content=f"{latex(factored)} = 0",
                    why="Factoring breaks the polynomial into simpler linear/quadratic pieces.",
                )
            )
        steps.append(
            Step(
                number=len(steps) + 1,
                title="Solve Each Factor",
                content=" or ".join(f"{symbol} = {latex(sol)}" for sol in solutions),
                why="Each factor set to zero gives one root of the polynomial.",
            )
        )
        answer = " or ".join(f"{symbol} = {latex(sol)}" for sol in solutions)
        return StepSequence(steps=steps, final_answer=answer, difficulty=3)

    def _steps_for_system(self, result: SolverResult) -> StepSequence:
        equations = result.intermediate["equations"]
        solution = result.intermediate["solution"]

        steps = [
            Step(
                number=1,
                title="Write Down the System",
                content=" ; ".join(latex(eq) for eq in equations),
                why="Writing both equations clearly lets us choose elimination or substitution.",
            ),
            Step(
                number=2,
                title="Eliminate/Substitute",
                content="Combine the equations to eliminate one variable.",
                why="Reducing to a single-variable equation makes it directly solvable.",
            ),
        ]
        if isinstance(solution, dict):
            answer = ", ".join(f"{sym} = {latex(val)}" for sym, val in solution.items())
        else:
            answer = str(solution)

        steps.append(
            Step(
                number=3,
                title="Solve for Each Variable",
                content=answer,
                why="Back-substitute the found value into either original equation to get the rest.",
            )
        )
        return StepSequence(steps=steps, final_answer=answer, difficulty=2)

    def _steps_for_derivative(self, result: SolverResult) -> StepSequence:
        original = result.intermediate["original"]
        symbol = result.intermediate["symbol"]
        derivative = result.intermediate["derivative"]

        steps = [
            Step(
                number=1,
                title="Identify the Function",
                content=f"f({symbol}) = {latex(original)}",
                why="We differentiate term by term.",
            ),
            Step(
                number=2,
                title="Apply Differentiation Rules",
                content=f"Use the power rule: d/d{symbol}({symbol}^n) = n*{symbol}^(n-1)",
                why="The power rule handles polynomial terms directly.",
            ),
            Step(
                number=3,
                title="Combine Terms",
                content=f"f'({symbol}) = {latex(derivative)}",
                why="Summing the derivative of each term gives the full derivative.",
            ),
        ]
        return StepSequence(steps=steps, final_answer=latex(derivative), difficulty=2)

    def _steps_for_integral(self, result: SolverResult) -> StepSequence:
        integrand = result.intermediate["integrand"]
        integral_result = result.intermediate["result"]
        is_definite = result.intermediate.get("is_definite", False)

        steps = [
            Step(
                number=1,
                title="Identify the Integrand",
                content=latex(integrand),
                why="We integrate term by term using the reverse power rule.",
            ),
            Step(
                number=2,
                title="Apply Integration Rules",
                content="Use the power rule: ∫x^n dx = x^(n+1)/(n+1) + C",
                why="The reverse power rule handles polynomial terms directly.",
            ),
            Step(
                number=3,
                title="Write the Result",
                content=latex(integral_result) if is_definite else f"{latex(integral_result)} + C",
                why=(
                    "Definite integrals evaluate to a number."
                    if is_definite
                    else "Indefinite integrals always include a constant of integration, C."
                ),
            ),
        ]
        answer = latex(integral_result) if is_definite else f"{latex(integral_result)} + C"
        return StepSequence(steps=steps, final_answer=answer, difficulty=2)

    def _steps_for_matrix(self, result: SolverResult) -> StepSequence:
        matrix = result.intermediate["matrix"]
        det = result.intermediate.get("determinant")

        steps = [
            Step(
                number=1,
                title="Write the Matrix",
                content=latex(matrix),
                why="Clearly writing the matrix helps track rows/columns for the operation.",
            )
        ]
        if det is not None:
            steps.append(
                Step(
                    number=2,
                    title="Compute the Determinant",
                    content=f"\\det(A) = {latex(det)}",
                    why="For a 2x2 matrix [[a,b],[c,d]], det = ad - bc.",
                )
            )
        if "inverse" in result.intermediate:
            steps.append(
                Step(
                    number=len(steps) + 1,
                    title="Compute the Inverse",
                    content=f"A^{{-1}} = {latex(result.intermediate['inverse'])}",
                    why="The inverse exists only when the determinant is non-zero.",
                )
            )
        return StepSequence(steps=steps, final_answer=result.answer, difficulty=3)

    # ------------------------------------------------------------------
    def _steps_for_expression(self, result: SolverResult) -> StepSequence:
        """Steps for evaluating a numeric expression or simplifying an algebraic one."""
        inter = result.intermediate
        expr, best, method = inter["expression"], inter["simplified"], inter["method"]
        constant = inter["constant"]
        method_names = {
            "simplify": "Simplify using algebraic rules",
            "trigsimp": "Apply trigonometric identities (e.g. sin²x + cos²x = 1)",
            "factor": "Factorise the expression",
            "expand": "Expand the brackets",
        }
        steps = [Step(1, "Read the Expression", f"{latex(expr)}",
                      "First we write down exactly what has to be " + ("evaluated." if constant else "simplified."))]
        if method == "original":
            steps.append(Step(2, "Check for Simplification", "No shorter equivalent form exists.",
                              "Trying simplify, factor and expand shows the expression is already in simplest form."))
        else:
            steps.append(Step(2, method_names.get(method, "Simplify"), f"{latex(expr)} = {latex(best)}",
                              "Rewriting with a standard rule gives an equal but simpler expression."))
        if constant and inter.get("numeric") is not None and not (best.is_Integer or best.is_Rational):
            steps.append(Step(3, "Decimal Value", f"\\approx {inter['numeric']:.6g}",
                              "A decimal approximation helps to judge the size of the answer."))
        return StepSequence(steps=steps, final_answer=result.answer, difficulty=1 if constant else 2)

    def _steps_for_two_variable(self, result: SolverResult) -> StepSequence:
        """Steps for a two-variable relation such as ``y = 2x + 3`` (line / parabola)."""
        inter = result.intermediate
        eq, dep, indep = inter["equation"], inter["symbol"], inter["independent"]
        main = inter["solutions"][0]
        steps = [
            Step(1, "Identify the Relation", f"{latex(eq)} links {indep} and {dep}.",
                 "An equation with two variables describes a curve; each x gives a matching y."),
            Step(2, f"Write {dep} in terms of {indep}", f"{dep} = {latex(main)}",
                 f"Isolating {dep} turns the relation into a function we can evaluate and plot."),
        ]
        if "slope" in inter:
            steps.append(Step(3, "Slope and Intercept", f"slope m = {latex(inter['slope'])},  {dep}-intercept c = {latex(inter['intercept'])}",
                              "In y = mx + c, m is the steepness of the line and c is where it cuts the y-axis."))
        elif "vertex" in inter:
            vx, vy = inter["vertex"]
            steps.append(Step(3, "Vertex of the Parabola", f"({latex(vx)}, {latex(vy)})",
                              "For y = ax² + bx + c the turning point is at x = -b/(2a)."))
        xi = inter.get("x_intercepts")
        if xi:
            steps.append(Step(len(steps) + 1, f"Where it Cuts the {indep}-axis",
                              ", ".join(f"{indep} = {latex(r)}" for r in xi),
                              f"Setting {dep} = 0 and solving gives the points where the graph touches the {indep}-axis."))
        steps.append(Step(len(steps) + 1, f"Where it Cuts the {dep}-axis", f"{indep} = 0 gives {dep} = {latex(inter['y_intercept'])}",
                          f"Putting {indep} = 0 gives the point where the graph crosses the {dep}-axis."))
        return StepSequence(steps=steps, final_answer=result.answer, difficulty=2)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _verification_text(equation: sympy.Eq, symbol: sympy.Symbol, solution: Any) -> str:
        lhs_value = equation.lhs.subs(symbol, solution)
        rhs_value = equation.rhs.subs(symbol, solution)
        check = "check" if sympy.simplify(lhs_value - rhs_value) == 0 else "MISMATCH"
        return f"Verify {symbol}={latex(solution)}: {latex(lhs_value)} = {latex(rhs_value)} [{check}]"

    def _verification_step(
        self, number: int, equation: sympy.Eq, symbol: sympy.Symbol, solution: Any
    ) -> Step:
        return Step(
            number=number,
            title="Verify",
            content=self._verification_text(equation, symbol, solution),
            why="Substituting the solution back into the original equation confirms it's correct.",
            verified=True,
        )

    # ------------------------------------------------------------------
    def estimate_difficulty(self, problem_type: ProblemType, num_steps: int) -> int:
        """Rough 1-3 star difficulty heuristic used by the frontend badge.

        Args:
            problem_type: The classified problem type.
            num_steps: Number of steps in the generated sequence.

        Returns:
            An integer in {1, 2, 3}.
        """
        base = {
            ProblemType.LINEAR_EQUATION: 1,
            ProblemType.QUADRATIC_EQUATION: 2,
            ProblemType.SIMULTANEOUS_EQUATIONS: 2,
            ProblemType.POLYNOMIAL: 3,
            ProblemType.TRIGONOMETRY: 2,
            ProblemType.DIFFERENTIATION: 2,
            ProblemType.INTEGRATION: 2,
            ProblemType.MATRIX: 3,
            ProblemType.ARITHMETIC: 1,
        }.get(problem_type, 2)
        if num_steps > 6:
            base = min(3, base + 1)
        return base


if __name__ == "__main__":
    from src.latex_parser import LatexParser
    from src.math_solver import MathSolver
    from src.problem_classifier import ProblemClassifier

    parser = LatexParser()
    classifier = ProblemClassifier()
    solver = MathSolver()
    extractor = StepExtractor()

    parsed_problem = parser.parse(r"x^2 - 5x + 6 = 0")
    ptype = classifier.classify(parsed_problem).problem_type
    solver_result = solver.solve(parsed_problem, ptype)
    sequence = extractor.extract(ptype, solver_result)

    for step in sequence.steps:
        print(f"Step {step.number}: {step.title}\n  {step.content}\n  Why: {step.why}\n")
    print(f"Answer: {sequence.final_answer}")
