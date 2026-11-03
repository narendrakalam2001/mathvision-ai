"""Multi-engine mathematical problem solver (Module 3B, 3C, 3D).

Routes a classified problem to the best available solving engine:

    3B. SymPy      — primary, exact, free, offline. Handles the vast
                     majority of NCERT/JEE algebra, calculus, matrix
                     and simple statistics problems.
    3C. WolframAlpha — fallback for anything SymPy cannot solve
                       (free tier: 2000 queries/month, tracked via
                       a simple on-disk counter so we never silently
                       exceed the quota).
    3D. Gemini      — used for (a) word problems that need language
                       understanding to set up the equation, and
                       (b) cross-verification of the SymPy/Wolfram
                       answer ("does the LLM agree with the exact
                       symbolic answer?").

All three engines return the same :class:`SolverResult` shape so
:mod:`src.step_extractor` and the API layer don't need to know which
engine actually produced the answer.

Example:
    >>> from src.latex_parser import LatexParser
    >>> from src.problem_classifier import ProblemClassifier
    >>> from src.math_solver import MathSolver
    >>> parsed = LatexParser().parse("x^2 - 5x + 6 = 0")
    >>> ptype = ProblemClassifier().classify(parsed).problem_type
    >>> MathSolver().solve(parsed, ptype).answer
    'x = 2 or x = 3'
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import sympy
from sympy import Eq, Matrix, latex, solve

from src.config import settings
from src.gemini_client import GeminiError, generate_text
from src.latex_parser import ParsedProblem
from src.problem_classifier import ProblemType
from src.vector_ops import VectorError, solve_vector

logger = logging.getLogger(__name__)


_EXPRESSION_TYPES = {
    ProblemType.ARITHMETIC, ProblemType.EXPRESSION, ProblemType.TRIGONOMETRY,
    ProblemType.LINEAR_EQUATION, ProblemType.QUADRATIC_EQUATION, ProblemType.POLYNOMIAL,
    ProblemType.UNKNOWN,
}


_GREEK_NAMES = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
}


class SolverError(Exception):
    """Raised when no engine (SymPy, WolframAlpha, Gemini) can solve the problem."""


class SolverInputError(SolverError):
    """The input itself is invalid (e.g. vectors of different sizes).

    Unlike a plain :class:`SolverError` ("SymPy could not do it"), retrying with
    WolframAlpha or an LLM would only produce a made-up answer, so these skip the fallbacks.
    """


class QuotaExceededError(Exception):
    """Raised when the free WolframAlpha monthly quota has been used up."""


@dataclass
class SolverResult:
    """Uniform result shape returned by every engine.

    Attributes:
        answer: Human-readable final answer (e.g. ``"x = 2 or x = 3"``).
        exact_value: The raw SymPy object(s), when available (``None``
            for Wolfram/Gemini-only results), for downstream graphing
            or further symbolic manipulation.
        numerical_value: Optional float approximation(s) for display.
        engine_used: ``"sympy"``, ``"wolframalpha"``, or ``"gemini"``.
        intermediate: Engine-specific intermediate data consumed by
            :mod:`src.step_extractor` to build the step list (e.g. for
            SymPy quadratics: factored form, discriminant, roots).
        verified: Whether a second engine cross-checked this answer.
    """

    answer: str
    exact_value: Any
    numerical_value: Optional[list[float]]
    engine_used: str
    intermediate: dict[str, Any] = field(default_factory=dict)
    verified: bool = False


class WolframQuotaTracker:
    """Tracks WolframAlpha API usage against the free-tier monthly quota.

    Persists a small JSON counter to disk (keyed by year-month) so the
    quota is respected across process restarts, avoiding surprise
    billing or hard API failures mid-demo.
    """

    def __init__(self, quota: Optional[int] = None, storage_path: Optional[Path] = None) -> None:
        self.quota = quota or settings.wolframalpha_monthly_quota
        self.storage_path = storage_path or (settings.data_dir / "wolfram_quota.json")

    def _load(self) -> dict[str, int]:
        if not self.storage_path.exists():
            return {}
        try:
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read Wolfram quota file, resetting", exc_info=True)
            return {}

    def _save(self, data: dict[str, int]) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            logger.warning("Could not persist Wolfram quota file", exc_info=True)

    def _month_key(self) -> str:
        today = date.today()
        return f"{today.year}-{today.month:02d}"

    def remaining(self) -> int:
        """Queries left this calendar month."""
        used = self._load().get(self._month_key(), 0)
        return max(0, self.quota - used)

    def record_call(self) -> None:
        """Increment the usage counter for the current month.

        Raises:
            QuotaExceededError: If the quota was already exhausted
                *before* this call (checked, then incremented).
        """
        if self.remaining() <= 0:
            raise QuotaExceededError(
                f"WolframAlpha free-tier quota ({self.quota}/month) exhausted."
            )
        data = self._load()
        key = self._month_key()
        data[key] = data.get(key, 0) + 1
        self._save(data)


class MathSolver:
    """Routes classified problems to the appropriate solving engine."""

    def __init__(
        self,
        wolfram_app_id: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        quota_tracker: Optional[WolframQuotaTracker] = None,
    ) -> None:
        """Create a solver.

        Args:
            wolfram_app_id: WolframAlpha App ID; defaults to ``settings``.
            gemini_api_key: Gemini API key; defaults to ``settings``.
            quota_tracker: Injectable quota tracker (mainly for tests).
        """
        self.wolfram_app_id = wolfram_app_id or settings.wolframalpha_app_id
        self.gemini_api_key = gemini_api_key or settings.gemini_api_key
        self.quota_tracker = quota_tracker or WolframQuotaTracker()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def solve(
        self,
        parsed: ParsedProblem,
        problem_type: ProblemType,
        problem_text: Optional[str] = None,
    ) -> SolverResult:
        """Solve a parsed, classified problem.

        Routing logic:
            1. Try SymPy (3B) — works offline for ~90% of problem types.
            2. If SymPy can't solve it (raises or returns empty), and a
               WolframAlpha App ID is configured with quota remaining,
               fall back to WolframAlpha (3C).
            3. If it's a word problem (needs language understanding) or
               both of the above fail, fall back to Gemini (3D).

        Args:
            parsed: Output of :class:`src.latex_parser.LatexParser`.
            problem_type: Output of :class:`src.problem_classifier.ProblemClassifier`.
            problem_text: Original word-problem narrative, if any.

        Returns:
            A :class:`SolverResult`.

        Raises:
            SolverError: If every available engine fails.
        """
        if not problem_text:
            self._reject_unreadable(parsed)

        if problem_type == ProblemType.WORD_PROBLEM and problem_text:
            try:
                return self._solve_with_gemini(problem_text)
            except Exception:  # noqa: BLE001
                logger.warning("Gemini word-problem solve failed, trying SymPy anyway", exc_info=True)

        try:
            result = self._solve_with_sympy(parsed, problem_type)
            if result is not None:
                return result
        except SolverInputError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("SymPy solve failed for %s, trying fallbacks", problem_type, exc_info=True)

        if settings.enable_wolfram_fallback and self.wolfram_app_id:
            try:
                return self._solve_with_wolfram(parsed.raw_latex)
            except QuotaExceededError:
                logger.warning("WolframAlpha quota exhausted, trying Gemini")
            except Exception:  # noqa: BLE001
                logger.warning("WolframAlpha solve failed, trying Gemini", exc_info=True)

        if settings.enable_gemini_explanations and self.gemini_api_key:
            try:
                return self._solve_with_gemini(problem_text or parsed.raw_latex)
            except Exception:  # noqa: BLE001
                logger.warning("Gemini fallback solve failed", exc_info=True)

        raise SolverError(
            f"Could not solve problem (type={problem_type.value}). "
            "SymPy failed and no working fallback (WolframAlpha/Gemini) is configured."
        )

    @staticmethod
    def _reject_unreadable(parsed: ParsedProblem) -> None:
        """Refuse inputs that are clearly not a math problem (misread text, vectors, ...).

        Multi-letter "variables" such as ``vec`` or ``Find``, undefined
        functions and a swarm of free symbols are the typical signature of
        text the LaTeX parser split into letters. Returning a "simplified"
        version of that to a student would be worse than an honest error.

        Raises:
            SolverError: With a message asking the student to check the LaTeX.
        """
        targets: list[Any] = list(parsed.equations)
        if parsed.expression is not None:
            targets.append(parsed.expression)
        bad_names = sorted(
            str(sym) for sym in parsed.symbols
            if len(str(sym)) > 1 and str(sym).rstrip("0123456789_") not in _GREEK_NAMES
            and not str(sym).replace("_", "").isdigit() and "_" not in str(sym)
        )
        has_undefined = any(
            getattr(t, "atoms", None) and t.atoms(sympy.core.function.AppliedUndef) for t in targets
        )
        if bad_names or has_undefined or len(parsed.symbols) > 5:
            raise SolverError(
                "The recognised text does not look like a supported math problem "
                f"(unexpected symbols: {', '.join(bad_names) or 'unknown functions'}). "
                "Please check or edit the LaTeX."
            )

    # ------------------------------------------------------------------
    # 3B — SymPy (primary)
    # ------------------------------------------------------------------
    def _solve_with_sympy(
        self, parsed: ParsedProblem, problem_type: ProblemType
    ) -> Optional[SolverResult]:
        """Route to a type-specific SymPy routine. Returns None if unsupported."""
        if parsed.vector is not None:
            return self._solve_vector(parsed)
        if not parsed.equations and problem_type in _EXPRESSION_TYPES:
            return self._solve_expression(parsed)
        handlers = {
            ProblemType.LINEAR_EQUATION: self._solve_algebraic,
            ProblemType.QUADRATIC_EQUATION: self._solve_algebraic,
            ProblemType.POLYNOMIAL: self._solve_algebraic,
            ProblemType.SIMULTANEOUS_EQUATIONS: self._solve_system,
            ProblemType.TRIGONOMETRY: self._solve_algebraic,
            ProblemType.DIFFERENTIATION: self._solve_derivative,
            ProblemType.INTEGRATION: self._solve_integral,
            ProblemType.LIMIT: self._solve_limit,
            ProblemType.MATRIX: self._solve_matrix,
        }
        handler = handlers.get(problem_type)
        if handler is None:
            return None
        return handler(parsed)

    def _solve_algebraic(self, parsed: ParsedProblem) -> SolverResult:
        """Solve a single algebraic/trig equation via ``sympy.solve``."""
        if not parsed.equations:
            raise SolverError("No equation to solve.")
        equation = parsed.equations[0]
        free_syms = sorted(equation.free_symbols, key=str)
        if not free_syms:
            raise SolverError("Equation has no free variable to solve for.")
        if len(free_syms) == 2:
            return self._solve_two_variable(equation, free_syms)
        symbol = next((s for s in free_syms if str(s) in ("x", "y", "z", "t")), free_syms[0])

        solutions = solve(equation, symbol)
        if not solutions:
            raise SolverError(f"SymPy found no solutions for {equation}.")

        answer = " or ".join(f"{symbol} = {latex(sol)}" for sol in solutions)
        numerical = self._safe_floats(solutions)

        lhs_minus_rhs = sympy.expand(equation.lhs - equation.rhs)
        intermediate: dict[str, Any] = {
            "equation": equation,
            "symbol": symbol,
            "solutions": solutions,
            "expanded_lhs_minus_rhs": lhs_minus_rhs,
        }
        try:
            intermediate["factored"] = sympy.factor(lhs_minus_rhs)
        except Exception:  # noqa: BLE001
            pass
        try:
            poly = sympy.Poly(lhs_minus_rhs, symbol)
            if poly.degree() == 2:
                a, b, c = poly.all_coeffs()
                intermediate.update({"a": a, "b": b, "c": c, "discriminant": b**2 - 4 * a * c})
        except Exception:  # noqa: BLE001
            pass

        return SolverResult(
            answer=answer,
            exact_value=solutions,
            numerical_value=numerical,
            engine_used="sympy",
            intermediate=intermediate,
        )

    def _solve_vector(self, parsed: ParsedProblem) -> SolverResult:
        """Dot/cross/magnitude/sum/angle of vectors (see :mod:`src.vector_ops`)."""
        try:
            solution = solve_vector(parsed.vector)
        except VectorError as exc:
            raise SolverInputError(str(exc)) from exc
        return SolverResult(
            answer=solution.answer, exact_value=solution.value, numerical_value=solution.numeric,
            engine_used="sympy",
            intermediate={"vector": {"operation": parsed.vector.operation, "steps": solution.steps,
                                     "details": solution.details}},
        )

    def _solve_two_variable(self, equation: Eq, free_syms: list) -> SolverResult:
        """Solve a relation like ``y = 2x + 3`` for the dependent variable.

        ``y`` is treated as dependent when present, otherwise the
        alphabetically last symbol. Collects graph data (intercepts, slope,
        vertex) that the step builder and plotter use.
        """
        names = {str(s): s for s in free_syms}
        dep = names.get("y", free_syms[-1])
        indep = next(s for s in free_syms if s != dep)
        solutions = solve(equation, dep)
        if not solutions:
            raise SolverError(f"Could not express {dep} in terms of {indep}.")

        main = solutions[0]
        intermediate: dict[str, Any] = {
            "equation": equation, "symbol": dep, "independent": indep, "solutions": solutions,
            "two_variable": True, "plot_expr": main, "plot_symbol": indep,
            "y_intercept": sympy.simplify(main.subs(indep, 0)),
        }
        try:
            roots = solve(main, indep)
            intermediate["x_intercepts"] = roots
            intermediate["plot_roots"] = self._safe_floats(roots)
        except Exception:  # noqa: BLE001
            pass
        try:
            poly = sympy.Poly(main, indep)
            if poly.degree() == 1:
                slope, intercept = poly.all_coeffs()
                intermediate.update({"slope": slope, "intercept": intercept})
            elif poly.degree() == 2:
                a, b, c = poly.all_coeffs()
                vx = -b / (2 * a)
                intermediate.update({"a": a, "b": b, "c": c, "vertex": (vx, sympy.simplify(main.subs(indep, vx)))})
        except Exception:  # noqa: BLE001
            pass

        answer = " or ".join(f"{dep} = {latex(sol)}" for sol in solutions)
        return SolverResult(answer=answer, exact_value=solutions, numerical_value=None,
                            engine_used="sympy", intermediate=intermediate)

    def _solve_expression(self, parsed: ParsedProblem) -> SolverResult:
        """Evaluate a numeric expression or simplify an algebraic/trig one."""
        expr = self._target_expression(parsed)
        candidates: list[tuple[str, sympy.Expr]] = []
        for name, fn in (("simplify", sympy.simplify), ("trigsimp", sympy.trigsimp),
                         ("factor", sympy.factor), ("expand", sympy.expand)):
            try:
                candidates.append((name, fn(expr)))
            except Exception:  # noqa: BLE001
                continue
        candidates.append(("original", expr))
        method, best = min(candidates, key=lambda kv: sympy.count_ops(kv[1]))

        constant = not expr.free_symbols
        numeric: Optional[float] = None
        if constant:
            try:
                numeric = float(sympy.N(best))
            except (TypeError, ValueError):
                numeric = None
        answer = latex(best)
        if numeric is not None and not (best.is_Integer or best.is_Rational):
            answer += f" \\approx {numeric:.6g}"
        return SolverResult(
            answer=answer, exact_value=best,
            numerical_value=[numeric] if numeric is not None else None, engine_used="sympy",
            intermediate={"expression": expr, "simplified": best, "method": method,
                          "constant": constant, "numeric": numeric},
        )

    def _solve_system(self, parsed: ParsedProblem) -> SolverResult:
        """Solve a system of simultaneous equations via ``sympy.solve``."""
        if len(parsed.equations) < 2:
            raise SolverError("Need at least 2 equations for a system.")
        solution = solve(parsed.equations, list(parsed.symbols))
        if not solution:
            raise SolverError("SymPy found no solution for this system.")

        if isinstance(solution, dict):
            answer = ", ".join(f"{sym} = {latex(val)}" for sym, val in solution.items())
            numerical = self._safe_floats(list(solution.values()))
        else:  # list of tuples (multiple solution sets)
            answer = "; ".join(
                ", ".join(f"{sym} = {latex(val)}" for sym, val in zip(sorted(parsed.symbols, key=str), sol))
                for sol in solution
            )
            numerical = None

        return SolverResult(
            answer=answer,
            exact_value=solution,
            numerical_value=numerical,
            engine_used="sympy",
            intermediate={"equations": parsed.equations, "solution": solution},
        )

    def _solve_derivative(self, parsed: ParsedProblem) -> SolverResult:
        """Compute a derivative, extracting the target expression if needed."""
        expr = self._target_expression(parsed)
        free_syms = sorted(expr.free_symbols, key=str) or [sympy.Symbol("x")]
        symbol = free_syms[0]

        original = expr.args[0] if isinstance(expr, sympy.Derivative) else expr
        derivative = sympy.diff(original, symbol)

        return SolverResult(
            answer=latex(derivative),
            exact_value=derivative,
            numerical_value=None,
            engine_used="sympy",
            intermediate={"original": original, "symbol": symbol, "derivative": derivative},
        )

    def _solve_integral(self, parsed: ParsedProblem) -> SolverResult:
        """Compute a definite or indefinite integral."""
        expr = self._target_expression(parsed)

        if isinstance(expr, sympy.Integral):
            result = expr.doit()
            is_definite = len(expr.limits) > 0 and len(expr.limits[0]) == 3
        else:
            free_syms = sorted(expr.free_symbols, key=str) or [sympy.Symbol("x")]
            symbol = free_syms[0]
            result = sympy.integrate(expr, symbol)
            is_definite = False

        answer = latex(result) if is_definite else f"{latex(result)} + C"
        return SolverResult(
            answer=answer,
            exact_value=result,
            numerical_value=self._safe_floats([result]) if is_definite else None,
            engine_used="sympy",
            intermediate={"integrand": expr, "result": result, "is_definite": is_definite},
        )

    def _solve_limit(self, parsed: ParsedProblem) -> SolverResult:
        """Evaluate a limit expression (already containing a Limit object)."""
        expr = self._target_expression(parsed)
        result = expr.doit() if isinstance(expr, sympy.Limit) else expr
        return SolverResult(
            answer=latex(result),
            exact_value=result,
            numerical_value=self._safe_floats([result]),
            engine_used="sympy",
            intermediate={"limit_expr": expr, "result": result},
        )

    def _solve_matrix(self, parsed: ParsedProblem) -> SolverResult:
        """Compute determinant/inverse/eigenvalues for a matrix expression."""
        expr = self._target_expression(parsed)
        matrices = [a for a in sympy.preorder_traversal(expr) if isinstance(a, Matrix)]
        if not matrices:
            raise SolverError("No matrix found to operate on.")
        matrix = matrices[0]

        det = matrix.det() if matrix.is_square else None
        intermediate: dict[str, Any] = {"matrix": matrix, "determinant": det}
        answer_parts = [f"\\det = {latex(det)}"] if det is not None else []

        if matrix.is_square and det != 0:
            try:
                inverse = matrix.inv()
                intermediate["inverse"] = inverse
                answer_parts.append(f"A^{{-1}} = {latex(inverse)}")
            except Exception:  # noqa: BLE001
                pass
        if matrix.is_square:
            try:
                eigenvals = matrix.eigenvals()
                intermediate["eigenvalues"] = eigenvals
                answer_parts.append(f"\\lambda = {latex(list(eigenvals.keys()))}")
            except Exception:  # noqa: BLE001
                pass

        return SolverResult(
            answer="; ".join(answer_parts) if answer_parts else str(matrix),
            exact_value=matrix,
            numerical_value=None,
            engine_used="sympy",
            intermediate=intermediate,
        )

    @staticmethod
    def _target_expression(parsed: ParsedProblem) -> sympy.Expr:
        """Return the expression to operate on (from an equation's LHS or a bare expr)."""
        if parsed.expression is not None:
            return parsed.expression
        if parsed.equations:
            return parsed.equations[0].lhs
        raise SolverError("No expression or equation found to solve.")

    @staticmethod
    def _safe_floats(values: list) -> Optional[list[float]]:
        """Best-effort conversion of SymPy values to floats (skips complex/symbolic)."""
        floats = []
        for v in values:
            try:
                floats.append(float(v))
            except (TypeError, ValueError):
                continue
        return floats or None

    # ------------------------------------------------------------------
    # 3C — WolframAlpha (fallback)
    # ------------------------------------------------------------------
    def _solve_with_wolfram(self, query: str) -> SolverResult:
        """Query the WolframAlpha Short Answers / Full Results API.

        Args:
            query: A plain-text or LaTeX-ish query string.

        Returns:
            A :class:`SolverResult` with ``engine_used="wolframalpha"``.

        Raises:
            QuotaExceededError: If the monthly free quota is used up.
            SolverError: If the API call fails or returns no result.
        """
        try:
            import wolframalpha
        except ImportError as exc:
            raise SolverError(
                "WolframAlpha fallback requires the 'wolframalpha' package. "
                "Install with: pip install wolframalpha"
            ) from exc

        self.quota_tracker.record_call()

        client = wolframalpha.Client(self.wolfram_app_id)
        for attempt in range(settings.api_max_retries):
            try:
                res = client.query(query)
                answer_text = next(res.results).text
                return SolverResult(
                    answer=answer_text,
                    exact_value=None,
                    numerical_value=None,
                    engine_used="wolframalpha",
                    intermediate={"query": query},
                )
            except StopIteration as exc:
                raise SolverError(f"WolframAlpha returned no results for: {query!r}") from exc
            except Exception:  # noqa: BLE001
                if attempt == settings.api_max_retries - 1:
                    raise
                time.sleep(settings.api_backoff_seconds * (attempt + 1))
        raise SolverError("WolframAlpha query failed after retries.")

    # ------------------------------------------------------------------
    # 3D — Gemini (word problems + verification)
    # ------------------------------------------------------------------
    def _solve_with_gemini(self, prompt_text: str) -> SolverResult:
        """Ask Gemini to solve a word problem or otherwise-unsolvable query.

        Args:
            prompt_text: The problem statement (natural language and/or LaTeX).

        Returns:
            A :class:`SolverResult` with ``engine_used="gemini"``.

        Raises:
            SolverError: If the API call fails or Gemini isn't configured.
        """
        if not self.gemini_api_key:
            raise SolverError("GEMINI_API_KEY is not configured.")

        system_prompt = (
            "You are a precise math solver for Indian school students. "
            "Solve the following problem and reply with ONLY the final numeric "
            "or symbolic answer on the first line, then a blank line, then a "
            "short 2-3 sentence explanation of the method used."
        )

        try:
            text = generate_text(f"Problem: {prompt_text}", system=system_prompt, api_key=self.gemini_api_key)
        except GeminiError as exc:
            raise SolverError(f"Gemini solve failed for {prompt_text!r}: {exc}") from exc
        return SolverResult(
            answer=text.split("\n", 1)[0].strip(),
            exact_value=None,
            numerical_value=None,
            engine_used="gemini",
            intermediate={"full_response": text},
        )

    # ------------------------------------------------------------------
    # Cross-verification (3D secondary use)
    # ------------------------------------------------------------------
    def verify_with_gemini(self, problem_text: str, sympy_answer: str) -> bool:
        """Ask Gemini to independently confirm a SymPy-derived answer.

        Used to flag potential edge cases (e.g. extraneous roots, domain
        restrictions) before showing the answer to a student.

        Args:
            problem_text: The original problem statement.
            sympy_answer: The answer SymPy computed.

        Returns:
            True if Gemini agrees (or verification is unavailable, in
            which case we don't want to block the pipeline); False only
            if Gemini explicitly disagrees.
        """
        if not self.gemini_api_key:
            return True  # can't verify without a key; don't block the pipeline

        try:
            verdict = generate_text(
                f"Problem: {problem_text}\nProposed answer: {sympy_answer}\n"
                "Reply with exactly one word: AGREE or DISAGREE.",
                api_key=self.gemini_api_key,
                temperature=0.0,
            ).upper()
            return "DISAGREE" not in verdict
        except GeminiError:
            logger.warning("Gemini verification call failed; assuming agreement", exc_info=True)
            return True


if __name__ == "__main__":
    from src.latex_parser import LatexParser
    from src.problem_classifier import ProblemClassifier

    latex_parser = LatexParser()
    classifier = ProblemClassifier()
    solver = MathSolver()

    for sample in [r"x^2 - 5x + 6 = 0", r"\frac{d}{dx} x^3", r"\int x^2 dx", r"2x+y=5 \\ x-y=1"]:
        parsed_problem = latex_parser.parse(sample)
        ptype = classifier.classify(parsed_problem).problem_type
        result = solver.solve(parsed_problem, ptype)
        print(f"{sample!r} [{ptype.value}] -> {result.answer} (via {result.engine_used})")
