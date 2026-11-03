"""MathBot — doubt-clearing chatbot (Module 6).

A context-aware chatbot that knows which problem the student is
currently looking at and can: re-explain a step differently, generate
a new graph, show the JEE shortcut (Vieta's formulas etc.), give a
similar practice problem, or explain a related concept via RAG.

Gemini (via :mod:`src.gemini_client`) answers free-form doubts, grounded in the
and a ``ConversationBufferMemory`` per chat session (keyed by
``session_id``, so a Streamlit/FastAPI server can hold many students'
conversations at once).

Tools exposed to the agent (matching the product spec):
    * ``solve_equation``   -> :mod:`src.math_solver`
    * ``explain_concept``  -> :mod:`src.rag_engine` (RAG over NCERT)
    * ``similar_problems``  -> :mod:`src.problem_generator`
    * ``plot_graph``        -> :mod:`src.graph_plotter`
    * ``formula_lookup``    -> :mod:`src.rag_engine`

Like the other Day-2 modules that depend on external services
(Gemini), MathBot degrades gracefully: without a configured
``GEMINI_API_KEY`` it still answers the fixed set of "suggested doubt"
chips (re-explain, shortcut, similar problem, graph, formula) using
purely local logic, so the doubt-clearing UI is never a dead end.

Example:
    >>> from src.mathbot import MathBot
    >>> bot = MathBot()
    >>> bot.set_context(session_id="s1", problem_latex="x^2 - 5x + 6 = 0")
    >>> reply = bot.ask("s1", "What is the shortcut?")
    >>> "Vieta" in reply.text or "vieta" in reply.text.lower()
    True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.config import settings
from src.gemini_client import GeminiError, generate_text
from src.math_text import pretty_math
from src.graph_plotter import GraphPlotter, GraphPlottingError
from src.latex_parser import LatexParser, LatexParseError
from src.math_solver import MathSolver, SolverError
from src.problem_classifier import ProblemClassifier
from src.rag_engine import RAGEngine, RAGEngineError
from src.step_extractor import StepExtractor

logger = logging.getLogger(__name__)


@dataclass
class ProblemContext:
    """The problem the student is currently viewing/asking about.

    Attributes:
        problem_latex: The original OCR'd/typed LaTeX.
        step_sequence: The already-computed step-by-step solution, if any.
        answer: The final answer string, if already solved.
    """

    problem_latex: str
    step_sequence: object = None
    answer: str = ""


@dataclass
class BotReply:
    """MathBot's response to a single turn.

    Attributes:
        text: The chat message text (Hinglish tutor tone).
        graph_base64: Optional embedded graph, when the reply includes one.
        source: ``"gemini"`` or ``"local_fallback"``.
    """

    text: str
    graph_base64: Optional[str] = None
    source: str = "local_fallback"


# Suggested-doubt chip keywords -> handler name. Intents in _TOOL_INTENTS are answered
# by deterministic tools (graph image, practice problem, formula, shortcut) even when
# Gemini is available; everything else is a free-form doubt for Gemini.
_SUGGESTED_DOUBT_PATTERNS: dict[str, tuple[str, ...]] = {
    "reexplain": ("explain", "don't understand", "do not understand", "didn't understand", "not clear",
                  "again", "nahi samjha", "phir se"),
    "shortcut": ("shortcut", "trick", "jee"),
    "similar_problem": ("similar", "another problem", "one more", "practice", "aur ek"),
    "graph": ("graph", "plot", "draw", "dikhao"),
    "formula": ("formula", "kya use hua"),
}
_TOOL_INTENTS = ("shortcut", "similar_problem", "graph", "formula")


class MathBot:
    """Context-aware, tool-using doubt-clearing chatbot."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        rag_engine: Optional[RAGEngine] = None,
    ) -> None:
        """Create a MathBot instance.

        Args:
            gemini_api_key: Overrides ``settings.gemini_api_key``.
            rag_engine: Injectable :class:`~src.rag_engine.RAGEngine` (for tests).
        """
        self.gemini_api_key = gemini_api_key or settings.gemini_api_key
        self.rag_engine = rag_engine or RAGEngine()
        self.latex_parser = LatexParser()
        self.classifier = ProblemClassifier()
        self.solver = MathSolver()
        self.step_extractor = StepExtractor()
        self.plotter = GraphPlotter()

        self._contexts: dict[str, ProblemContext] = {}
        self._histories: dict[str, list[tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # Session/context management
    # ------------------------------------------------------------------
    def set_context(self, session_id: str, problem_latex: str) -> None:
        """Tell MathBot which problem the student is currently viewing.

        Args:
            session_id: Unique chat/session identifier.
            problem_latex: The problem's LaTeX (drives solve/graph/formula tools).
        """
        context = ProblemContext(problem_latex=problem_latex)
        try:
            parsed = self.latex_parser.parse(problem_latex)
            ptype = self.classifier.classify(parsed).problem_type
            result = self.solver.solve(parsed, ptype)
            context.step_sequence = self.step_extractor.extract(ptype, result)
            context.answer = context.step_sequence.final_answer
        except (LatexParseError, SolverError):
            logger.warning("Could not pre-solve context problem: %s", problem_latex, exc_info=True)

        self._contexts[session_id] = context
        self._histories.setdefault(session_id, [])

    def suggested_chips(self) -> list[str]:
        """Return the fixed suggested-doubt chips shown in Screen 4."""
        return [
            "Explain step 3",
            "What's the shortcut?",
            "Give a similar problem",
            "Show the graph",
            "Which formula was used?",
        ]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def ask(self, session_id: str, message: str, language: str = "english") -> BotReply:
        """Answer a student's doubt about the current problem.

        Args:
            session_id: Chat session id (from :meth:`set_context`).
            message: The student's message (typed or Whisper-transcribed).
            language: ``english`` (default), ``hindi`` or ``hinglish`` -- the language
                Gemini answers in. Offline fallback replies are English.

        Returns:
            A :class:`BotReply`.
        """
        if not message.strip():                    # e.g. "Ask Doubt" only sets the context
            return BotReply(text="Ask me anything about this problem. I'm here to help! 😊")
        context = self._contexts.get(session_id)
        self._histories.setdefault(session_id, []).append(("user", message))

        tool = self._detect_tool_intent(message)
        if tool is not None:                       # graph / practice / formula / shortcut: exact tools
            reply = getattr(self, f"_handle_{tool}")(context)
        elif settings.enable_gemini_explanations and self.gemini_api_key:
            try:
                reply = self._ask_gemini_grounded(session_id, message, context, language)
            except GeminiError:
                logger.warning("Gemini chat failed; using local fallback", exc_info=True)
                reply = self._ask_local_fallback(message, context)
        else:
            reply = self._ask_local_fallback(message, context)
        self._histories[session_id].append(("bot", reply.text))
        return reply

    @staticmethod
    def _detect_tool_intent(message: str) -> Optional[str]:
        """Return the deterministic-tool intent named in ``message``, if any."""
        lowered = message.lower()
        for intent in _TOOL_INTENTS:
            if any(kw in lowered for kw in _SUGGESTED_DOUBT_PATTERNS[intent]):
                return intent
        return None

    # ------------------------------------------------------------------
    # Local (offline) fallback — always available
    # ------------------------------------------------------------------
    def _ask_local_fallback(self, message: str, context: Optional[ProblemContext]) -> BotReply:
        """Pattern-match the message to one of the fixed doubt-chip behaviors."""
        lowered = message.lower()

        for intent in (*_TOOL_INTENTS, "reexplain"):          # exact tools win over "explain"/"again"
            if any(kw in lowered for kw in _SUGGESTED_DOUBT_PATTERNS[intent]):
                handler: Callable[[Optional[ProblemContext]], BotReply] = getattr(self, f"_handle_{intent}")
                return handler(context)

        if context is None:
            return BotReply(text="Solve a problem first, then I can clear your doubts about it! 😊")

        return BotReply(
            text=(
                f"I couldn't tell exactly what you want to know. We are working on '{context.problem_latex}'. "
                "Try: 'shortcut', 'show the graph', 'give a similar problem' or 'explain step 2'."
            )
        )

    def _handle_reexplain(self, context: Optional[ProblemContext]) -> BotReply:
        if context is None or context.step_sequence is None:
            return BotReply(text="Solve a problem first, then I can re-explain its steps.")
        lines = ["Let me walk through it again:"]
        for step in context.step_sequence.steps:
            lines.append(f"• {step.title}: {pretty_math(step.content)} (because {pretty_math(step.why)})")
        return BotReply(text="\n".join(lines))

    def _handle_shortcut(self, context: Optional[ProblemContext]) -> BotReply:
        if context is None:
            return BotReply(text="Solve a problem first, then I can suggest a shortcut!")
        intermediate = getattr(context.step_sequence, "steps", None)
        a = b = c = None
        try:
            parsed = self.latex_parser.parse(context.problem_latex)
            if parsed.equations:
                import sympy

                lhs_minus_rhs = sympy.expand(parsed.equations[0].lhs - parsed.equations[0].rhs)
                symbol = sorted(lhs_minus_rhs.free_symbols, key=str)[0]
                poly = sympy.Poly(lhs_minus_rhs, symbol)
                if poly.degree() == 2:
                    a, b, c = poly.all_coeffs()
        except Exception:  # noqa: BLE001
            pass

        if a is not None:
            return BotReply(
                text=(
                    "JEE trick: Vieta's formulas!\n"
                    f"Sum of roots = -b/a = {-b/a}\n"
                    f"Product of roots = c/a = {c/a}\n"
                    "Faster than full factorization — verify by checking sum & product match!"
                )
            )
        return BotReply(
            text="There is no special shortcut for this problem, but a general tip: "
            "try to match known identities or patterns first, and brute-force only if that fails."
        )

    def _handle_similar_problem(self, context: Optional[ProblemContext]) -> BotReply:
        from src.problem_generator import ProblemGenerator

        generator = ProblemGenerator()
        try:
            problem = generator.generate_similar(context.problem_latex if context else None)
            return BotReply(text=f"Try this: {pretty_math(problem.latex_problem)}\nHint: use the same method we just used.")
        except Exception:  # noqa: BLE001
            logger.warning("Could not generate similar problem", exc_info=True)
            return BotReply(text="I could not generate a similar problem right now. Please try again in a moment.")

    def _handle_graph(self, context: Optional[ProblemContext]) -> BotReply:
        if context is None:
            return BotReply(text="Solve an equation first, then I can show its graph!")
        try:
            parsed = self.latex_parser.parse(context.problem_latex)
            expr = parsed.equations[0].lhs - parsed.equations[0].rhs if parsed.equations else parsed.expression
            symbol = sorted(expr.free_symbols, key=str)[0]
            roots = None
            if context.step_sequence and "solutions" in getattr(context.step_sequence, "__dict__", {}):
                roots = context.step_sequence.solutions
            png = self.plotter.plot_function(expr, symbol, roots=roots)
            return BotReply(
                text="Here is the graph. Notice where the curve crosses the x-axis: those points are the solutions!",
                graph_base64=png,
            )
        except (GraphPlottingError, LatexParseError, IndexError):
            logger.warning("Could not plot graph for doubt", exc_info=True)
            return BotReply(text="Sorry, I could not plot a graph for this problem.")

    def _handle_formula(self, context: Optional[ProblemContext]) -> BotReply:
        query = context.problem_latex if context else "quadratic formula"
        doc = self.rag_engine.formula_lookup(query)
        if doc is None:
            return BotReply(
                text="Formula lookup is not available yet (the knowledge base has not been ingested). "
                "Run `python scripts/ingest_knowledge.py` to enable it."
            )
        return BotReply(text=f"Formula used: {doc.metadata.get('name', '')} — {pretty_math(doc.text)}")

    # ------------------------------------------------------------------
    # Gemini path: free-form doubts, grounded in the solved problem + NCERT RAG
    # ------------------------------------------------------------------
    _LANGUAGE_RULE = {
        "english": "Reply in simple, clear English.",
        "hindi": "Reply in Hindi written in Devanagari script (keep maths symbols as they are).",
        "hinglish": "Reply in Hinglish: Hindi words in Roman script mixed with simple English.",
    }
    _SYSTEM_PROMPT = (
        "You are MathGuru, a friendly maths teacher for Indian school students. {language_rule} A student is "
        "stuck on ONE specific problem. Answer only the doubt they ask, using the "
        "problem, its verified solution steps and the reference notes below. "
        "Never contradict the verified final answer. Explain WHY, not just what. "
        "Keep it under 150 words, warm and encouraging, and end with a short "
        "check question such as 'Does that make sense?'."
    )

    def _ask_gemini_grounded(
        self, session_id: str, message: str, context: Optional[ProblemContext], language: str = "english"
    ) -> BotReply:
        """Answer ``message`` with Gemini, grounded in the verified solution and RAG notes.

        Raises:
            GeminiError: If the Gemini call fails (the caller falls back to local logic).
        """
        parts: list[str] = []
        if context is not None:
            parts.append(f"PROBLEM (LaTeX): {context.problem_latex}")
            if context.answer:
                parts.append(f"VERIFIED FINAL ANSWER: {context.answer}")
            if context.step_sequence is not None:
                steps = "\n".join(
                    f"{i}. {st.title}: {st.content} (why: {st.why})"
                    for i, st in enumerate(context.step_sequence.steps, start=1)
                )
                parts.append("VERIFIED SOLUTION STEPS:\n" + steps)
        try:
            hits = self.rag_engine.query(message, top_k=2)
        except RAGEngineError:
            hits = []
        if hits:
            parts.append("REFERENCE NOTES:\n" + "\n".join(f"- {h.text}" for h in hits))

        history = self._histories.get(session_id, [])[:-1][-6:]      # exclude the current message
        if history:
            parts.append("CHAT SO FAR:\n" + "\n".join(
                f"{'Student' if role == 'user' else 'MathGuru'}: {text}" for role, text in history))
        parts.append(f"STUDENT'S DOUBT: {message}")

        system = self._SYSTEM_PROMPT.format(language_rule=self._LANGUAGE_RULE.get(language, self._LANGUAGE_RULE["english"]))
        text = generate_text("\n\n".join(parts), system=system, api_key=self.gemini_api_key)
        return BotReply(text=text, source="gemini")


if __name__ == "__main__":
    bot = MathBot()
    bot.set_context("demo", r"x^2 - 5x + 6 = 0")

    for question in ["Explain step 3", "What is the shortcut?", "Show the graph", "Give a similar problem"]:
        reply = bot.ask("demo", question)
        print(f">>> {question}\n{reply.text}\n")
