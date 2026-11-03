"""LLM explanation engine (Module 4).

Turns the exact SymPy step sequence from :mod:`src.step_extractor`
into a warm, student-friendly explanation at the level the student
selects (Class 8-10 / Class 11-12 / JEE-Advanced), in Hindi, English,
or Hinglish, using Gemini 1.5 Flash.

The prompt-engineering follows the project spec exactly: "MathGuru"
persona, per-step "why" annotations, a common-mistakes callout, a
memory trick, and a follow-up practice problem.

This module never crashes the pipeline if Gemini is unavailable
(missing key, network error, quota) — :meth:`LLMExplainer.explain`
degrades gracefully to a deterministic, template-based explanation
built directly from the :class:`~src.step_extractor.StepSequence`, so
the product still works end-to-end offline / without an API key.

Example:
    >>> from src.llm_explainer import LLMExplainer, ExplanationLevel, Language
    >>> explainer = LLMExplainer()
    >>> result = explainer.explain(step_sequence, level=ExplanationLevel.CLASS_10, language=Language.HINGLISH)
    >>> result.source
    'template'  # or 'gemini' if GEMINI_API_KEY is configured
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.config import settings
from src.gemini_client import GeminiError, generate_text
from src.step_extractor import Step, StepSequence

logger = logging.getLogger(__name__)


class ExplanationLevel(str, Enum):
    """Student level selector shown on Screen 3."""

    CLASS_10 = "class_8_10"
    CLASS_12 = "class_11_12"
    JEE_ADVANCED = "jee_advanced"


class Language(str, Enum):
    """Explanation language selector."""

    ENGLISH = "english"
    HINDI = "hindi"
    HINGLISH = "hinglish"


class ExplanationStyle(str, Enum):
    """The four explanation styles from the product spec."""

    STANDARD = "standard"
    VISUAL = "visual"
    STORY = "story"
    SHORTCUT = "shortcut"


_LEVEL_DESCRIPTIONS = {
    ExplanationLevel.CLASS_10: "a Class 8-10 student — use very simple language and short sentences",
    ExplanationLevel.CLASS_12: "a Class 11-12 student — more detailed, can use standard mathematical terms",
    ExplanationLevel.JEE_ADVANCED: "a JEE/competitive-exam student — focus on shortcut methods and time-saving tricks",
}


@dataclass
class Explanation:
    """The final rendered explanation shown on Screen 3.

    Attributes:
        text: The full formatted explanation (step-by-step, in the
            requested language/level).
        common_mistakes: Bullet list of common mistakes to avoid.
        memory_trick: A short mnemonic/shortcut.
        practice_problem: A similar problem for the student to try next.
        difficulty_stars: 1-3 stars.
        source: ``"gemini"`` or ``"template"`` (fallback), so the caller
            can tell whether the richer LLM explanation was used.
    """

    text: str
    common_mistakes: list[str] = field(default_factory=list)
    memory_trick: str = ""
    practice_problem: str = ""
    difficulty_stars: int = 2
    source: str = "template"


class LLMExplainer:
    """Generates level-and-language-aware explanations from a step sequence."""

    def __init__(self, gemini_api_key: Optional[str] = None) -> None:
        """Create an explainer.

        Args:
            gemini_api_key: Overrides ``settings.gemini_api_key`` (useful for tests).
        """
        self.gemini_api_key = gemini_api_key or settings.gemini_api_key

    # ------------------------------------------------------------------
    def explain(
        self,
        step_sequence: StepSequence,
        level: ExplanationLevel = ExplanationLevel.CLASS_10,
        language: Language = Language.HINGLISH,
        style: ExplanationStyle = ExplanationStyle.STANDARD,
    ) -> Explanation:
        """Produce a student-facing explanation for a solved problem.

        Tries Gemini first (if configured); falls back to a
        deterministic template built from the step sequence itself,
        which is always available and requires no network access.

        Args:
            step_sequence: Output of :class:`src.step_extractor.StepExtractor`.
            level: Target student level.
            language: Target language.
            style: Explanation style (standard/visual/story/shortcut).

        Returns:
            An :class:`Explanation`.
        """
        if settings.enable_gemini_explanations and self.gemini_api_key:
            try:
                return self._explain_with_gemini(step_sequence, level, language, style)
            except Exception:  # noqa: BLE001
                logger.warning("Gemini explanation failed; falling back to template", exc_info=True)

        return self._explain_with_template(step_sequence, level, language)

    # ------------------------------------------------------------------
    def translate_steps(self, steps: list[Step], language: Language) -> list[Step]:
        """Translate the verified solution steps into Hindi or Hinglish.

        Only the wording (title, content, why) is translated; every number,
        symbol and formula stays exactly as the solver produced it.

        Args:
            steps: The verified English steps.
            language: Target language. English returns ``steps`` unchanged.

        Returns:
            New :class:`Step` objects (same numbering and ``verified`` flags).

        Raises:
            RuntimeError: If Gemini is not configured, fails, or returns a
                malformed answer (the caller keeps showing the English steps).
        """
        if language == Language.ENGLISH or not steps:
            return list(steps)
        if not (settings.enable_gemini_explanations and self.gemini_api_key):
            raise RuntimeError("Translating steps needs a Gemini API key (GEMINI_API_KEY).")

        target = {
            Language.HINDI: "Hindi written in Devanagari script",
            Language.HINGLISH: "Hinglish (Hindi words written in the Latin alphabet, mixed with simple English)",
        }[language]
        payload = json.dumps(
            [{"title": st.title, "content": st.content, "why": st.why} for st in steps], ensure_ascii=False
        )
        prompt = (
            f"Translate the text fields of these maths solution steps into {target}.\n"
            "RULES: keep every number, variable, symbol, equation and LaTeX command EXACTLY unchanged; "
            "translate only the surrounding words; keep the same number of items and the same keys "
            "(title, content, why). Reply with ONLY a JSON array, no markdown fences, no commentary.\n\n"
            f"{payload}"
        )
        try:
            raw = generate_text(prompt, api_key=self.gemini_api_key, temperature=0.1)
        except GeminiError as exc:
            raise RuntimeError(f"Step translation failed: {exc}") from exc

        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            items = json.loads(cleaned)
            if not isinstance(items, list) or len(items) != len(steps):
                raise ValueError("wrong number of items")
            return [
                Step(number=orig.number, title=str(new["title"]).strip(), content=str(new["content"]).strip(),
                     why=str(new["why"]).strip(), verified=orig.verified)
                for orig, new in zip(steps, items)
            ]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Step translation returned an unusable answer: {exc}") from exc

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        step_sequence: StepSequence,
        level: ExplanationLevel,
        language: Language,
        style: ExplanationStyle,
    ) -> str:
        """Build the MathGuru system+user prompt per the product spec."""
        steps_text = "\n".join(
            f"Step {s.number}: {s.title} — {s.content} (Why: {s.why})" for s in step_sequence.steps
        )
        language_instruction = {
            Language.ENGLISH: "Respond entirely in English.",
            Language.HINDI: "Respond entirely in Hindi (Devanagari script).",
            Language.HINGLISH: (
                "Respond in Hinglish (Hindi conversational tone, Latin script), "
                "the way an Indian tutor actually talks to students."
            ),
        }[language]

        style_instruction = {
            ExplanationStyle.STANDARD: "Explain step by step, plainly.",
            ExplanationStyle.VISUAL: "Describe what a diagram/graph of this would look like at each step.",
            ExplanationStyle.STORY: "Frame the explanation as a short relatable word-problem story.",
            ExplanationStyle.SHORTCUT: "Focus on the fastest JEE-style shortcut method, minimal steps.",
        }[style]

        return (
            "You are MathGuru, an expert Indian math teacher. "
            f"Explain this solution step by step for {_LEVEL_DESCRIPTIONS[level]}. "
            f"{language_instruction} {style_instruction} "
            "After each step, briefly explain why we do that step, not just what we did. "
            "In Hindi/Hinglish mode, occasionally ask 'Samjha?' after a key step. "
            "End your response with three clearly labeled sections:\n"
            "COMMON MISTAKES: (2-3 bullet points)\n"
            "MEMORY TRICK: (one short mnemonic or shortcut)\n"
            "PRACTICE PROBLEM: (one similar problem, same difficulty)\n\n"
            f"Solved steps (ground truth, do not change the math):\n{steps_text}\n\n"
            f"Final answer: {step_sequence.final_answer}"
        )

    def _explain_with_gemini(
        self,
        step_sequence: StepSequence,
        level: ExplanationLevel,
        language: Language,
        style: ExplanationStyle,
    ) -> Explanation:
        """Call Gemini (``settings.gemini_model``) to generate the rich explanation."""
        prompt = self._build_prompt(step_sequence, level, language, style)
        try:
            text = generate_text(prompt, api_key=self.gemini_api_key)
        except GeminiError as exc:
            raise RuntimeError(f"Gemini explanation failed: {exc}") from exc
        return self._parse_gemini_response(text, step_sequence)

    @staticmethod
    def _parse_gemini_response(text: str, step_sequence: StepSequence) -> Explanation:
        """Split Gemini's free-text response into structured fields."""
        common_mistakes: list[str] = []
        memory_trick = ""
        practice_problem = ""
        main_text = text

        markers = ["COMMON MISTAKES:", "MEMORY TRICK:", "PRACTICE PROBLEM:"]
        positions = {m: text.find(m) for m in markers if m in text}
        if positions:
            first_marker_pos = min(positions.values())
            main_text = text[:first_marker_pos].strip()

            if "COMMON MISTAKES:" in positions:
                start = positions["COMMON MISTAKES:"] + len("COMMON MISTAKES:")
                end = positions.get("MEMORY TRICK:", positions.get("PRACTICE PROBLEM:", len(text)))
                mistakes_block = text[start:end].strip()
                common_mistakes = [
                    line.strip("-• ").strip() for line in mistakes_block.splitlines() if line.strip()
                ]

            if "MEMORY TRICK:" in positions:
                start = positions["MEMORY TRICK:"] + len("MEMORY TRICK:")
                end = positions.get("PRACTICE PROBLEM:", len(text))
                memory_trick = text[start:end].strip()

            if "PRACTICE PROBLEM:" in positions:
                start = positions["PRACTICE PROBLEM:"] + len("PRACTICE PROBLEM:")
                practice_problem = text[start:].strip()

        return Explanation(
            text=main_text,
            common_mistakes=common_mistakes,
            memory_trick=memory_trick,
            practice_problem=practice_problem,
            difficulty_stars=step_sequence.difficulty,
            source="gemini",
        )

    # ------------------------------------------------------------------
    def _explain_with_template(
        self, step_sequence: StepSequence, level: ExplanationLevel, language: Language
    ) -> Explanation:
        """Deterministic, offline fallback explanation.

        Builds a readable explanation directly from the step sequence's
        own titles/content/why fields — always available, always
        mathematically correct (since it's derived straight from the
        verified SymPy steps), just less linguistically rich than the
        Gemini version.
        """
        prefix = {
            Language.ENGLISH: "Step",
            Language.HINDI: "चरण",
            Language.HINGLISH: "Step",
        }[language]
        suffix = " — Samjha?" if language in (Language.HINDI, Language.HINGLISH) else ""

        lines = []
        for step in step_sequence.steps:
            lines.append(f"{prefix} {step.number}: {step.title}")
            lines.append(f"  {step.content}")
            lines.append(f"  💡 Why: {step.why}{suffix if step.verified else ''}")
            lines.append("")
        lines.append(f"✅ Answer: {step_sequence.final_answer}")

        return Explanation(
            text="\n".join(lines),
            common_mistakes=[
                "Forgetting to check the domain/validity of the solution.",
                "Sign errors when moving terms across the equals sign.",
            ],
            memory_trick="Always verify by substituting your answer back into the original equation.",
            practice_problem="Try a similar problem with different numbers to practice this method.",
            difficulty_stars=step_sequence.difficulty,
            source="template",
        )


if __name__ == "__main__":
    from src.latex_parser import LatexParser
    from src.math_solver import MathSolver
    from src.problem_classifier import ProblemClassifier
    from src.step_extractor import StepExtractor

    parsed_problem = LatexParser().parse(r"x^2 - 5x + 6 = 0")
    ptype = ProblemClassifier().classify(parsed_problem).problem_type
    result = MathSolver().solve(parsed_problem, ptype)
    sequence = StepExtractor().extract(ptype, result)

    explainer = LLMExplainer()
    explanation = explainer.explain(sequence, level=ExplanationLevel.CLASS_10, language=Language.HINGLISH)
    print(explanation.text)
    print(f"\nSource: {explanation.source}")
