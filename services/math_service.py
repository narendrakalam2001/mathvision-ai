"""Orchestration layer (feeds both ``serving/api.py`` and ``frontend/app.py``).

This is the single place that wires all 6 modules together into the
handful of high-level operations the product actually needs:

    solve_image(...)     Module 1 -> 2 -> latex_parser -> 3A -> 3B/C/D -> 3E
    solve_latex(...)     Same pipeline, skipping OCR (typed/edited LaTeX)
    explain(...)         Module 4 on top of an already-computed step sequence
    speak(...)           Module 5 TTS
    transcribe(...)      Module 5 STT
    chat(...)            Module 6 MathBot
    practice(...)        problem_generator
    record_progress(...) student_profile

Keeping this orchestration in one service module means the FastAPI
layer and the Streamlit layer both call the exact same code path, so
behavior never diverges between the API and the demo UI.

Every method returns a plain, JSON-serializable dict (rather than raw
dataclasses) so it can be handed directly to FastAPI's response model
or Streamlit's session state without extra glue code.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import sympy

from src.graph_plotter import GraphPlotter, GraphPlottingError
from src.image_processor import ImageProcessingError, ImageProcessor
from src.layout_analyzer import LayoutAnalyzer
from src.latex_parser import LatexParseError, LatexParser
from src.llm_explainer import ExplanationLevel, ExplanationStyle, Language, LLMExplainer
from src.math_solver import MathSolver, SolverError
from src.mathbot import MathBot
from src.problem_classifier import ProblemClassifier, ProblemType
from src.problem_generator import ProblemGenerator
from src.step_extractor import Step, StepExtractor
from src.student_profile import StudentProfileStore
from src.voice_engine import VoiceEngine, VoiceEngineError, VoiceLanguage

logger = logging.getLogger(__name__)


class MathServiceError(Exception):
    """Raised when the pipeline fails in a way the caller must handle
    (e.g. return a 4xx to the student rather than a raw stack trace)."""


class MathService:
    """Facade over all 6 MathVision AI modules."""

    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.layout_analyzer = LayoutAnalyzer()
        self.latex_parser = LatexParser()
        self.classifier = ProblemClassifier()
        self.solver = MathSolver()
        self.step_extractor = StepExtractor()
        self.explainer = LLMExplainer()
        self.plotter = GraphPlotter()
        self.voice_engine = VoiceEngine()
        self.mathbot = MathBot()
        self.problem_generator = ProblemGenerator()
        self.profile_store = StudentProfileStore()

        # Lazily created; only needed if the OCR path is actually used.
        self._ocr_engine = None

    # ------------------------------------------------------------------
    def _local_ocr_available(self) -> bool:
        """True when the fine-tuned TrOCR weights AND torch/transformers are present.

        Uses ``find_spec`` so nothing heavy is imported just to check.
        """
        from importlib.util import find_spec

        from src.config import settings

        weights = (settings.trocr_finetuned_dir / "config.json").is_file()
        libs = find_spec("torch") is not None and find_spec("transformers") is not None
        return weights and libs

    def ocr_backend_name(self) -> str:
        """Which handwriting-OCR backend will be used: ``local``, ``gemini`` or ``none``."""
        from src.config import settings

        if self._local_ocr_available():
            return "local"
        if settings.gemini_api_key:
            return "gemini"
        return "none"

    def _get_ocr_engine(self):
        """Create the OCR engine on first use.

        * ``local``  -> fine-tuned TrOCR (+ Pix2Tex fallback), needs weights + torch.
        * ``gemini`` -> light cloud OCR (Gemini vision), needs only ``GEMINI_API_KEY``.
          This is what Streamlit Community Cloud uses (no 2 GB weights, no torch).
        """
        if self._ocr_engine is None:
            backend = self.ocr_backend_name()
            if backend == "local":
                from src.ocr_engine import OCREngine

                self._ocr_engine = OCREngine()
            elif backend == "gemini":
                from src.gemini_ocr import GeminiOCREngine

                self._ocr_engine = GeminiOCREngine()
            else:
                raise MathServiceError(
                    "Photo reading is not available on this server (no OCR model and no "
                    "GEMINI_API_KEY). Please type the equation instead."
                )
            logger.info("OCR backend selected: %s", backend)
        return self._ocr_engine

    def warm_up_ocr(self) -> bool:
        """Load the OCR model now (and self-check the fast decoder).

        Moves the ~20 s model load off the first student's request. Safe to call
        from a background thread; never raises (OCR just stays lazy on failure).
        The Gemini backend has nothing to load, so it returns ``True`` immediately.

        Returns:
            ``True`` if the OCR backend is loaded and ready.
        """
        try:
            engine = self._get_ocr_engine()
            trocr = getattr(engine, "_trocr", None)
            if trocr is not None:
                trocr.warm_up()
            logger.info("OCR warm-up complete")
            return True
        except Exception:  # noqa: BLE001 - e.g. weights not downloaded on this host
            logger.warning("OCR warm-up skipped", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # POST /solve
    # ------------------------------------------------------------------
    def solve_image(self, image: Any, line_index: int = 0) -> dict:
        """Full pipeline: photo -> quality check -> line split -> OCR -> solve -> steps.

        A photo may contain several expressions (one per line). Every line is
        recognised; ``line_index`` selects which one is solved (default: the
        first). All recognised lines are returned under ``ocr["lines"]`` so the
        UI can let the student pick a different one or edit the LaTeX.

        Args:
            image: File path, raw bytes, or ``np.ndarray`` (BGR) — see
                :class:`src.image_processor.ImageProcessor`.
            line_index: Zero-based index of the detected line to solve.

        Returns:
            A dict with keys: ``quality``, ``ocr``, ``problem_type``,
            ``answer``, ``steps``, ``graph_base64`` (nullable). If the
            handwriting was read but could not be solved, the dict holds
            ``quality``, ``ocr`` and ``solve_error`` instead. If the photo
            fails the quality check: ``quality`` and ``retake_recommended``.

        Raises:
            MathServiceError: On any unrecoverable pipeline failure,
                with a student-facing message in ``str(error)``.
        """
        try:
            processed = self.image_processor.process(image)
        except ImageProcessingError as exc:
            raise MathServiceError(f"Could not read the image: {exc}") from exc

        quality = asdict(processed.quality) | {"message": processed.quality.message}
        if not processed.quality.is_acceptable:
            return {"quality": quality, "retake_recommended": True}

        crops = self.layout_analyzer.analyze(processed.cropped)
        if not crops:
            raise MathServiceError("No handwriting was found in the image. Please take a clearer photo.")

        engine = self._get_ocr_engine()
        lines: list[dict] = []
        for crop in crops:
            try:
                ocr_result = engine.recognize(crop.image)
            except Exception as exc:  # noqa: BLE001
                raise MathServiceError(f"Could not read the handwriting: {exc}") from exc
            lines.append({
                "index": crop.index,
                "latex": ocr_result.latex,
                "confidence": ocr_result.confidence,
                "confidence_band": ocr_result.confidence_band(),
                "engine_used": ocr_result.engine_used,
                "needs_review": ocr_result.needs_review,
            })

        selected = lines[min(max(line_index, 0), len(lines) - 1)]
        ocr_payload = {**selected, "lines": lines, "selected_line": selected["index"]}
        try:
            result = self.solve_latex(selected["latex"])
        except MathServiceError as exc:
            # OCR worked but the text is not solvable (often a misread symbol):
            # return it so the UI can show the edit box instead of a dead end.
            return {"quality": quality, "ocr": ocr_payload, "solve_error": str(exc)}
        result["quality"] = quality
        result["ocr"] = ocr_payload
        return result

    # ------------------------------------------------------------------
    def solve_latex(self, latex: str, problem_text: Optional[str] = None) -> dict:
        """Solve an already-recognized (or typed/edited) LaTeX problem.

        Args:
            latex: The LaTeX to solve.
            problem_text: Optional accompanying word-problem narrative.

        Returns:
            A dict with keys: ``latex``, ``problem_type``, ``confidence``,
            ``answer``, ``steps`` (list of step dicts), ``difficulty``,
            ``graph_base64`` (nullable).

        Raises:
            MathServiceError: If parsing or solving fails entirely.
        """
        try:
            parsed = self.latex_parser.parse(latex)
        except LatexParseError as exc:
            raise MathServiceError(f"We could not understand this equation: {exc}") from exc

        classification = self.classifier.classify(parsed, problem_text=problem_text)

        try:
            solver_result = self.solver.solve(parsed, classification.problem_type, problem_text)
        except SolverError as exc:
            raise MathServiceError(f"This problem could not be solved: {exc}") from exc

        step_sequence = self.step_extractor.extract(classification.problem_type, solver_result)

        graph_base64 = None
        try:
            graph_base64 = self._maybe_plot(parsed, solver_result, classification.problem_type)
        except Exception:  # noqa: BLE001 - a missing graph must never fail the solve
            logger.debug("Graph plotting skipped/failed for %r", latex, exc_info=True)

        return {
            "latex": latex,
            "problem_type": classification.problem_type.value,
            "classification_confidence": classification.confidence,
            "answer": step_sequence.final_answer,
            "steps": [asdict(s) for s in step_sequence.steps],
            "difficulty": step_sequence.difficulty,
            "engine_used": solver_result.engine_used,
            "graph_base64": graph_base64,
        }

    def _maybe_plot(self, parsed, solver_result, problem_type=None) -> Optional[str]:
        """Plot a graph when one is meaningful.

        * two-variable relations (``y = 2x + 3``): plot y against x
        * differentiation / integration: plot the resulting function
        * single-variable equations / expressions: plot the function with its roots
        """
        inter = solver_result.intermediate
        if inter.get("two_variable"):
            return self.plotter.plot_function(
                inter["plot_expr"], inter["plot_symbol"], roots=inter.get("plot_roots")
            )

        if problem_type in (ProblemType.DIFFERENTIATION, ProblemType.INTEGRATION):
            result_expr = solver_result.exact_value
            if isinstance(result_expr, sympy.Expr) and len(result_expr.free_symbols) == 1:
                return self.plotter.plot_function(result_expr, next(iter(result_expr.free_symbols)))
            return None

        target = parsed.equations[0] if parsed.equations else parsed.expression
        if target is None or isinstance(target, sympy.MatrixBase):
            return None
        expr = target.lhs - target.rhs if hasattr(target, "lhs") else target
        if len(expr.free_symbols) != 1:
            return None
        symbol = next(iter(expr.free_symbols))
        return self.plotter.plot_function(expr, symbol, roots=solver_result.numerical_value)

    def steps_in_language(self, steps: list[dict], language: str) -> dict:
        """Return the solution steps written in ``language``.

        Args:
            steps: Step dicts as returned by :meth:`solve_latex`.
            language: ``english`` (returned as-is), ``hindi`` or ``hinglish``.

        Returns:
            ``{"steps": [...], "translated": bool, "note": str}``. When the
            translation is unavailable the English steps are returned with
            ``translated=False`` and a short ``note`` for the UI.
        """
        try:
            target = Language(language)
        except ValueError as exc:
            raise MathServiceError(f"Unknown language: {language}") from exc
        if target == Language.ENGLISH:
            return {"steps": steps, "translated": False, "note": ""}
        sequence = [Step(**s) for s in steps]
        try:
            translated = self.explainer.translate_steps(sequence, target)
        except RuntimeError as exc:
            logger.warning("Step translation unavailable: %s", exc)
            return {"steps": steps, "translated": False, "note": str(exc)}
        return {"steps": [asdict(s) for s in translated], "translated": True, "note": ""}

    # ------------------------------------------------------------------
    # POST /explain
    # ------------------------------------------------------------------
    def explain(
        self,
        latex: str,
        level: str = "class_8_10",
        language: str = "hinglish",
        style: str = "standard",
    ) -> dict:
        """Generate a level/language/style-aware explanation for a problem.

        Args:
            latex: The problem's LaTeX (re-solved internally so this
                endpoint is stateless / cacheable).
            level: One of :class:`~src.llm_explainer.ExplanationLevel` values.
            language: One of :class:`~src.llm_explainer.Language` values.
            style: One of :class:`~src.llm_explainer.ExplanationStyle` values.

        Returns:
            A dict with keys: ``text``, ``common_mistakes``,
            ``memory_trick``, ``practice_problem``, ``difficulty_stars``,
            ``source``.
        """
        solved = self.solve_latex(latex)
        # Rebuild the StepSequence object the explainer expects.
        from src.step_extractor import Step, StepSequence

        steps = [Step(**s) for s in solved["steps"]]
        sequence = StepSequence(steps=steps, final_answer=solved["answer"], difficulty=solved["difficulty"])

        explanation = self.explainer.explain(
            sequence,
            level=ExplanationLevel(level),
            language=Language(language),
            style=ExplanationStyle(style),
        )
        return asdict(explanation)

    # ------------------------------------------------------------------
    # POST /chat
    # ------------------------------------------------------------------
    def chat(self, session_id: str, message: str, problem_latex: Optional[str] = None,
             language: str = "english") -> dict:
        """Route a doubt-clearing message to MathBot.

        Args:
            session_id: Unique chat session id.
            message: The student's message.
            problem_latex: If provided, (re)sets the bot's problem context.
            language: ``english`` (default), ``hindi`` or ``hinglish`` (Gemini replies).

        Returns:
            A dict with keys: ``text``, ``graph_base64`` (nullable), ``source``.
        """
        if problem_latex:
            self.mathbot.set_context(session_id, problem_latex)
        reply = self.mathbot.ask(session_id, message, language)
        return asdict(reply)

    # ------------------------------------------------------------------
    # POST /voice-question
    # ------------------------------------------------------------------
    def voice_question(self, session_id: str, audio_path: str) -> dict:
        """Transcribe a spoken doubt and route it through MathBot.

        Args:
            session_id: Unique chat session id (context should already
                be set via a prior :meth:`chat` call).
            audio_path: Path to the recorded audio file.

        Returns:
            A dict with keys: ``transcribed_text``, ``reply`` (the
            :meth:`chat`-shaped bot reply dict).

        Raises:
            MathServiceError: If transcription fails.
        """
        try:
            transcription = self.voice_engine.transcribe(audio_path)
        except VoiceEngineError as exc:
            raise MathServiceError(f"The voice input could not be understood: {exc}") from exc

        reply = self.chat(session_id, transcription.text)
        return {"transcribed_text": transcription.text, "reply": reply}

    def synthesize_speech(self, text: str, language: str = "hinglish") -> bytes:
        """Convert explanation text to speech.

        Args:
            text: Text to speak.
            language: One of :class:`~src.voice_engine.VoiceLanguage` values.

        Returns:
            Raw MP3 bytes.

        Raises:
            MathServiceError: If every TTS backend fails.
        """
        try:
            audio = self.voice_engine.synthesize(text, language=VoiceLanguage(language))
            return audio.audio_bytes
        except VoiceEngineError as exc:
            raise MathServiceError(f"The voice could not be generated: {exc}") from exc

    # ------------------------------------------------------------------
    # GET /practice/{topic}
    # ------------------------------------------------------------------
    def practice(self, topic: str) -> dict:
        """Generate one fresh, SymPy-verified practice problem for a topic.

        Args:
            topic: Topic keyword or a reference problem's LaTeX.

        Returns:
            A dict with keys: ``topic``, ``class_level``, ``latex_problem``,
            ``latex_answer``, ``difficulty``.
        """
        problem = self.problem_generator.generate_similar(topic)
        return asdict(problem)

    # ------------------------------------------------------------------
    # GET /student-profile
    # ------------------------------------------------------------------
    def student_profile(self, student_id: str) -> dict:
        """Fetch a student's dashboard data (Screen 5).

        Args:
            student_id: Unique student identifier.

        Returns:
            A dict with keys: ``summary`` (totals/streak/badges),
            ``heatmap`` (per-topic strength), ``weak_topics``,
            ``recent_solutions``.
        """
        profile = self.profile_store.get_or_create(student_id)
        return {
            "student_id": profile.student_id,
            "name": profile.name,
            "class_level": profile.class_level,
            "summary": profile.dashboard_summary(),
            "heatmap": profile.heatmap(),
            "weak_topics": profile.weak_topics(),
            "recent_solutions": [asdict(r) for r in profile.recent_solutions],
            "notebook": list(profile.notebook),
        }

    def record_attempt(
        self, student_id: str, topic: str, correct: bool, solve_time_seconds: float, latex_problem: str = ""
    ) -> dict:
        """Record a solved/attempted problem for a student and persist it.

        Args:
            student_id: Unique student identifier.
            topic: Classified topic name.
            correct: Whether the student marked it as correctly solved.
            solve_time_seconds: Time taken.
            latex_problem: The problem's LaTeX, for the recent-solutions list.

        Returns:
            A dict with key ``newly_earned_badges`` (list of badge names).
        """
        profile = self.profile_store.get_or_create(student_id)
        newly_earned = profile.record_attempt(topic, correct, solve_time_seconds, latex_problem)
        self.profile_store.save(profile)
        return {"newly_earned_badges": newly_earned}

    def mark_last_attempt(self, student_id: str, correct: bool) -> bool:
        """Update the self-assessment of a student's latest attempt (and persist it).

        Args:
            student_id: Unique student identifier.
            correct: ``False`` when the student says they did not understand.

        Returns:
            ``True`` if the record changed.
        """
        profile = self.profile_store.get_or_create(student_id)
        changed = profile.mark_last_attempt(correct)
        if changed:
            self.profile_store.save(profile)
        return changed

    def save_to_notebook(self, student_id: str, latex_problem: str, topic: str, answer: str = "") -> bool:
        """Bookmark a solved problem in the student's notebook.

        Returns:
            ``True`` if it was added, ``False`` if it was already saved.
        """
        profile = self.profile_store.get_or_create(student_id)
        added = profile.save_to_notebook(latex_problem, topic, answer)
        if added:
            self.profile_store.save(profile)
        return added

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """Report which optional backends are actually available right now.

        Returns:
            A dict of component -> bool/str status, used by the
            ``/health`` endpoint and the Streamlit sidebar.
        """
        from src.config import settings

        def _importable(module_name: str) -> bool:
            try:
                __import__(module_name)
                return True
            except ImportError:
                return False

        return {
            "status": "ok",
            "sympy_solver": True,  # always available, no external deps
            "ocr_backend_installed": _importable("transformers") and _importable("torch"),
            "ocr_backend": self.ocr_backend_name(),  # local | gemini | none
            "wolfram_configured": bool(settings.wolframalpha_app_id),
            "gemini_configured": bool(settings.gemini_api_key),
            "elevenlabs_configured": bool(settings.elevenlabs_api_key),
            "rag_backend_installed": _importable("chromadb"),
            "voice_stt_installed": _importable("whisper"),
        }
