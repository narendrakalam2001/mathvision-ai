"""Light handwriting OCR backed by Gemini vision (no model weights, no torch).

Why this exists
---------------
The fine-tuned TrOCR checkpoint is 2.2 GB and needs >= 4 GB RAM, which does not
fit on Streamlit Community Cloud or Render's free tier. This module offers the
same interface as :class:`src.ocr_engine.OCREngine` (``recognize(image) ->
OCRResult``) but sends the cropped line image to the Gemini API instead, so
photo-to-solution works on a tiny host that only has ``GEMINI_API_KEY``.

The local TrOCR engine is still preferred automatically whenever its weights
and torch are present (see ``MathService.ocr_backend_name``).

Usage:
    from src.gemini_ocr import GeminiOCREngine

    result = GeminiOCREngine().recognize(image_array)
    print(result.latex, result.confidence)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from src.gemini_client import GeminiError, generate_from_image
from src.ocr_engine import OCRError, OCRResult, apply_symbol_corrections, validate_latex

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise OCR engine for handwritten and printed school mathematics "
    "(Indian NCERT / CBSE / JEE level). You only transcribe; you never solve."
)

_USER_PROMPT = (
    "Transcribe the mathematical expression in this image into LaTeX.\n"
    "Rules:\n"
    "- Output ONLY the LaTeX on a single line. No explanation, no markdown, no code fences.\n"
    "- Do not wrap it in $ or \\[ \\].\n"
    "- Keep the equals sign and every symbol exactly as written; do not simplify or solve.\n"
    "- Use \\frac{a}{b}, \\sqrt{x}, x^{2}, \\sin, \\int, \\sum, \\pi etc. where appropriate.\n"
    "- Distinguish carefully between 1/l/I, 0/O, x/\\times and minus/dash.\n"
    "- If the image has no readable math, output exactly: UNREADABLE"
)

# Confidence is a heuristic: Gemini returns no per-symbol probabilities.
_CONF_VALID = 0.80
_CONF_INVALID = 0.30


def clean_model_output(text: str) -> str:
    """Strip markdown fences, math delimiters and labels from a model reply.

    Args:
        text: Raw model reply.

    Returns:
        A single-line LaTeX string (possibly empty).
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"^(latex|LaTeX)\s*:\s*", "", cleaned).strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if cleaned.startswith(left) and cleaned.endswith(right) and len(cleaned) >= len(left) + len(right):
            cleaned = cleaned[len(left): len(cleaned) - len(right)].strip()
            break
    return re.sub(r"\s*\n\s*", " ", cleaned).strip()


def _encode_png(image: np.ndarray) -> bytes:
    """Encode an OpenCV image (gray or BGR) as PNG bytes."""
    if image is None or image.size == 0:
        raise OCRError("Empty image passed to Gemini OCR.")
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise OCRError("Could not encode the image for Gemini OCR.")
    return buf.tobytes()


class GeminiOCREngine:
    """Drop-in, weights-free replacement for :class:`src.ocr_engine.OCREngine`."""

    def __init__(self, model: Optional[str] = None) -> None:
        """Create the engine.

        Args:
            model: Optional Gemini model override (defaults to ``GEMINI_MODEL``).
        """
        self.model = model

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Read one cropped expression image and return validated LaTeX.

        Args:
            image: Cropped line image (gray or BGR ``np.ndarray``).

        Returns:
            An :class:`OCRResult` with ``engine_used="gemini"``.

        Raises:
            OCRError: Gemini failed, or the image has no readable math.
        """
        png = _encode_png(image)
        try:
            raw = generate_from_image(
                png, _USER_PROMPT, mime_type="image/png", system=_SYSTEM_PROMPT,
                model=self.model, temperature=0.0,
            )
        except GeminiError as exc:
            raise OCRError(f"Gemini OCR failed: {exc}") from exc

        latex = apply_symbol_corrections(clean_model_output(raw))
        if not latex or latex.upper() == "UNREADABLE":
            raise OCRError("No readable math was found in the image.")

        valid = validate_latex(latex)
        logger.info("Gemini OCR result: %r (valid=%s)", latex, valid)
        return OCRResult(
            latex=latex,
            raw_latex=raw.strip(),
            confidence=_CONF_VALID if valid else _CONF_INVALID,
            engine_used="gemini",
            is_valid=valid,
        )
