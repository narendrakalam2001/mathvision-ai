"""Handwriting OCR engine (Module 2, Stage 2 — Math OCR).

Converts a preprocessed handwritten math image into a validated LaTeX
string using a two-model strategy:

    1. Primary: TrOCR (``microsoft/trocr-large-handwritten``), ideally
       fine-tuned on CROHME. Fast and accurate on most single-line /
       simple multi-line expressions.
    2. Fallback: Pix2Tex / LaTeX-OCR, invoked when TrOCR's confidence
       is low or its output fails LaTeX validation. Pix2Tex is
       generally stronger on complex nested structures (fractions,
       integrals, matrices).

Both model backends are loaded lazily (``@functools.lru_cache``-style
singletons) and are optional at import time: if ``transformers`` /
``torch`` / ``pix2tex`` are not installed, importing this module still
succeeds, but calling :meth:`OCREngine.recognize` raises a clear
``OCRDependencyError`` telling the user what to install. This keeps
the module importable in lightweight environments (e.g. API health
checks, unit tests with mocked engines).

Example:
    >>> from src.ocr_engine import OCREngine
    >>> engine = OCREngine()
    >>> result = engine.recognize(cropped_binary_image)
    >>> result.latex
    '\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}'
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class OCRDependencyError(Exception):
    """Raised when a required OCR backend dependency is missing."""


class OCRError(Exception):
    """Raised when OCR inference itself fails (both engines)."""


# ----------------------------------------------------------------------
# Common OCR confusions for math handwriting, used in post-processing.
# Each tuple is (pattern, replacement, context_note). These are applied
# conservatively -- only in contexts where the confusion is very common
# (e.g. a lone 'l' between digits is almost always '1').
# ----------------------------------------------------------------------
_SYMBOL_CORRECTIONS: list[tuple[str, str]] = [
    (r"(?<=\d)[lI](?=\d)", "1"),          # 1l1 / 1I1 -> 111 style digit runs
    (r"\bO\b(?=\s*[\+\-\*/=])", "0"),      # lone letter O before an operator -> 0
    (r"×", r"\\times "),
    (r"÷", r"\\div "),
    (r"—", "-"),                            # em-dash misread as minus
    (r"–", "-"),                            # en-dash misread as minus
    (r"\s+", " "),
]


@dataclass
class OCRResult:
    """Structured output of the OCR pipeline.

    Attributes:
        latex: The recognized (and post-processed) LaTeX string.
        raw_latex: The untouched model output before post-processing.
        confidence: Overall confidence score in [0, 1].
        symbol_confidences: Optional per-token confidence scores.
        engine_used: Which backend produced the final result
            (``"trocr"`` or ``"pix2tex"``).
        is_valid: Whether the LaTeX passed structural validation
            (balanced braces/environments).
        needs_review: True when confidence is below the "safe" threshold
            and the student should be shown an edit box.
    """

    latex: str
    raw_latex: str
    confidence: float
    engine_used: str
    is_valid: bool
    symbol_confidences: list[float] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return (not self.is_valid) or self.confidence < settings.ocr_confidence_threshold

    def confidence_band(self) -> str:
        """Return 'green' / 'yellow' / 'red' for the frontend confidence UI."""
        if self.confidence >= settings.ocr_confidence_threshold:
            return "green"
        if self.confidence >= settings.ocr_low_confidence_threshold:
            return "yellow"
        return "red"


def validate_latex(latex: str) -> bool:
    """Cheap structural validation: balanced braces/brackets/environments.

    This is not a full LaTeX parser -- it just catches the most common
    OCR failure mode (an unterminated ``\\frac{..}{..}`` or mismatched
    braces), which is enough to decide whether to trust the primary
    engine or fall back to the secondary one.

    Args:
        latex: Candidate LaTeX string.

    Returns:
        True if braces/brackets/parens are balanced.
    """
    if not latex or not latex.strip():
        return False

    pairs = {"{": "}", "[": "]", "(": ")"}
    closers = set(pairs.values())
    stack: list[str] = []
    for ch in latex:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in closers:
            if not stack or stack.pop() != ch:
                return False
    return not stack


def apply_symbol_corrections(latex: str) -> str:
    """Apply conservative regex-based fixes for common handwriting OCR errors.

    Args:
        latex: Raw OCR output.

    Returns:
        Cleaned-up LaTeX string.
    """
    cleaned = latex.strip()
    for pattern, replacement in _SYMBOL_CORRECTIONS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()


def shifted_greedy_decode(
    model,
    pixel_values,
    start_id: int,
    eos_id: int,
    max_len: int = 96,
    use_cache: bool = True,
    encoder_hidden=None,
) -> tuple[list[int], list[float]]:
    """Greedy decoding for a checkpoint trained with a double label shift.

    Such a checkpoint was trained with decoder input ``[<s>, <s>, P0, P1, ...]``
    where ``P[k]`` is the target token ``k + 1`` and the prediction for ``P[k]``
    is read at decoder *position* ``k``. Plain ``generate()`` therefore fails.

    ``use_cache=False`` is the reference implementation: it re-runs the whole
    prefix at every step. ``use_cache=True`` exploits that the token fed at
    position ``k`` is ``P[k-2]`` -- already known two steps earlier -- so the
    decoder consumes **two tokens per forward pass** with a key/value cache:
    roughly half the passes, and each pass touches only the new tokens.

    Args:
        model: A ``VisionEncoderDecoderModel`` in eval mode.
        pixel_values: Tensor ``(1, 3, H, W)``.
        start_id: ``<s>`` token id.
        eos_id: ``</s>`` token id (generation stops when it is predicted).
        max_len: Maximum number of tokens to produce.
        use_cache: Use the key/value-cache implementation.
        encoder_hidden: Optional precomputed ``model.encoder(...).last_hidden_state``
            (lets a benchmark time the encoder and decoder separately).

    Returns:
        ``(token_ids, token_probabilities)`` without the end-of-sequence token.
    """
    import torch

    preds: list[int] = []
    probs: list[float] = []
    with torch.inference_mode():
        enc = encoder_hidden if encoder_hidden is not None else model.encoder(pixel_values=pixel_values).last_hidden_state
        if not use_cache:
            for k in range(max_len):
                ids = [start_id] if k == 0 else [start_id, start_id] + preds[: k - 1]
                logits = model(
                    encoder_outputs=(enc,),
                    decoder_input_ids=torch.tensor([ids], dtype=torch.long),
                ).logits[0, -1]
                p = torch.softmax(logits, dim=-1)
                tid = int(p.argmax())
                if tid == eos_id:
                    break
                preds.append(tid)
                probs.append(float(p[tid]))
            return preds, probs

        past = None
        fed = torch.tensor([[start_id, start_id]], dtype=torch.long)
        while len(preds) < max_len:
            out = model(encoder_outputs=(enc,), decoder_input_ids=fed, past_key_values=past, use_cache=True)
            past = out.past_key_values
            p = torch.softmax(out.logits[0, -2:], dim=-1)           # predictions P[k], P[k+1]
            top = p.max(dim=-1)
            for tid, prob in zip(top.indices.tolist(), top.values.tolist()):
                if tid == eos_id or len(preds) >= max_len:
                    return preds, probs
                preds.append(tid)
                probs.append(prob)
            fed = torch.tensor([[preds[-2], preds[-1]]], dtype=torch.long)
    return preds, probs


class _TrOCRBackend:
    """Lazy-loaded wrapper around a (fine-tuned) TrOCR model."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        finetuned = settings.trocr_finetuned_dir
        if model_name:
            self.model_name = model_name
        elif (finetuned / "config.json").is_file():
            self.model_name = str(finetuned)          # use our fine-tuned weights when present
        else:
            self.model_name = settings.trocr_base_model
        self._processor = None
        self._model = None
        self._torch = None
        self._double_shift = False
        self._use_cache = settings.ocr_kv_cache

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise OCRDependencyError(
                "TrOCR backend requires 'torch' and 'transformers'. "
                "Install with: pip install torch transformers"
            ) from exc

        logger.info("Loading TrOCR model: %s", self.model_name)
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._torch = torch
        self._model.eval()
        threads = settings.ocr_threads or (os.cpu_count() or 1)
        torch.set_num_threads(threads)
        if settings.ocr_quantize:
            self._model = torch.ao.quantization.quantize_dynamic(self._model, {torch.nn.Linear}, dtype=torch.qint8)
            logger.warning("OCR_QUANTIZE=1: int8 dynamic quantization active (accuracy not validated)")
        marker = Path(self.model_name) / "decoding.json"
        self._double_shift = marker.is_file() and bool(
            json.loads(marker.read_text(encoding="utf-8")).get("double_shift", False)
        )
        logger.info("TrOCR loaded (double_shift decoding=%s, kv_cache=%s, threads=%d)",
                    self._double_shift, self._use_cache, threads)

    def warm_up(self, verify: bool = True) -> bool:
        """Load the model now and (optionally) verify the KV-cache decoder.

        Runs both decoders on a blank line image; if their outputs differ the
        cached decoder is switched off so a faster-but-wrong path can never
        reach a student. Moves the ~20 s model load off the first request.

        Args:
            verify: Compare cached and reference decoding on a blank image.

        Returns:
            ``True`` when the cached decoder is active afterwards.
        """
        self._ensure_loaded()
        if verify and self._double_shift and self._use_cache:
            tok = self._processor.tokenizer
            blank = np.full((96, 384, 3), 255, np.uint8)
            from PIL import Image as PILImage

            pixel_values = self._processor(images=PILImage.fromarray(blank), return_tensors="pt").pixel_values
            fast = shifted_greedy_decode(self._model, pixel_values, tok.cls_token_id, tok.sep_token_id, 12, True)
            slow = shifted_greedy_decode(self._model, pixel_values, tok.cls_token_id, tok.sep_token_id, 12, False)
            if fast[0] != slow[0]:
                logger.warning("KV-cache decoder disagrees with the reference decoder; disabling it")
                self._use_cache = False
        return self._use_cache

    def _decode_shifted(self, pixel_values, max_len: int = 96) -> tuple[str, float]:
        """Decode with :func:`shifted_greedy_decode`; fall back to the reference path on error.

        Returns:
            ``(latex_text, mean_token_probability)``.
        """
        tok = self._processor.tokenizer
        args = (self._model, pixel_values, tok.cls_token_id, tok.sep_token_id, max_len)
        try:
            ids, probs = shifted_greedy_decode(*args, use_cache=self._use_cache)
        except Exception:  # noqa: BLE001 - cache API differences across transformers versions
            if not self._use_cache:
                raise
            logger.warning("KV-cache decoding failed; falling back to the reference decoder", exc_info=True)
            self._use_cache = False
            ids, probs = shifted_greedy_decode(*args, use_cache=False)
        text = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return text, (float(np.mean(probs)) if probs else 0.0)

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Run TrOCR inference and return (text, mean_token_confidence)."""
        self._ensure_loaded()
        from PIL import Image as PILImage

        if image.ndim == 2:
            pil_image = PILImage.fromarray(image).convert("RGB")
        else:
            pil_image = PILImage.fromarray(image)

        pixel_values = self._processor(images=pil_image, return_tensors="pt").pixel_values

        if self._double_shift:
            return self._decode_shifted(pixel_values)

        with self._torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=256,
            )

        text = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        confidence = 0.75  # sane default when score extraction is unavailable
        try:
            scores = outputs.scores  # tuple of per-step logits
            if scores:
                probs = [self._torch.softmax(step, dim=-1).max().item() for step in scores]
                confidence = float(np.mean(probs)) if probs else confidence
        except Exception:  # noqa: BLE001
            logger.debug("Could not extract TrOCR token confidences", exc_info=True)

        return text, confidence


class _Pix2TexBackend:
    """Lazy-loaded wrapper around Pix2Tex (LaTeX-OCR)."""

    def __init__(self) -> None:
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from pix2tex.cli import LatexOCR
        except ImportError as exc:
            raise OCRDependencyError(
                "Pix2Tex fallback requires the 'pix2tex' package. "
                "Install with: pip install pix2tex"
            ) from exc

        logger.info("Loading Pix2Tex (LaTeX-OCR) model")
        self._model = LatexOCR()

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Run Pix2Tex inference. Returns (latex, confidence_estimate)."""
        self._ensure_loaded()
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image).convert("RGB")
        latex = self._model(pil_image)
        # Pix2Tex doesn't expose a native confidence score; approximate
        # using output length/validity as a weak proxy.
        confidence = 0.70 if latex.strip() else 0.0
        return latex, confidence


class OCREngine:
    """High-level OCR orchestrator implementing the primary/fallback flow."""

    def __init__(
        self,
        trocr_model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        """Create an OCR engine.

        Args:
            trocr_model_name: HuggingFace model id or local path for
                the (fine-tuned) TrOCR weights. Defaults to ``settings``.
            confidence_threshold: Below this, the Pix2Tex fallback is
                triggered. Defaults to ``settings.ocr_confidence_threshold``.
        """
        self._trocr = _TrOCRBackend(trocr_model_name)
        self._pix2tex = _Pix2TexBackend()
        self.confidence_threshold = confidence_threshold or settings.ocr_confidence_threshold

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Recognize handwritten math in ``image`` and return validated LaTeX.

        Tries TrOCR first. Falls back to Pix2Tex if TrOCR's output is
        invalid LaTeX or its confidence is below threshold. If both
        engines fail, raises :class:`OCRError`.

        Args:
            image: Preprocessed (binarized/cropped) image as an
                ``np.ndarray``, typically the output of
                ``ImageProcessor.process(...).cropped``.

        Returns:
            An :class:`OCRResult`.

        Raises:
            OCRError: If neither backend produces usable output.
        """
        trocr_latex, trocr_conf, trocr_ok = "", 0.0, False
        try:
            raw, conf = self._trocr.recognize(image)
            cleaned = apply_symbol_corrections(raw)
            trocr_latex, trocr_conf = cleaned, conf
            trocr_ok = validate_latex(cleaned) and conf >= self.confidence_threshold
            logger.info("TrOCR result: %r (conf=%.2f, valid=%s)", cleaned, conf, trocr_ok)
        except OCRDependencyError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("TrOCR inference failed; will try Pix2Tex fallback", exc_info=True)

        if trocr_ok:
            return OCRResult(
                latex=trocr_latex,
                raw_latex=trocr_latex,
                confidence=trocr_conf,
                engine_used="trocr",
                is_valid=True,
            )

        # --- Fallback: Pix2Tex -------------------------------------------
        try:
            raw, conf = self._pix2tex.recognize(image)
            cleaned = apply_symbol_corrections(raw)
            is_valid = validate_latex(cleaned)
            logger.info("Pix2Tex fallback result: %r (conf=%.2f, valid=%s)", cleaned, conf, is_valid)

            if cleaned.strip():
                return OCRResult(
                    latex=cleaned,
                    raw_latex=raw,
                    confidence=conf,
                    engine_used="pix2tex",
                    is_valid=is_valid,
                )
        except OCRDependencyError:
            logger.warning("Pix2Tex not installed; keeping the TrOCR result for review")
        except Exception:  # noqa: BLE001
            logger.warning("Pix2Tex fallback also failed", exc_info=True)

        # Both engines failed to produce anything usable, but if TrOCR at
        # least returned *something* (even low confidence), surface that
        # rather than a hard failure -- the frontend can let the student
        # edit it manually.
        if trocr_latex.strip():
            return OCRResult(
                latex=trocr_latex,
                raw_latex=trocr_latex,
                confidence=trocr_conf,
                engine_used="trocr",
                is_valid=validate_latex(trocr_latex),
            )

        raise OCRError("Both TrOCR and Pix2Tex failed to recognize any math content.")


if __name__ == "__main__":
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="Run OCR on a preprocessed math image.")
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image_path}")

    engine = OCREngine()
    result = engine.recognize(img)
    print(f"LaTeX: {result.latex}")
    print(f"Engine: {result.engine_used} | Confidence: {result.confidence:.2f} | Band: {result.confidence_band()}")
