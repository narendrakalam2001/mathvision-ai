"""Tests for the weights-free Gemini OCR backend (no network: Gemini is mocked)."""

from __future__ import annotations

import numpy as np
import pytest

from services.math_service import MathService, MathServiceError
from src import gemini_ocr
from src.gemini_client import GeminiError
from src.gemini_ocr import GeminiOCREngine, clean_model_output
from src.ocr_engine import OCRError


@pytest.fixture()
def image() -> np.ndarray:
    img = np.full((60, 200), 255, dtype=np.uint8)
    img[20:30, 20:180] = 0
    return img


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("x^{2} - 5x + 6 = 0", "x^{2} - 5x + 6 = 0"),
        ("```latex\nx^2 + 1 = 0\n```", "x^2 + 1 = 0"),
        ("$x^2 = 4$", "x^2 = 4"),
        ("$$\\frac{a}{b}$$", "\\frac{a}{b}"),
        ("\\[ y = 2x \\]", "y = 2x"),
        ("LaTeX: 3x + 2 = 11", "3x + 2 = 11"),
    ],
)
def test_clean_model_output(raw: str, expected: str) -> None:
    assert clean_model_output(raw) == expected


def test_recognize_valid_latex(monkeypatch: pytest.MonkeyPatch, image: np.ndarray) -> None:
    monkeypatch.setattr(gemini_ocr, "generate_from_image", lambda *a, **k: "```latex\nx^{2} - 5x + 6 = 0\n```")
    result = GeminiOCREngine().recognize(image)
    assert result.latex == "x^{2} - 5x + 6 = 0"
    assert result.engine_used == "gemini"
    assert result.is_valid and not result.needs_review


def test_recognize_unbalanced_needs_review(monkeypatch: pytest.MonkeyPatch, image: np.ndarray) -> None:
    monkeypatch.setattr(gemini_ocr, "generate_from_image", lambda *a, **k: "\\frac{1}{2")
    result = GeminiOCREngine().recognize(image)
    assert not result.is_valid and result.needs_review


def test_unreadable_raises(monkeypatch: pytest.MonkeyPatch, image: np.ndarray) -> None:
    monkeypatch.setattr(gemini_ocr, "generate_from_image", lambda *a, **k: "UNREADABLE")
    with pytest.raises(OCRError):
        GeminiOCREngine().recognize(image)


def test_gemini_failure_becomes_ocr_error(monkeypatch: pytest.MonkeyPatch, image: np.ndarray) -> None:
    def boom(*a, **k):
        raise GeminiError("quota")

    monkeypatch.setattr(gemini_ocr, "generate_from_image", boom)
    with pytest.raises(OCRError):
        GeminiOCREngine().recognize(image)


def test_empty_image_raises() -> None:
    with pytest.raises(OCRError):
        GeminiOCREngine().recognize(np.zeros((0, 0), dtype=np.uint8))


def test_backend_none_without_key_or_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = MathService()
    monkeypatch.setattr(svc, "_local_ocr_available", lambda: False)
    monkeypatch.setattr("src.config.settings", type("S", (), {"gemini_api_key": ""})())
    assert svc.ocr_backend_name() == "none"
    with pytest.raises(MathServiceError):
        svc._get_ocr_engine()


def test_backend_gemini_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = MathService()
    monkeypatch.setattr(svc, "_local_ocr_available", lambda: False)
    monkeypatch.setattr("src.config.settings", type("S", (), {"gemini_api_key": "k"})())
    assert svc.ocr_backend_name() == "gemini"
    assert isinstance(svc._get_ocr_engine(), GeminiOCREngine)


def test_solve_image_uses_gemini_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: photo -> (mocked) Gemini OCR -> SymPy solution."""
    import cv2

    svc = MathService()
    monkeypatch.setattr(svc, "_local_ocr_available", lambda: False)
    monkeypatch.setattr("src.config.settings", type("S", (), {"gemini_api_key": "k"})())
    monkeypatch.setattr(gemini_ocr, "generate_from_image", lambda *a, **k: "x^{2} - 5x + 6 = 0")

    img = np.full((300, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "x^2 - 5x + 6 = 0", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    result = svc.solve_image(buf.tobytes())
    if result.get("retake_recommended"):
        pytest.skip("synthetic image rejected by quality check")
    assert result["ocr"]["engine_used"] == "gemini"
    assert "2" in result["answer"] and "3" in result["answer"]
