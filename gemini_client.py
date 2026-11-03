"""Thin, testable wrapper around the Gemini API (``google-genai`` SDK).

Every Gemini call in the project (explanations, solver fallback, answer
verification, MathBot) goes through :func:`generate_text`, so the model name,
retries and error messages live in one place.

* The model comes from ``GEMINI_MODEL`` (default ``gemini-3.1-flash-lite``).
  Google retires models regularly -- ``gemini-1.5-flash`` no longer exists --
  so changing the model is a one-line ``.env`` edit, not a code change.
* Transient errors (rate limits, 5xx, network) are retried with back-off;
  "model not found" / "bad API key" are not retried and raise a clear message.
* :func:`generate_from_image` sends one image + a prompt (used by the light,
  weights-free handwriting OCR in :mod:`src.gemini_ocr`).
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class GeminiError(RuntimeError):
    """Raised when a Gemini call cannot be completed."""


@lru_cache(maxsize=4)
def _client(api_key: str) -> Any:
    """Return a cached ``google.genai.Client`` for ``api_key``."""
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiError("Gemini needs the 'google-genai' package: pip install google-genai") from exc
    return genai.Client(api_key=api_key)


def _status_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def generate_text(
    prompt: str,
    *,
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.4,
    retries: Optional[int] = None,
) -> str:
    """Generate text with Gemini.

    Args:
        prompt: The user prompt.
        system: Optional system instruction (persona / rules).
        api_key: Overrides ``settings.gemini_api_key``.
        model: Overrides ``settings.gemini_model``.
        temperature: Sampling temperature.
        retries: Attempts for transient failures (default ``settings.api_max_retries``).

    Returns:
        The response text, stripped (never empty).

    Raises:
        GeminiError: Missing key, missing package, unknown model, bad key,
            empty reply, or repeated transient failures.
    """
    key = api_key or settings.gemini_api_key
    if not key:
        raise GeminiError("GEMINI_API_KEY is not configured.")
    model_name = model or settings.gemini_model
    attempts = max(1, retries if retries is not None else settings.api_max_retries)

    try:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    except ImportError as exc:
        raise GeminiError("Gemini needs the 'google-genai' package: pip install google-genai") from exc

    client = _client(key)
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt, config=config)
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise GeminiError("Gemini returned an empty response.")
            return text
        except GeminiError as exc:
            last = exc
        except Exception as exc:  # noqa: BLE001 - SDK raises several error types
            last = exc
            code = _status_code(exc)
            if code == 404:
                raise GeminiError(
                    f"Gemini model '{model_name}' was not found (it may have been retired). "
                    "Set GEMINI_MODEL in .env to a current model, e.g. gemini-3.1-flash-lite."
                ) from exc
            if code in (400, 401, 403):
                raise GeminiError(f"Gemini rejected the request (HTTP {code}): check GEMINI_API_KEY and GEMINI_MODEL.") from exc
        if attempt < attempts - 1:
            time.sleep(settings.api_backoff_seconds * (attempt + 1))
    raise GeminiError(f"Gemini call failed after {attempts} attempt(s): {last}") from last


def generate_from_image(
    image_bytes: bytes,
    prompt: str,
    *,
    mime_type: str = "image/png",
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    retries: Optional[int] = None,
) -> str:
    """Send one image plus a text prompt to Gemini and return the text reply.

    Args:
        image_bytes: Encoded image bytes (PNG/JPEG/WEBP).
        prompt: Instruction for the model (e.g. "transcribe this to LaTeX").
        mime_type: MIME type of ``image_bytes``.
        system: Optional system instruction.
        api_key: Overrides ``settings.gemini_api_key``.
        model: Overrides ``settings.gemini_model``.
        temperature: Sampling temperature (0 = deterministic, best for OCR).
        retries: Attempts for transient failures (default ``settings.api_max_retries``).

    Returns:
        The response text, stripped (never empty).

    Raises:
        GeminiError: Missing key/package, unknown model, bad key, empty reply,
            or repeated transient failures.
    """
    key = api_key or settings.gemini_api_key
    if not key:
        raise GeminiError("GEMINI_API_KEY is not configured.")
    if not image_bytes:
        raise GeminiError("No image data was provided.")
    model_name = model or settings.gemini_model
    attempts = max(1, retries if retries is not None else settings.api_max_retries)

    try:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        contents = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt]
    except ImportError as exc:
        raise GeminiError("Gemini needs the 'google-genai' package: pip install google-genai") from exc

    client = _client(key)
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise GeminiError("Gemini returned an empty response.")
            return text
        except GeminiError as exc:
            last = exc
        except Exception as exc:  # noqa: BLE001 - SDK raises several error types
            last = exc
            code = _status_code(exc)
            if code == 404:
                raise GeminiError(
                    f"Gemini model '{model_name}' was not found (it may have been retired). "
                    "Set GEMINI_MODEL to a current model, e.g. gemini-3.1-flash-lite."
                ) from exc
            if code in (400, 401, 403):
                raise GeminiError(f"Gemini rejected the request (HTTP {code}): check GEMINI_API_KEY and GEMINI_MODEL.") from exc
        if attempt < attempts - 1:
            time.sleep(settings.api_backoff_seconds * (attempt + 1))
    raise GeminiError(f"Gemini image call failed after {attempts} attempt(s): {last}") from last
