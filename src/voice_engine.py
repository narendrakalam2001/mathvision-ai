"""Voice explanation engine (Module 5).

Text-to-speech for reading explanations aloud, and speech-to-text for
students asking follow-up doubts by voice.

    TTS: gTTS (Google TTS, free, always available with just an API-less
         HTTP call) as the basic engine; ElevenLabs as a more natural-
         sounding upgrade when an API key is configured.
    STT: OpenAI Whisper (local model, no API key needed) transcribes
         voice questions like "Main step 3 nahi samjha".

Also implements the Hindi math-term phrasing table from the product
spec (e.g. "x squared" -> "x ka varg") so TTS output sounds natural to
Hindi-speaking students rather than reading raw math notation.

Both backends are optional at import time (lazy-loaded), matching the
pattern used in :mod:`src.ocr_engine`: importing this module never
fails even if ``gtts``/``elevenlabs``/``openai-whisper`` aren't
installed; only calling the relevant method does.

Example:
    >>> from src.voice_engine import VoiceEngine, VoiceLanguage
    >>> engine = VoiceEngine()
    >>> phrased = engine.mathify_text("x squared plus 3 equals 7", VoiceLanguage.HINGLISH)
    >>> "x ka varg" in phrased
    True
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config import settings
from src.math_text import speak_math

logger = logging.getLogger(__name__)


class VoiceEngineError(Exception):
    """Raised when TTS/STT generation fails on every available backend."""


class VoiceLanguage(str, Enum):
    """Voice language options from the product spec."""

    HINDI = "hindi"
    ENGLISH = "english"
    HINGLISH = "hinglish"


class VoiceSpeed(str, Enum):
    """Playback speed options from the product spec."""

    SLOW = "slow"      # 0.8x
    NORMAL = "normal"  # 1.0x
    FAST = "fast"      # 1.3x


class VoiceGender(str, Enum):
    MALE = "male"
    FEMALE = "female"


_SPEED_MULTIPLIERS = {VoiceSpeed.SLOW: 0.8, VoiceSpeed.NORMAL: 1.0, VoiceSpeed.FAST: 1.3}

# Hindi math-term phrasing table, exactly as specified in the product brief.
# Applied as whole-word, case-insensitive substitutions so we don't mangle
# unrelated text (e.g. "plus" inside another word).
_MATH_TERM_TABLE: list[tuple[str, str]] = [
    (r"\bx\s*squared\b", "x ka varg"),
    (r"\by\s*squared\b", "y ka varg"),
    (r"\bequals\b", "barabar"),
    (r"\bplus\b", "jod"),
    (r"\bminus\b", "ghata"),
    (r"\bdivided by\b", "bhag"),
    (r"\bmultiplied by\b", "guna"),
    (r"\bsquare root\b", "vargmool"),
    (r"\bcube root\b", "ghanmool"),
    (r"\bpercent\b", "pratishat"),
    (r"\btriangle\b", "trikon"),
    (r"\bcircle\b", "vrutt"),
]


def apply_hindi_math_terms(text: str) -> str:
    """Replace English math phrasing with the Hindi terms students expect.

    Args:
        text: Explanation text (typically English or mixed).

    Returns:
        Text with math terms swapped for their Hindi equivalents,
        matching the product spec's phrasing table.
    """
    result = text
    for pattern, replacement in _MATH_TERM_TABLE:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


@dataclass
class AudioResult:
    """Output of a TTS call.

    Attributes:
        audio_bytes: Raw MP3 bytes, ready for ``st.audio`` / an HTTP
            response with ``Content-Type: audio/mpeg``.
        engine_used: ``"gtts"`` or ``"elevenlabs"``.
        language: The language the audio was synthesized in.
    """

    audio_bytes: bytes
    engine_used: str
    language: VoiceLanguage


@dataclass
class TranscriptionResult:
    """Output of an STT call.

    Attributes:
        text: The transcribed text.
        detected_language: Language code Whisper detected (e.g. "hi", "en").
        confidence: Approximate confidence, when exposed by the backend.
    """

    text: str
    detected_language: str
    confidence: Optional[float] = None


class VoiceEngine:
    """Text-to-speech and speech-to-text orchestrator for MathBot."""

    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        default_language: Optional[VoiceLanguage] = None,
    ) -> None:
        """Create a voice engine.

        Args:
            elevenlabs_api_key: Overrides ``settings.elevenlabs_api_key``.
            default_language: Default TTS language if none is specified per call.
        """
        self.elevenlabs_api_key = elevenlabs_api_key or settings.elevenlabs_api_key
        self.default_language = default_language or VoiceLanguage(
            "hinglish" if settings.default_voice_language == "hi-en" else settings.default_voice_language
        )
        self._whisper_model = None

    # ------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------
    def mathify_text(self, text: str, language: VoiceLanguage) -> str:
        """Apply Hindi math-term substitution when the target language needs it.

        Args:
            text: Raw explanation text.
            language: Target voice language.

        Returns:
            Text ready for TTS synthesis (math symbols spelled out as words).
        """
        spoken = speak_math(text)            # "x^2 = 4" -> "x squared equals 4" (TTS reads symbols badly)
        if language in (VoiceLanguage.HINDI, VoiceLanguage.HINGLISH):
            return apply_hindi_math_terms(spoken)
        return spoken

    # ------------------------------------------------------------------
    # Text-to-Speech
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        language: Optional[VoiceLanguage] = None,
        speed: VoiceSpeed = VoiceSpeed.NORMAL,
        gender: VoiceGender = VoiceGender.FEMALE,
    ) -> AudioResult:
        """Convert text to spoken audio (MP3).

        Tries ElevenLabs first (more natural) if configured, otherwise
        uses gTTS (always available, no key required).

        Args:
            text: The explanation text to speak.
            language: Voice language; defaults to ``self.default_language``.
            speed: Playback speed setting.
            gender: Preferred voice gender (ElevenLabs only; gTTS has one voice per language).

        Returns:
            An :class:`AudioResult`.

        Raises:
            VoiceEngineError: If both backends fail.
        """
        language = language or self.default_language
        phrased_text = self.mathify_text(text, language)

        if self.elevenlabs_api_key:
            try:
                return self._synthesize_elevenlabs(phrased_text, language, speed, gender)
            except Exception:  # noqa: BLE001
                logger.warning("ElevenLabs TTS failed, falling back to gTTS", exc_info=True)

        try:
            return self._synthesize_gtts(phrased_text, language, speed)
        except Exception as exc:  # noqa: BLE001
            raise VoiceEngineError(f"All TTS backends failed: {exc}") from exc

    def _synthesize_gtts(self, text: str, language: VoiceLanguage, speed: VoiceSpeed) -> AudioResult:
        """Synthesize speech using gTTS (Google Translate TTS, free)."""
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise VoiceEngineError(
                "gTTS backend requires the 'gtts' package. Install with: pip install gTTS"
            ) from exc

        lang_code = {VoiceLanguage.HINDI: "hi", VoiceLanguage.ENGLISH: "en", VoiceLanguage.HINGLISH: "hi"}[
            language
        ]
        slow = speed == VoiceSpeed.SLOW

        buffer = io.BytesIO()
        tts = gTTS(text=text, lang=lang_code, slow=slow)
        tts.write_to_fp(buffer)
        buffer.seek(0)

        return AudioResult(audio_bytes=buffer.read(), engine_used="gtts", language=language)

    def _synthesize_elevenlabs(
        self, text: str, language: VoiceLanguage, speed: VoiceSpeed, gender: VoiceGender
    ) -> AudioResult:
        """Synthesize speech using the ElevenLabs API (natural-sounding voices)."""
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise VoiceEngineError(
                "ElevenLabs backend requires the 'elevenlabs' package. "
                "Install with: pip install elevenlabs"
            ) from exc

        client = ElevenLabs(api_key=self.elevenlabs_api_key)
        # Voice IDs would normally be looked up via client.voices.get_all();
        # using named presets here keeps this runnable without an extra API round-trip.
        voice_id = "21m00Tcm4TlvDq8ikWAM" if gender == VoiceGender.FEMALE else "TxGEqnHWrfWFTfGW9XjX"

        audio_stream = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings={"speed": _SPEED_MULTIPLIERS[speed]},
        )
        audio_bytes = b"".join(chunk for chunk in audio_stream)
        return AudioResult(audio_bytes=audio_bytes, engine_used="elevenlabs", language=language)

    # ------------------------------------------------------------------
    # Speech-to-Text
    # ------------------------------------------------------------------
    def _ensure_whisper_loaded(self, model_size: str = "base") -> None:
        if self._whisper_model is not None:
            return
        try:
            import whisper
        except ImportError as exc:
            raise VoiceEngineError(
                "STT requires 'openai-whisper'. Install with: pip install openai-whisper"
            ) from exc
        logger.info("Loading Whisper model: %s", model_size)
        self._whisper_model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str, model_size: str = "base") -> TranscriptionResult:
        """Transcribe a voice question (e.g. "Main step 3 nahi samjha").

        Args:
            audio_path: Path to an audio file (wav/mp3/m4a — anything
                ffmpeg can decode, since Whisper shells out to ffmpeg).
            model_size: Whisper model size ("tiny", "base", "small", ...).
                "base" balances speed and accuracy well for short doubts.

        Returns:
            A :class:`TranscriptionResult`.

        Raises:
            VoiceEngineError: If Whisper isn't installed or transcription fails.
        """
        self._ensure_whisper_loaded(model_size)
        try:
            result = self._whisper_model.transcribe(audio_path)
            return TranscriptionResult(
                text=result.get("text", "").strip(),
                detected_language=result.get("language", "unknown"),
            )
        except Exception as exc:  # noqa: BLE001
            raise VoiceEngineError(f"Whisper transcription failed: {exc}") from exc


if __name__ == "__main__":
    engine = VoiceEngine()
    sample = "x squared plus 3 equals 7, so x squared equals 4"
    print("Original:", sample)
    print("Mathified (Hinglish):", engine.mathify_text(sample, VoiceLanguage.HINGLISH))

    try:
        audio = engine.synthesize(sample, language=VoiceLanguage.HINGLISH)
        print(f"Synthesized {len(audio.audio_bytes)} bytes via {audio.engine_used}")
    except VoiceEngineError as e:
        print(f"TTS unavailable in this environment: {e}")
