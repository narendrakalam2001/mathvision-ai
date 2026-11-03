"""Central configuration for MathVision AI.

All environment-dependent values (API keys, paths, thresholds, feature
flags) are defined here so the rest of the codebase never reads
``os.environ`` directly. Values are loaded from a ``.env`` file (via
python-dotenv) when present, falling back to sensible defaults so the
project still runs locally without any secrets configured.

Usage:
    from src.config import settings

    print(settings.PROJECT_ROOT)
    if settings.GEMINI_API_KEY:
        ...
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is an optional convenience
    pass


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean-like environment variable.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is unset.

    Returns:
        The parsed boolean value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    """Parse an integer environment variable, falling back on failure."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid integer for %s=%r, using default=%s", name, raw, default
        )
        return default


def _get_float(name: str, default: float) -> float:
    """Parse a float environment variable, falling back on failure."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid float for %s=%r, using default=%s", name, raw, default
        )
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Grouped by concern: paths, API keys, model config, OCR thresholds,
    solver behaviour, voice settings and logging.
    """

    # --- Paths -----------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    ncert_problems_dir: Path = PROJECT_ROOT / "data" / "ncert_problems"
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    student_profile_dir: Path = field(
        default_factory=lambda: Path(os.getenv("STUDENT_PROFILE_DIR") or PROJECT_ROOT / "data" / "student_profiles")
    )
    models_dir: Path = PROJECT_ROOT / "models"
    trocr_finetuned_dir: Path = PROJECT_ROOT / "models" / "trocr_finetuned"
    symbol_classifier_dir: Path = PROJECT_ROOT / "models" / "symbol_classifier"
    chromadb_dir: Path = PROJECT_ROOT / "chromadb_store"
    sample_problems_dir: Path = PROJECT_ROOT / "sample_problems"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # --- API keys (never hard-code secrets; read from environment) -
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    wolframalpha_app_id: str = field(default_factory=lambda: os.getenv("WOLFRAMALPHA_APP_ID", ""))
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    huggingface_token: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", ""))

    # --- OCR model configuration -------------------------------------
    trocr_base_model: str = field(
        default_factory=lambda: os.getenv("TROCR_BASE_MODEL", "microsoft/trocr-large-handwritten")
    )
    pix2tex_model: str = field(
        default_factory=lambda: os.getenv("PIX2TEX_MODEL", "lukas-blecher/LaTeX-OCR")
    )
    # --- OCR speed knobs ------------------------------------------------
    ocr_kv_cache: bool = field(default_factory=lambda: _get_bool("OCR_KV_CACHE", True))
    ocr_quantize: bool = field(default_factory=lambda: _get_bool("OCR_QUANTIZE", False))
    ocr_threads: int = field(default_factory=lambda: _get_int("OCR_THREADS", 0))   # 0 = all cores
    preload_ocr: bool = field(default_factory=lambda: _get_bool("PRELOAD_OCR", False))

    ocr_confidence_threshold: float = field(
        default_factory=lambda: _get_float("OCR_CONFIDENCE_THRESHOLD", 0.60)
    )
    ocr_low_confidence_threshold: float = field(
        default_factory=lambda: _get_float("OCR_LOW_CONFIDENCE_THRESHOLD", 0.35)
    )

    # --- Image quality thresholds --------------------------------------
    blur_variance_threshold: float = field(
        default_factory=lambda: _get_float("BLUR_VARIANCE_THRESHOLD", 30.0)
    )
    min_brightness: float = field(default_factory=lambda: _get_float("MIN_BRIGHTNESS", 40.0))
    max_brightness: float = field(default_factory=lambda: _get_float("MAX_BRIGHTNESS", 235.0))
    target_image_size: int = field(default_factory=lambda: _get_int("TARGET_IMAGE_SIZE", 1024))

    # --- Solver behaviour ------------------------------------------------
    wolframalpha_monthly_quota: int = field(
        default_factory=lambda: _get_int("WOLFRAMALPHA_MONTHLY_QUOTA", 2000)
    )
    solver_timeout_seconds: int = field(default_factory=lambda: _get_int("SOLVER_TIMEOUT_SECONDS", 10))

    # --- Voice engine -----------------------------------------------------
    default_voice_language: str = field(
        default_factory=lambda: os.getenv("DEFAULT_VOICE_LANGUAGE", "hi-en")
    )
    default_voice_speed: float = field(default_factory=lambda: _get_float("DEFAULT_VOICE_SPEED", 1.0))

    # --- Retry / networking ----------------------------------------------
    api_max_retries: int = field(default_factory=lambda: _get_int("API_MAX_RETRIES", 3))
    api_backoff_seconds: float = field(default_factory=lambda: _get_float("API_BACKOFF_SECONDS", 1.5))

    # --- Feature flags -----------------------------------------------------
    enable_wolfram_fallback: bool = field(
        default_factory=lambda: _get_bool("ENABLE_WOLFRAM_FALLBACK", True)
    )
    enable_gemini_explanations: bool = field(
        default_factory=lambda: _get_bool("ENABLE_GEMINI_EXPLANATIONS", True)
    )

    # --- Logging ----------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    def ensure_directories(self) -> None:
        """Create all data/model/log directories if they do not exist."""
        for directory in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.ncert_problems_dir,
            self.knowledge_base_dir,
            self.models_dir,
            self.trocr_finetuned_dir,
            self.symbol_classifier_dir,
            self.chromadb_dir,
            self.sample_problems_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def configure_logging(settings_obj: "Settings") -> None:
    """Configure root logging once for the whole application.

    Args:
        settings_obj: The active :class:`Settings` instance.
    """
    settings_obj.ensure_directories()
    log_file = settings_obj.logs_dir / "mathvision.log"

    logging.basicConfig(
        level=getattr(logging, settings_obj.log_level.upper(), logging.INFO),
        format=settings_obj.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


# Module-level singleton used throughout the codebase.
settings = Settings()
configure_logging(settings)
