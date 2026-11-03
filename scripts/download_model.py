"""Download the fine-tuned TrOCR checkpoint from the HuggingFace Hub.

Usage (from the project root):
    python scripts/download_model.py                       # uses HF_TOKEN env var / cached login
    python scripts/download_model.py --token hf_xxx        # explicit read token
    python scripts/download_model.py --repo user/name --out models/trocr_finetuned

The weights land in ``models/trocr_finetuned/`` where ``OCREngine`` picks them
up automatically (it also reads ``decoding.json`` to enable the custom
decoding loop required by this checkpoint).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "narendrakalam2001/trocr-math-ep4"
DEFAULT_OUT = PROJECT_ROOT / "models" / "trocr_finetuned"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("download_model")


def download(repo_id: str, out_dir: Path, token: Optional[str]) -> Path:
    """Download ``repo_id`` into ``out_dir`` and verify the essential files.

    Args:
        repo_id: HuggingFace repo id, e.g. ``user/trocr-math-ep4``.
        out_dir: Local destination directory.
        token: HF read token (needed for private repos); ``None`` uses the cached login.

    Returns:
        The destination directory.

    Raises:
        SystemExit: If huggingface_hub is missing or essential files are absent.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s (about 2.2 GB)", repo_id, out_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        token=token,
        ignore_patterns=["training_args.bin", ".gitattributes"],
    )

    missing = [f for f in ("config.json", "model.safetensors", "decoding.json", "tokenizer.json")
               if not (out_dir / f).is_file()]
    if missing:
        sys.exit(f"Download incomplete, missing: {missing}")
    logger.info("Done. Files: %s", sorted(p.name for p in out_dir.iterdir()))
    return out_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download fine-tuned TrOCR weights from HuggingFace.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"))
    args = parser.parse_args()
    download(args.repo, args.out, args.token)


if __name__ == "__main__":
    main()
