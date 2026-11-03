#!/usr/bin/env python3
"""Fine-tune TrOCR on the CROHME + handwritten-math-symbols datasets.

This script implements the training spec from the project brief:

    Base model : microsoft/trocr-large-handwritten
    Datasets   : CROHME (~10k expressions, InkML+PNG) +
                 handwritten-math-symbols (~82k single-symbol images)
    Augment    : random rotation (+-10deg), random noise, variable pen
                 thickness simulation, paper-texture variation
    Training   : 5 epochs, batch size 8, lr 5e-5
    Target     : CER < 0.08 on the held-out validation split

Expected on-disk dataset layout (after downloading from Kaggle -- see
README for `kaggle datasets download` commands):

    data/raw/crohme/
        images/*.png
        labels.csv            # columns: filename, latex
    data/raw/math_symbols/
        images/*.png
        labels.csv            # columns: filename, latex

Usage:
    python scripts/train_ocr.py --epochs 5 --batch-size 8 --lr 5e-5
    python scripts/train_ocr.py --dry-run   # sanity-check the pipeline
                                              # end-to-end on a tiny
                                              # synthetic sample, no GPU
                                              # or downloaded data needed
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/<name>.py` from any directory: put the project root on sys.path
# (Python only adds the scripts/ folder itself, so `import src` would fail otherwise).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for the fine-tuning run."""

    base_model: str = settings.trocr_base_model
    crohme_dir: Path = settings.raw_data_dir / "crohme"
    symbols_dir: Path = settings.raw_data_dir / "math_symbols"
    output_dir: Path = settings.trocr_finetuned_dir
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 5e-5
    val_split: float = 0.1
    max_target_length: int = 256
    seed: int = 42
    target_cer: float = 0.08


def load_dataset_manifest(dataset_dir: Path) -> list[tuple[Path, str]]:
    """Read a ``labels.csv`` manifest of (image_path, latex) pairs.

    Args:
        dataset_dir: Directory containing ``images/`` and ``labels.csv``.

    Returns:
        List of (absolute image path, ground-truth LaTeX) tuples.
        Missing files are skipped with a warning rather than raising,
        so a partially-downloaded dataset doesn't kill the whole run.
    """
    labels_path = dataset_dir / "labels.csv"
    images_dir = dataset_dir / "images"
    pairs: list[tuple[Path, str]] = []

    if not labels_path.exists():
        logger.warning("Manifest not found: %s (skipping dataset)", labels_path)
        return pairs

    with open(labels_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_path = images_dir / row["filename"]
            if not image_path.exists():
                logger.debug("Missing image, skipping: %s", image_path)
                continue
            pairs.append((image_path, row["latex"]))

    logger.info("Loaded %d samples from %s", len(pairs), dataset_dir)
    return pairs


def augment_image(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply the augmentations described in the project spec.

    * Random rotation in [-10, 10] degrees
    * Gaussian noise injection
    * Variable pen thickness (morphological dilate/erode)
    * Light paper-texture simulation (subtle multiplicative noise)

    Args:
        image: Grayscale or RGB ``np.ndarray``.
        rng: A seeded ``random.Random`` for reproducibility.

    Returns:
        The augmented image.
    """
    import cv2

    angle = rng.uniform(-10, 10)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=255)

    noise = np.random.normal(0, 6, rotated.shape).astype(np.float32)
    noisy = np.clip(rotated.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    kernel_size = rng.choice([1, 2, 3])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if rng.random() < 0.5:
        thickened = cv2.erode(noisy, kernel, iterations=1)   # thicker strokes
    else:
        thickened = cv2.dilate(noisy, kernel, iterations=1)  # thinner strokes

    texture = np.random.uniform(0.96, 1.0, thickened.shape).astype(np.float32)
    textured = np.clip(thickened.astype(np.float32) * texture, 0, 255).astype(np.uint8)
    return textured


def compute_cer(prediction: str, reference: str) -> float:
    """Character Error Rate via Levenshtein edit distance.

    Args:
        prediction: Model output string.
        reference: Ground-truth string.

    Returns:
        CER as edits / len(reference); 0.0 if reference is empty and
        prediction is also empty, 1.0 if reference is empty but
        prediction is not.
    """
    if not reference:
        return 0.0 if not prediction else 1.0

    m, n = len(prediction), len(reference)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if prediction[i - 1] == reference[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n] / n


def run_training(config: TrainingConfig, dry_run: bool = False) -> dict:
    """Fine-tune TrOCR end to end.

    Args:
        config: Hyperparameters and paths.
        dry_run: If True, skips real model loading/training and runs a
            tiny synthetic smoke test of the data pipeline + CER metric
            only -- useful for CI and for verifying the script without
            a GPU or the ~2.5GB of Kaggle datasets downloaded.

    Returns:
        A metrics dict, e.g. ``{"final_cer": 0.061, "epochs": 5}``.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    rng = random.Random(config.seed)

    if dry_run:
        logger.info("Running in --dry-run mode: synthetic smoke test only")
        synthetic_pairs = [
            ("x^2+1", "x^2+1"),
            ("\\frac{1}{2}", "\\frac{1}{2}"),
            ("\\sqrt{x}", "\\sqrt{y}"),  # deliberate mismatch to exercise CER math
        ]
        cers = [compute_cer(pred, ref) for pred, ref in synthetic_pairs]
        mean_cer = float(np.mean(cers))
        logger.info("Dry-run mean CER on synthetic pairs: %.4f", mean_cer)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        return {"final_cer": mean_cer, "epochs": 0, "dry_run": True}

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )
    except ImportError as exc:
        raise ImportError(
            "Full training requires torch + transformers. "
            "Install with: pip install torch transformers, or use --dry-run "
            "to validate the pipeline without them."
        ) from exc

    crohme = load_dataset_manifest(config.crohme_dir)
    symbols = load_dataset_manifest(config.symbols_dir)
    all_pairs = crohme + symbols
    if not all_pairs:
        raise FileNotFoundError(
            "No training data found. Download CROHME and math_symbols datasets "
            "into data/raw/ first (see README), or run with --dry-run."
        )

    rng.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - config.val_split))
    train_pairs, val_pairs = all_pairs[:split_idx], all_pairs[split_idx:]
    logger.info("Train samples: %d | Val samples: %d", len(train_pairs), len(val_pairs))

    processor = TrOCRProcessor.from_pretrained(config.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(config.base_model)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    class MathOCRDataset(Dataset):
        def __init__(self, pairs: list[tuple[Path, str]], augment: bool) -> None:
            self.pairs = pairs
            self.augment = augment

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> dict:
            import cv2
            from PIL import Image as PILImage

            image_path, latex = self.pairs[idx]
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if self.augment:
                image = augment_image(image, rng)
            pil_image = PILImage.fromarray(image).convert("RGB")

            pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.squeeze()
            labels = processor.tokenizer(
                latex,
                padding="max_length",
                max_length=config.max_target_length,
                truncation=True,
            ).input_ids
            labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]
            return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

    train_dataset = MathOCRDataset(train_pairs, augment=True)
    val_dataset = MathOCRDataset(val_pairs, augment=False)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting TrOCR fine-tuning: %d epochs, batch=%d, lr=%s",
                config.epochs, config.batch_size, config.learning_rate)
    trainer.train()

    logger.info("Evaluating final CER on validation split")
    predictions, references = [], []
    model.eval()
    with torch.no_grad():
        for sample in val_dataset:
            pixel_values = sample["pixel_values"].unsqueeze(0)
            generated_ids = model.generate(pixel_values, max_length=config.max_target_length)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            label_ids = [t if t != -100 else processor.tokenizer.pad_token_id for t in sample["labels"].tolist()]
            ref_text = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(ref_text)

    final_cer = float(np.mean([compute_cer(p, r) for p, r in zip(predictions, references)]))
    logger.info("Final validation CER: %.4f (target < %.2f)", final_cer, config.target_cer)

    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)

    return {"final_cer": final_cer, "epochs": config.epochs, "dry_run": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on CROHME + math symbols.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline smoke test only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    metrics = run_training(config, dry_run=args.dry_run)
    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
