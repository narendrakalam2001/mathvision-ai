"""Quick sanity check: run the fine-tuned TrOCR on one or more images.

Usage (from the project root):
    python scripts/check_ocr.py sample_problems/test1.png
    python scripts/check_ocr.py sample_problems/photo1.jpeg --preprocess
    python scripts/check_ocr.py some.png --whole        # skip line splitting

Pipeline: [optional ImageProcessor] -> LayoutAnalyzer (one crop per line,
normalised to the 384x96 training look) -> TrOCR (shifted decoding).
Every normalised line is saved as ``<name>_line<N>.png`` next to the input so
you can see exactly what the model received.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Run TrOCR on the given images and print per-line results."""
    parser = argparse.ArgumentParser(description="Run fine-tuned TrOCR on images.")
    parser.add_argument("images", nargs="+", type=Path, help="real image paths")
    parser.add_argument("--preprocess", action="store_true", help="run ImageProcessor first (phone photos)")
    parser.add_argument("--whole", action="store_true", help="feed the whole image to OCR (no line splitting)")
    args = parser.parse_args()

    import cv2

    from src.layout_analyzer import LayoutAnalyzer
    from src.ocr_engine import OCREngine

    backend = OCREngine()._trocr  # noqa: SLF001 - direct backend access for the sanity check
    print(f"Model: {backend.model_name}")
    analyzer = LayoutAnalyzer()
    processor = None
    if args.preprocess:
        from src.image_processor import ImageProcessor

        processor = ImageProcessor()

    for path in args.images:
        if not path.is_file():
            print(f"[skip] file not found: {path}")
            continue
        if processor is not None:
            result = processor.process(str(path))
            img = result.cropped
            print(f"{path.name} | quality: {result.quality.message}")
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[skip] cannot decode image: {path}")
                continue
            print(path.name)

        if args.whole:
            crops = [(0, img)]
        else:
            crops = [(ln.index, ln.image) for ln in analyzer.analyze(img)]
            print(f"  {len(crops)} line(s) detected")

        for idx, crop in crops:
            out = path.with_name(f"{path.stem}_line{idx + 1}.png")
            cv2.imwrite(str(out), crop)
            t0 = time.time()
            text, conf = backend.recognize(crop)
            print(f"  line {idx + 1}: {text}\n          conf {conf:.2f} | {time.time() - t0:.1f}s | saved {out.name}")
        print()

    print(f"double_shift decoding active: {backend._double_shift}")  # noqa: SLF001


if __name__ == "__main__":
    main()
