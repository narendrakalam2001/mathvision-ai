"""Benchmark the OCR decoders on YOUR machine: reference vs KV-cache (and optional int8).

Usage (from the project root):
    python scripts/benchmark_ocr.py sample_problems/photo2.jpeg --preprocess
    python scripts/benchmark_ocr.py line.png --repeat 5
    python scripts/benchmark_ocr.py line.png --quantize        # also try int8 (accuracy not validated!)

Prints, per image: model-load time, encoder time, decoder time for both
implementations, the speed-up, and whether the two produced IDENTICAL text.
If "identical" is ever False, keep ``OCR_KV_CACHE=false``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _timed(fn, repeat: int):
    """Run ``fn`` ``repeat`` times; return (last_result, best_seconds)."""
    best, result = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Benchmark OCR decoding speed.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--preprocess", action="store_true", help="run ImageProcessor first (phone photos)")
    parser.add_argument("--repeat", type=int, default=3, help="timing repeats (best of N)")
    parser.add_argument("--quantize", action="store_true", help="enable int8 dynamic quantization")
    args = parser.parse_args()
    if args.quantize:
        os.environ["OCR_QUANTIZE"] = "true"

    import cv2
    from PIL import Image as PILImage

    from src.layout_analyzer import LayoutAnalyzer
    from src.ocr_engine import _TrOCRBackend, shifted_greedy_decode

    backend = _TrOCRBackend()
    t0 = time.perf_counter()
    backend._ensure_loaded()  # noqa: SLF001
    print(f"Model: {backend.model_name}\nLoad time: {time.perf_counter() - t0:.1f}s | "
          f"double_shift={backend._double_shift} | quantized={args.quantize}")  # noqa: SLF001
    if not backend._double_shift:  # noqa: SLF001
        sys.exit("This benchmark targets the shifted-label checkpoint (needs decoding.json).")

    tok = backend._processor.tokenizer  # noqa: SLF001
    analyzer, processor = LayoutAnalyzer(), None
    if args.preprocess:
        from src.image_processor import ImageProcessor

        processor = ImageProcessor()

    total_ref = total_fast = 0.0
    for path in args.images:
        if processor is not None:
            img = processor.process(str(path)).cropped
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] cannot read {path}")
            continue
        lines = analyzer.analyze(img)
        if not lines:
            print(f"[skip] no ink found in {path}")
            continue
        pixels = backend._processor(images=PILImage.fromarray(lines[0].image).convert("RGB"),  # noqa: SLF001
                                    return_tensors="pt").pixel_values
        model = backend._model  # noqa: SLF001
        import torch

        with torch.inference_mode():
            enc, enc_s = _timed(lambda: model.encoder(pixel_values=pixels).last_hidden_state, args.repeat)
        decode = lambda cache: shifted_greedy_decode(  # noqa: E731
            model, pixels, tok.cls_token_id, tok.sep_token_id, 96, cache, encoder_hidden=enc)
        (ref_ids, _), ref_s = _timed(lambda: decode(False), args.repeat)
        (fast_ids, _), fast_s = _timed(lambda: decode(True), args.repeat)
        ref_text = tok.decode(ref_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        fast_text = tok.decode(fast_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        total_ref += ref_s
        total_fast += fast_s
        print(f"\n{path.name}  ({len(ref_ids)} tokens)\n  text      : {fast_text}\n"
              f"  encoder   : {enc_s:5.2f}s (same for both)\n"
              f"  decoder   : reference {ref_s:5.2f}s | KV-cache {fast_s:5.2f}s  -> {ref_s / fast_s:4.1f}x faster\n"
              f"  per line  : {enc_s + ref_s:5.2f}s -> {enc_s + fast_s:5.2f}s  ({(enc_s + ref_s) / (enc_s + fast_s):.1f}x end-to-end)\n"
              f"  identical : {ref_text == fast_text}")

    if total_fast:
        print(f"\nDecoder total: reference {total_ref:.2f}s vs cached {total_fast:.2f}s ({total_ref / total_fast:.1f}x)")


if __name__ == "__main__":
    main()
