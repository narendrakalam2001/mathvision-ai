#!/usr/bin/env python3
"""Convert raw, as-downloaded dataset folders into the ``images/`` +
``labels.csv`` manifest format that ``scripts/train_ocr.py`` reads.

Why this script exists: the datasets as distributed by Kaggle/TC-11 do
NOT ship as pre-rendered PNGs with a simple CSV — they come in two very
different raw shapes:

    1. CROHME-family handwriting data (CROHME_training_2011,
       CROHME_test_2011, trainData_2012_part1/2, testData_2012,
       TrainINKML_2013/{expressmatch,extension,HAMEX,KAIST,MathBrush,
       MfrDB}, TestINKML_2013, MatricesTrain2014, MatricesTest2014, ...)
       is **InkML**: XML files containing pen-stroke coordinate traces
       plus a ground-truth LaTeX annotation — not images at all. Each
       file has to be parsed and the strokes rendered into a raster
       image before OCR training can use it.

    2. The "handwritten math symbols" dataset is a **folder-per-class**
       layout: ``dataset/0/``, ``dataset/1/``, ... ``dataset/add/``,
       ``dataset/eq/``, ``dataset/x/`` etc., each containing loose
       image files for that one symbol, with the label implied by the
       folder name rather than written anywhere in a CSV.

This script normalizes both into ``data/raw/crohme/{images/,labels.csv}``
and ``data/raw/math_symbols/{images/,labels.csv}`` — exactly the paths
``scripts/train_ocr.py`` already reads by default — so **you do not
need to modify or pass any dataset path to train_ocr.py at all**. Run
this once, then run train_ocr.py exactly as documented.

Usage (Windows example, matching a D:\\ drive download):
    python scripts/prepare_datasets.py --data-root "D:\\Data_Science_Capstone_Datasets\\mathvision-ai"

Usage (Linux/Mac):
    python scripts/prepare_datasets.py --data-root /path/to/mathvision-ai

Then, unchanged:
    python scripts/train_ocr.py --epochs 5 --batch-size 8 --lr 5e-5

Options:
    --crohme-only          Only process the InkML (CROHME-family) folders
    --symbols-only         Only process the folder-per-class symbols dataset
    --symbols-subfolder    Name of the symbols folder under --data-root (default: "dataset")
    --limit N              Only process the first N InkML files (quick smoke test)
    --canvas-width / --canvas-height / --line-width   Rendering parameters
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
import re
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw

from src.config import settings

logger = logging.getLogger(__name__)

# Every top-level folder name we've seen across CROHME 2011/2012/2013/2014
# distributions. discover_inkml_files() searches recursively under each of
# these (when present under --data-root), so it transparently handles both
# flat archives (CROHME_training_2011/*.inkml) and nested ones
# (TrainINKML_2013/expressmatch/*.inkml, TrainINKML_2013/HAMEX/*.inkml, ...).
DEFAULT_INKML_FOLDER_NAMES: list[str] = [
    "CROHME_training_2011",
    "CROHME_test_2011",
    "trainData_2012_part1",
    "trainData_2012_part2",
    "testData_2012",
    "TrainINKML_2013",
    "TestINKML_2013",
    "MatricesTrain2014",
    "MatricesTest2014",
]

# Maps the "handwritten-math-symbols" Kaggle dataset's folder names to the
# LaTeX token they represent. Digits and letters map to themselves and
# don't need an entry. Add more entries here if your download uses
# different class-folder names.
SYMBOL_FOLDER_TO_LATEX: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "\\times",
    "div": "\\div",
    "eq": "=",
    "dec": ".",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


# ----------------------------------------------------------------------
# InkML parsing + rendering
# ----------------------------------------------------------------------
def _strip_ns(tag: str) -> str:
    """Strip an XML namespace prefix, e.g. '{http://.../InkML}trace' -> 'trace'."""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_inkml(path: Path) -> Optional[tuple[list[list[tuple[float, float]]], str]]:
    """Parse one InkML file into its pen-stroke traces and ground-truth LaTeX.

    Args:
        path: Path to a ``.inkml`` file.

    Returns:
        A tuple ``(strokes, truth_latex)`` where ``strokes`` is a list of
        strokes, each a list of ``(x, y)`` points, and ``truth_latex`` is
        the ground-truth annotation with surrounding ``$`` and extra
        whitespace stripped. Returns ``None`` if the file is malformed,
        has no truth annotation, or has no usable trace data (all of
        which are logged and simply skipped by the caller rather than
        aborting the whole run).
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        logger.debug("Malformed InkML, skipping: %s", path)
        return None

    root = tree.getroot()

    truth: Optional[str] = None
    for child in root:
        if _strip_ns(child.tag) == "annotation" and child.attrib.get("type") == "truth":
            truth = (child.text or "").strip()
            break
    if not truth:
        return None

    truth = truth.strip("$").strip()
    truth = re.sub(r"\s+", " ", truth)

    strokes: list[list[tuple[float, float]]] = []
    for element in root.iter():
        if _strip_ns(element.tag) != "trace":
            continue
        raw = (element.text or "").strip()
        if not raw:
            continue
        points: list[tuple[float, float]] = []
        for point_str in raw.split(","):
            parts = point_str.strip().split()
            if len(parts) < 2:
                continue
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if points:
            strokes.append(points)

    if not strokes:
        return None
    return strokes, truth


def render_strokes(
    strokes: list[list[tuple[float, float]]],
    canvas_size: tuple[int, int] = (384, 96),
    padding: int = 10,
    line_width: int = 3,
    flip_y: bool = False,
) -> Image.Image:
    """Render InkML pen strokes onto a white raster canvas (black ink).

    Strokes are scaled uniformly (preserving aspect ratio) to fit the
    canvas with the given padding, matching the "ink on paper" look
    TrOCR expects, rather than stretching the expression to fill the
    whole frame.

    Args:
        strokes: List of strokes, each a list of ``(x, y)`` points, as
            returned by :func:`parse_inkml`.
        canvas_size: ``(width, height)`` of the output image in pixels.
        padding: Empty margin (px) kept around the rendered ink.
        line_width: Stroke thickness in pixels.
        flip_y: CROHME digitizer coordinates are y-down (same as PIL's
            image coordinates), so the default (``False``) renders
            correctly without any flip. Some InkML sources instead
            record y-up (math/plotting convention); if your rendered
            debug images come out upside-down, rerun with
            ``--flip-y`` (see the CLI) rather than editing this file.

    Returns:
        A grayscale (mode "L") :class:`PIL.Image.Image`.
    """
    all_points = [p for stroke in strokes for p in stroke]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width_range = max(max_x - min_x, 1e-6)
    height_range = max(max_y - min_y, 1e-6)

    target_w, target_h = canvas_size
    avail_w, avail_h = target_w - 2 * padding, target_h - 2 * padding
    scale = min(avail_w / width_range, avail_h / height_range)

    offset_x = padding + (avail_w - width_range * scale) / 2
    offset_y = padding + (avail_h - height_range * scale) / 2

    image = Image.new("L", (target_w, target_h), color=255)
    draw = ImageDraw.Draw(image)

    for stroke in strokes:
        scaled = []
        for x, y in stroke:
            norm_y = (max_y - y) if flip_y else (y - min_y)
            scaled.append((offset_x + (x - min_x) * scale, offset_y + norm_y * scale))
        if len(scaled) == 1:
            x, y = scaled[0]
            draw.ellipse([x - line_width, y - line_width, x + line_width, y + line_width], fill=0)
        else:
            draw.line(scaled, fill=0, width=line_width, joint="curve")

    return image


def discover_inkml_files(data_root: Path, folder_names: Optional[list[str]] = None) -> list[Path]:
    """Recursively find every ``.inkml`` file under the known dataset folders.

    Args:
        data_root: The directory containing your downloaded dataset
            folders (e.g. ``D:\\Data_Science_Capstone_Datasets\\mathvision-ai``).
        folder_names: Override the default list of top-level folder
            names to search under (see :data:`DEFAULT_INKML_FOLDER_NAMES`).

    Returns:
        A de-duplicated list of ``.inkml`` file paths. Folders that
        don't exist under ``data_root`` are silently skipped, so this
        works whether you have all of CROHME 2011-2014 or just one
        year's archive downloaded.
    """
    names = folder_names or DEFAULT_INKML_FOLDER_NAMES
    files: list[Path] = []
    seen: set[Path] = set()

    for name in names:
        folder = data_root / name
        if not folder.exists():
            logger.debug("Folder not present, skipping: %s", folder)
            continue
        for path in folder.rglob("*.inkml"):
            if path not in seen:
                seen.add(path)
                files.append(path)

    return files


def build_crohme_manifest(
    data_root: Path,
    output_dir: Path,
    canvas_size: tuple[int, int] = (384, 96),
    line_width: int = 3,
    limit: Optional[int] = None,
    flip_y: bool = False,
) -> int:
    """Parse + render every discoverable InkML file into a train_ocr.py-ready manifest.

    Args:
        data_root: Root folder containing your downloaded CROHME-family folders.
        output_dir: Where to write ``images/`` and ``labels.csv``
            (pass ``settings.raw_data_dir / "crohme"`` to match
            ``train_ocr.py``'s default).
        canvas_size: Rendered image size, ``(width, height)``.
        line_width: Stroke thickness in the rendered image.
        limit: Optional cap on how many InkML files to process (useful
            for a fast smoke test before committing to the full run).
        flip_y: See :func:`render_strokes`. Leave ``False`` unless you
            check a few images in ``output_dir/images/`` and find the
            expressions rendered upside-down.

    Returns:
        Number of samples successfully written.
    """
    inkml_files = discover_inkml_files(data_root)
    if not inkml_files:
        logger.warning(
            "No .inkml files found under any of %s in %s. "
            "Check --data-root points at the folder containing CROHME_training_2011 etc.",
            DEFAULT_INKML_FOLDER_NAMES,
            data_root,
        )
        return 0

    if limit:
        inkml_files = inkml_files[:limit]
    logger.info("Found %d InkML files to process", len(inkml_files))

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    skipped = 0
    for i, path in enumerate(inkml_files):
        parsed = parse_inkml(path)
        if parsed is None:
            skipped += 1
            continue
        strokes, truth = parsed
        try:
            image = render_strokes(strokes, canvas_size=canvas_size, line_width=line_width, flip_y=flip_y)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to render %s", path, exc_info=True)
            skipped += 1
            continue

        filename = f"{path.stem}_{i}.png"
        image.save(images_dir / filename)
        rows.append({"filename": filename, "latex": truth})

        if (i + 1) % 1000 == 0:
            logger.info("Processed %d/%d InkML files (%d skipped so far)", i + 1, len(inkml_files), skipped)

    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "latex"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "CROHME manifest written: %d samples, %d skipped (no truth annotation or render failure) -> %s",
        len(rows), skipped, labels_path,
    )
    return len(rows)


# ----------------------------------------------------------------------
# Folder-per-class symbols dataset
# ----------------------------------------------------------------------
def build_symbols_manifest(
    symbols_root: Path,
    output_dir: Path,
    limit_per_class: Optional[int] = None,
) -> int:
    """Convert a folder-per-class symbols dataset into a train_ocr.py-ready manifest.

    Args:
        symbols_root: The folder containing one subfolder per symbol
            class (e.g. ``.../mathvision-ai/dataset``, with
            subfolders ``0``, ``1``, ..., ``add``, ``eq``, ``x``, ...).
        output_dir: Where to write ``images/`` and ``labels.csv``
            (pass ``settings.raw_data_dir / "math_symbols"`` to match
            ``train_ocr.py``'s default).
        limit_per_class: Optional cap on images copied per class, for
            a faster smoke test.

    Returns:
        Number of samples successfully written.
    """
    if not symbols_root.exists():
        logger.warning("Symbols folder not found: %s", symbols_root)
        return 0

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    class_dirs = sorted(p for p in symbols_root.iterdir() if p.is_dir())
    if not class_dirs:
        logger.warning("No class subfolders found under %s", symbols_root)
        return 0

    for class_dir in class_dirs:
        latex_label = SYMBOL_FOLDER_TO_LATEX.get(class_dir.name, class_dir.name)
        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if limit_per_class:
            image_paths = image_paths[:limit_per_class]

        for img_path in image_paths:
            dest_name = f"{class_dir.name}_{img_path.name}"
            if not dest_name.lower().endswith(".png"):
                dest_name += ".png"
            try:
                Image.open(img_path).convert("L").save(images_dir / dest_name)
            except Exception:  # noqa: BLE001
                logger.warning("Could not process symbol image %s", img_path, exc_info=True)
                continue
            rows.append({"filename": dest_name, "latex": latex_label})

        logger.info("Class '%s' -> LaTeX '%s': %d images", class_dir.name, latex_label, len(image_paths))

    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "latex"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Math-symbols manifest written: %d samples -> %s", len(rows), labels_path)
    return len(rows)


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw downloaded CROHME/InkML + symbol-folder datasets into "
        "the images/+labels.csv format scripts/train_ocr.py expects."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help=r'Folder containing your downloaded dataset folders, '
        r'e.g. "D:\Data_Science_Capstone_Datasets\mathvision-ai"',
    )
    parser.add_argument(
        "--symbols-subfolder",
        default="dataset",
        help="Name of the folder-per-class symbols folder under --data-root (default: 'dataset')",
    )
    parser.add_argument("--crohme-only", action="store_true", help="Skip the symbols dataset")
    parser.add_argument("--symbols-only", action="store_true", help="Skip the InkML/CROHME datasets")
    parser.add_argument("--limit", type=int, default=None, help="Cap InkML files processed (smoke test)")
    parser.add_argument("--canvas-width", type=int, default=384)
    parser.add_argument("--canvas-height", type=int, default=96)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip the vertical axis when rendering strokes. Only needed if you "
        "check a few images in data/raw/crohme/images/ after a first run and find "
        "the handwriting upside-down (CROHME's own digitizer coordinates are "
        "y-down, matching image coordinates, so this is off by default).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"--data-root does not exist: {data_root}")

    if not args.symbols_only:
        n_crohme = build_crohme_manifest(
            data_root,
            settings.raw_data_dir / "crohme",
            canvas_size=(args.canvas_width, args.canvas_height),
            line_width=args.line_width,
            limit=args.limit,
            flip_y=args.flip_y,
        )
        print(f"CROHME (InkML): {n_crohme} samples -> {settings.raw_data_dir / 'crohme'}")
        print("  -> Open a couple of the images in that folder to eyeball orientation;")
        print("     if the handwriting looks upside-down, rerun this script with --flip-y.")

    if not args.crohme_only:
        n_symbols = build_symbols_manifest(
            data_root / args.symbols_subfolder,
            settings.raw_data_dir / "math_symbols",
        )
        print(f"Math symbols:   {n_symbols} samples -> {settings.raw_data_dir / 'math_symbols'}")

    print("\nDone. Now run train_ocr.py exactly as documented, no path changes needed:")
    print("  python scripts/train_ocr.py --epochs 5 --batch-size 8 --lr 5e-5")


if __name__ == "__main__":
    main()
