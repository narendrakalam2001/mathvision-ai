"""Module 2, Stage 1 — layout analysis for the OCR pipeline.

The fine-tuned TrOCR model reads exactly ONE expression per image, rendered
the way the training data was: dark strokes on a white 384x96 canvas with a
small margin and the aspect ratio preserved. A photographed page usually
contains several lines, so this module

1. splits a binarized page into single-line crops (horizontal projection
   profile on a noise-cleaned ink mask), and
2. normalises every crop to the training-data look (tight crop, aspect-
   preserving fit, white padding, stroke thickness close to the training
   line width).

Example:
    >>> analyzer = LayoutAnalyzer()
    >>> lines = analyzer.analyze(binary_page)   # black ink on white, uint8
    >>> for line in lines:
    ...     latex, conf = ocr_backend.recognize(line.image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CANVAS_W, CANVAS_H = 384, 96      # training canvas (scripts/prepare_datasets.py)
CANVAS_PADDING = 10               # training padding
TARGET_STROKE_PX = 3.0            # training line width


@dataclass
class LineCrop:
    """One detected expression line.

    Attributes:
        index: Zero-based top-to-bottom position of the line.
        bbox: ``(x, y, w, h)`` of the tight ink box in the source image.
        image: Normalised 96x384 grayscale image (black ink, white background)
            ready for the OCR backend.
    """

    index: int
    bbox: tuple[int, int, int, int]
    image: np.ndarray


class LayoutAnalyzer:
    """Split a binarized page into expression lines and normalise them."""

    def __init__(
        self,
        merge_gap_ratio: float = 0.3,
        min_component_area: int = 12,
        min_line_ink_ratio: float = 0.05,
    ) -> None:
        """Configure the segmenter.

        Args:
            merge_gap_ratio: Two ink row-runs are merged into one line when the
                blank gap between them is smaller than this fraction of the
                smaller run's height (keeps ``x^2`` and fractions together).
            min_component_area: Connected components with fewer ink pixels are
                treated as noise and removed before segmentation.
            min_line_ink_ratio: A run is dropped as stray marks when its ink
                pixel count is below this fraction of the heaviest run.
        """
        self.merge_gap_ratio = merge_gap_ratio
        self.min_component_area = min_component_area
        self.min_line_ink_ratio = min_line_ink_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image: np.ndarray, max_lines: Optional[int] = None) -> list[LineCrop]:
        """Segment ``image`` into normalised single-expression crops.

        Args:
            image: Grayscale or BGR image with dark ink on a light background
                (already binarized or not; Otsu is applied internally).
            max_lines: Optional cap on the number of returned lines.

        Returns:
            Lines ordered top to bottom. Empty list if no ink was found.
        """
        ink = self._ink_mask(image)
        ink = self._remove_noise(ink)
        boxes = self._split_rows(ink)
        lines: list[LineCrop] = []
        for i, (y0, y1) in enumerate(boxes):
            band = ink[y0 : y1 + 1]
            cols = np.where(band.any(axis=0))[0]
            if cols.size == 0:
                continue
            x0, x1 = int(cols[0]), int(cols[-1])
            tight = band[:, x0 : x1 + 1]
            lines.append(
                LineCrop(index=len(lines), bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                         image=self.normalize(tight))
            )
            if max_lines and len(lines) >= max_lines:
                break
        logger.info("Layout analysis: %d line(s) found", len(lines))
        return lines

    def normalize(self, ink_crop: np.ndarray) -> np.ndarray:
        """Render a tight boolean ink crop onto the training-style canvas.

        Args:
            ink_crop: Boolean/uint8 array, True/255 where there is ink,
                cropped tightly to the expression.

        Returns:
            ``(96, 384)`` uint8 image, black ink on white.
        """
        ink = (ink_crop > 0).astype(np.uint8) * 255
        h, w = ink.shape
        avail_w, avail_h = CANVAS_W - 2 * CANVAS_PADDING, CANVAS_H - 2 * CANVAS_PADDING
        scale = min(avail_w / w, avail_h / h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(ink, (new_w, new_h), interpolation=interp)
        resized = self._match_stroke_width(resized)
        canvas = np.full((CANVAS_H, CANVAS_W), 255, np.uint8)
        ox, oy = (CANVAS_W - new_w) // 2, (CANVAS_H - new_h) // 2
        region = canvas[oy : oy + new_h, ox : ox + new_w]
        region[resized > 127] = 0
        return canvas

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _ink_mask(image: np.ndarray) -> np.ndarray:
        """Return a uint8 mask with 255 on ink (dark pixels)."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if mask.mean() > 127:           # more "ink" than paper -> polarity was inverted
            mask = 255 - mask
        return mask

    def _remove_noise(self, ink: np.ndarray) -> np.ndarray:
        """Drop tiny connected components (specks, paper grain)."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        keep = np.zeros_like(ink)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] >= self.min_component_area:
                keep[labels == lbl] = 255
        return keep

    def _split_rows(self, ink: np.ndarray) -> list[tuple[int, int]]:
        """Find (y0, y1) row ranges of text lines via projection profile."""
        has_ink = ink.any(axis=1)
        runs: list[list[int]] = []
        start = None
        for y, flag in enumerate(has_ink):
            if flag and start is None:
                start = y
            elif not flag and start is not None:
                runs.append([start, y - 1]); start = None
        if start is not None:
            runs.append([start, len(has_ink) - 1])
        if not runs:
            return []

        merged: list[list[int]] = [runs[0]]
        for y0, y1 in runs[1:]:
            py0, py1 = merged[-1]
            gap = y0 - py1 - 1
            smaller = min(py1 - py0 + 1, y1 - y0 + 1)
            if gap < max(3, self.merge_gap_ratio * smaller):
                merged[-1][1] = y1
            else:
                merged.append([y0, y1])

        weights = [int((ink[a : b + 1] > 0).sum()) for a, b in merged]
        heaviest = max(weights)
        return [(a, b) for (a, b), wgt in zip(merged, weights)
                if wgt >= self.min_line_ink_ratio * heaviest]

    @staticmethod
    def _match_stroke_width(ink: np.ndarray) -> np.ndarray:
        """Thin overly thick strokes (and thicken hairlines) towards the training width."""
        mask = ink > 127
        if mask.sum() < 20:
            return ink
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        width = 2.0 * float(np.percentile(dist[mask], 90)) / 0.9
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if width > TARGET_STROKE_PX * 1.5:
            iters = min(2, int(round((width - TARGET_STROKE_PX) / 2)))
            return cv2.erode(ink, kernel, iterations=max(1, iters))
        if width < TARGET_STROKE_PX * 0.6:
            return cv2.dilate(ink, kernel, iterations=1)
        return ink
