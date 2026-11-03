"""Image capture & preprocessing pipeline (Module 1).

Takes a raw photo of a handwritten math problem (from camera or file
upload) and produces a clean, cropped, high-contrast image that is
ready for the OCR engine, along with a quality report the frontend can
use to prompt the student to retake the photo.

Pipeline stages:
    1. Load + EXIF auto-rotation
    2. Quality checks (blur, brightness) -- fail fast with a friendly
       message before spending compute on a bad image
    3. Perspective correction (if the page was photographed at an angle)
    4. Grayscale + CLAHE contrast enhancement
    5. Otsu binarization
    6. Morphological noise removal
    7. Problem-region detection & crop (largest ink bounding box)

Example:
    >>> from src.image_processor import ImageProcessor
    >>> processor = ImageProcessor()
    >>> result = processor.process("sample_problems/quadratic.jpg")
    >>> result.quality.is_acceptable
    True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python-headless is required for image_processor.py. "
        "Install with: pip install opencv-python-headless"
    ) from exc

try:
    from PIL import Image, ExifTags
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required for image_processor.py. pip install Pillow") from exc

from src.config import settings

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, bytes, np.ndarray]


class ImageProcessingError(Exception):
    """Raised when an image cannot be loaded or processed at all."""


@dataclass
class QualityReport:
    """Result of the automated image-quality checks.

    Attributes:
        blur_variance: Laplacian variance; lower means blurrier.
        brightness: Mean pixel intensity (0-255).
        is_blurry: True if the image is too blurry to OCR reliably.
        is_too_dark: True if the mean brightness is below threshold.
        is_too_bright: True if the mean brightness is above threshold
            (e.g. overexposed/glare).
        is_acceptable: Overall pass/fail flag.
        message: Human-readable, student-facing message (Hinglish-friendly)
            describing the issue, or a success message.
    """

    blur_variance: float
    brightness: float
    is_blurry: bool
    is_too_dark: bool
    is_too_bright: bool

    @property
    def is_acceptable(self) -> bool:
        return not (self.is_blurry or self.is_too_dark or self.is_too_bright)

    @property
    def message(self) -> str:
        if self.is_blurry:
            return "The photo is blurry. Please retake it and hold the camera steady."
        if self.is_too_dark:
            return "The photo is too dark. Please retake it in better light."
        if self.is_too_bright:
            return "The photo is overexposed (glare). Please retake it with less glare."
        return "Image quality is good — processing your problem!"


@dataclass
class ProcessedImage:
    """Final output of the preprocessing pipeline.

    Attributes:
        original: The original loaded image (BGR, np.ndarray).
        cropped: The cropped, binarized problem-region image ready
            for OCR (single channel, np.ndarray).
        quality: The :class:`QualityReport` computed on the original image.
        bounding_box: (x, y, w, h) of the detected problem region in the
            original image's coordinate space, or ``None`` if detection
            failed and the full image was used instead.
    """

    original: np.ndarray
    cropped: np.ndarray
    quality: QualityReport
    bounding_box: Optional[tuple[int, int, int, int]]


class ImageProcessor:
    """Encapsulates the full Module 1 preprocessing pipeline."""

    def __init__(
        self,
        blur_threshold: Optional[float] = None,
        min_brightness: Optional[float] = None,
        max_brightness: Optional[float] = None,
        target_size: Optional[int] = None,
    ) -> None:
        """Initialize the processor with configurable thresholds.

        Args:
            blur_threshold: Laplacian-variance cutoff below which an
                image is flagged as blurry. Defaults to ``settings``.
            min_brightness: Minimum acceptable mean brightness.
            max_brightness: Maximum acceptable mean brightness.
            target_size: Longest-edge size (px) images are resized to
                before processing, for consistent downstream behaviour.
        """
        self.blur_threshold = blur_threshold or settings.blur_variance_threshold
        self.min_brightness = min_brightness or settings.min_brightness
        self.max_brightness = max_brightness or settings.max_brightness
        self.target_size = target_size or settings.target_image_size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_image(self, image: ImageInput) -> np.ndarray:
        """Load an image from a path, bytes, or ndarray into BGR uint8.

        Applies EXIF-based auto-rotation when loading from a path or
        raw bytes (common for phone-camera photos).

        Raises:
            ImageProcessingError: If the image cannot be decoded.
        """
        try:
            if isinstance(image, np.ndarray):
                return image.copy()

            if isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            elif isinstance(image, bytes):
                import io

                pil_image = Image.open(io.BytesIO(image))
            else:
                raise ImageProcessingError(f"Unsupported image input type: {type(image)}")

            pil_image = self._apply_exif_rotation(pil_image)
            rgb = np.array(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImageProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as our own error type
            logger.exception("Failed to load image")
            raise ImageProcessingError(f"Could not load image: {exc}") from exc

    @staticmethod
    def _apply_exif_rotation(pil_image: "Image.Image") -> "Image.Image":
        """Rotate a PIL image according to its EXIF orientation tag."""
        try:
            exif = pil_image.getexif()
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key is None or orientation_key not in exif:
                return pil_image

            orientation = exif[orientation_key]
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                return pil_image.rotate(rotations[orientation], expand=True)
        except Exception:  # noqa: BLE001 - EXIF issues should never crash the pipeline
            logger.debug("No usable EXIF orientation data found", exc_info=True)
        return pil_image

    def _resize_longest_edge(self, image: np.ndarray) -> np.ndarray:
        """Resize so the longest edge equals ``self.target_size`` (downscale only)."""
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self.target_size:
            return image
        scale = self.target_size / longest
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------
    def check_quality(self, image: np.ndarray) -> QualityReport:
        """Run blur and brightness checks on a BGR image.

        Args:
            image: BGR image as a NumPy array.

        Returns:
            A populated :class:`QualityReport`.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        whole_variance = float(laplacian.var())
        # A sharp photo of a mostly-blank page has low whole-image variance
        # (flat paper dominates). Measure sharpness on the ink neighbourhood too.
        ink_variance = self._ink_region_variance(gray, laplacian)
        blur_variance = max(whole_variance, ink_variance)
        brightness = float(gray.mean())

        report = QualityReport(
            blur_variance=blur_variance,
            brightness=brightness,
            is_blurry=blur_variance < self.blur_threshold,
            is_too_dark=brightness < self.min_brightness,
            # Glare washes the ink out too; a clean digital/scanned page is bright
            # but its ink stays dark, so it must not be rejected.
            is_too_bright=brightness > self.max_brightness and float(np.percentile(gray, 1)) > 100,
        )
        logger.info(
            "Quality check: blur_var=%.2f brightness=%.2f acceptable=%s",
            blur_variance,
            brightness,
            report.is_acceptable,
        )
        return report

    @staticmethod
    def _ink_region_variance(gray: np.ndarray, laplacian: np.ndarray) -> float:
        """Laplacian variance restricted to the pixels around pen strokes.

        Args:
            gray: Grayscale image.
            laplacian: Precomputed Laplacian of ``gray`` (float64).

        Returns:
            Variance inside the dilated ink mask, or 0.0 when no ink is found.
        """
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
        )
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
        if int(mask.sum()) < 50:
            return 0.0
        return float(laplacian[mask].var())

    # ------------------------------------------------------------------
    # Perspective correction
    # ------------------------------------------------------------------
    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Attempt to detect the page/paper quadrilateral and warp it flat.

        Falls back to the original image untouched if no confident
        4-point contour is found (e.g. the paper fills the whole frame).

        Args:
            image: BGR input image.

        Returns:
            Perspective-corrected BGR image (or the original on failure).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        image_area = image.shape[0] * image.shape[1]
        if cv2.contourArea(largest) < 0.25 * image_area:
            # Too small to be confidently "the page" - skip correction.
            return image

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) != 4:
            return image

        try:
            return self._warp_perspective(image, approx.reshape(4, 2).astype(np.float32))
        except Exception:  # noqa: BLE001
            logger.warning("Perspective warp failed, returning original image", exc_info=True)
            return image

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _warp_perspective(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Warp ``image`` so that quadrilateral ``pts`` becomes a rectangle."""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        if max_width == 0 or max_height == 0:
            return image

        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    # ------------------------------------------------------------------
    # Enhancement + binarization
    # ------------------------------------------------------------------
    def enhance_and_binarize(self, image: np.ndarray) -> np.ndarray:
        """Grayscale -> CLAHE contrast boost -> Otsu binarization -> denoise.

        Args:
            image: BGR input image.

        Returns:
            Single-channel binary image (white background, black ink).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure ink is dark-on-light (standard convention for downstream OCR).
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        return denoised

    # ------------------------------------------------------------------
    # Problem-region detection & crop
    # ------------------------------------------------------------------
    def detect_and_crop(
        self, binary_image: np.ndarray, padding: int = 20
    ) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
        """Find the bounding box containing ink and crop to it.

        Args:
            binary_image: Single-channel binary image (white bg, black ink).
            padding: Pixels of whitespace padding to keep around the crop.

        Returns:
            A tuple of (cropped_image, bounding_box) where bounding_box is
            (x, y, w, h) in the input image's coordinate space, or ``None``
            if no ink was detected (the input is returned unchanged).
        """
        ink_mask = cv2.bitwise_not(binary_image)  # ink becomes white for contour finding
        coords = cv2.findNonZero(ink_mask)
        if coords is None:
            logger.warning("No ink detected; returning full image")
            return binary_image, None

        x, y, w, h = cv2.boundingRect(coords)
        h_img, w_img = binary_image.shape[:2]
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)

        cropped = binary_image[y0:y1, x0:x1]
        return cropped, (x0, y0, x1 - x0, y1 - y0)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def process(self, image: ImageInput) -> ProcessedImage:
        """Run the full preprocessing pipeline on an input image.

        Args:
            image: File path, raw bytes, or an ``np.ndarray`` (BGR).

        Returns:
            A :class:`ProcessedImage` bundling the original array, the
            cropped/binarized result, the quality report, and the
            detected bounding box.

        Raises:
            ImageProcessingError: If the image cannot be loaded/decoded.
        """
        original = self._load_image(image)
        original = self._resize_longest_edge(original)

        quality = self.check_quality(original)
        if not quality.is_acceptable:
            logger.info("Image failed quality check: %s", quality.message)
            # We still return a best-effort processed image so the caller
            # can decide whether to proceed or ask the student to retake it.

        corrected = self.correct_perspective(original)
        binary = self.enhance_and_binarize(corrected)
        cropped, bbox = self.detect_and_crop(binary)

        return ProcessedImage(
            original=original,
            cropped=cropped,
            quality=quality,
            bounding_box=bbox,
        )

    def save_debug_images(self, result: ProcessedImage, output_dir: Union[str, Path]) -> None:
        """Write intermediate images to disk for visual debugging.

        Args:
            result: The output of :meth:`process`.
            output_dir: Directory to write ``original.png`` / ``cropped.png`` into.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "original.png"), result.original)
        cv2.imwrite(str(output_dir / "cropped.png"), result.cropped)
        logger.info("Saved debug images to %s", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a handwritten math photo.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--out", type=str, default="debug_output", help="Debug output directory")
    args = parser.parse_args()

    processor = ImageProcessor()
    processed = processor.process(args.image_path)
    print(f"Quality: {processed.quality}")
    print(f"Message: {processed.quality.message}")
    processor.save_debug_images(processed, args.out)
