"""Unit tests for src.image_processor.

Uses synthetically generated images (no real handwriting samples
required) so these tests run anywhere, including CI, without any
downloaded datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.image_processor import ImageProcessingError, ImageProcessor


@pytest.fixture()
def processor() -> ImageProcessor:
    return ImageProcessor()


def _make_sharp_image(size: int = 300) -> np.ndarray:
    """A synthetic image with a crisp black square on white — should pass quality checks."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[100:200, 100:200] = 0
    return img


def _make_blurry_image(size: int = 300) -> np.ndarray:
    import cv2

    img = _make_sharp_image(size)
    return cv2.GaussianBlur(img, (31, 31), 15)


def _make_clean_digital_page() -> np.ndarray:
    """Pure-white canvas with thin black ink (Paint / stylus / scan) - must NOT count as glare."""
    img = np.full((300, 600, 3), 255, dtype=np.uint8)
    img[140:150, 40:560] = 0
    return img


def _make_glare_image() -> np.ndarray:
    """Washed-out photo: bright everywhere, even the 'ink' is pale grey."""
    return np.full((300, 300, 3), 255, dtype=np.uint8) - np.random.default_rng(0).integers(0, 20, (300, 300, 3), dtype=np.uint8)


def _make_dark_image(size: int = 300) -> np.ndarray:
    return np.full((size, size, 3), 10, dtype=np.uint8)


class TestQualityChecks:
    def test_sharp_image_passes(self, processor: ImageProcessor) -> None:
        report = processor.check_quality(_make_sharp_image())
        assert not report.is_blurry

    def test_blurry_image_flagged(self, processor: ImageProcessor) -> None:
        report = processor.check_quality(_make_blurry_image())
        assert report.is_blurry
        assert "blurry" in report.message.lower()

    def test_clean_white_digital_page_is_not_glare(self, processor: ImageProcessor) -> None:
        assert not processor.check_quality(_make_clean_digital_page()).is_too_bright

    def test_washed_out_photo_is_flagged_as_glare(self, processor: ImageProcessor) -> None:
        assert processor.check_quality(_make_glare_image()).is_too_bright

    def test_dark_image_flagged(self, processor: ImageProcessor) -> None:
        report = processor.check_quality(_make_dark_image())
        assert report.is_too_dark
        assert not report.is_acceptable


class TestBinarization:
    def test_binary_output_is_two_valued(self, processor: ImageProcessor) -> None:
        binary = processor.enhance_and_binarize(_make_sharp_image())
        unique_values = set(np.unique(binary).tolist())
        assert unique_values.issubset({0, 255})

    def test_ink_is_darker_than_background(self, processor: ImageProcessor) -> None:
        binary = processor.enhance_and_binarize(_make_sharp_image())
        # Background (white paper) should be the majority pixel value.
        white_count = int(np.sum(binary == 255))
        black_count = int(np.sum(binary == 0))
        assert white_count > black_count


class TestCropping:
    def test_crop_reduces_size_around_ink(self, processor: ImageProcessor) -> None:
        binary = processor.enhance_and_binarize(_make_sharp_image())
        cropped, bbox = processor.detect_and_crop(binary, padding=10)
        assert bbox is not None
        assert cropped.shape[0] <= binary.shape[0]
        assert cropped.shape[1] <= binary.shape[1]

    def test_blank_image_returns_none_bbox(self, processor: ImageProcessor) -> None:
        blank = np.full((100, 100), 255, dtype=np.uint8)
        cropped, bbox = processor.detect_and_crop(blank)
        assert bbox is None


class TestFullPipeline:
    def test_process_end_to_end(self, processor: ImageProcessor) -> None:
        result = processor.process(_make_sharp_image())
        assert result.cropped is not None
        assert result.quality is not None

    def test_invalid_input_raises(self, processor: ImageProcessor) -> None:
        with pytest.raises(ImageProcessingError):
            processor.process("/nonexistent/path/to/image.png")
