"""Unit tests for scripts.prepare_datasets.

Uses small synthetic InkML files (matching real CROHME XML structure)
and a synthetic folder-per-class symbols dataset, so these tests never
depend on the actual multi-GB downloaded datasets.
"""

from __future__ import annotations

import csv

import pytest
from PIL import Image

from scripts.prepare_datasets import (
    build_crohme_manifest,
    build_symbols_manifest,
    discover_inkml_files,
    parse_inkml,
    render_strokes,
)

VALID_INKML = """<?xml version="1.0" encoding="UTF-8"?>
<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="writer">w001</annotation>
  <annotation type="truth">$ x ^ { 2 } - 5 x + 6 = 0 $</annotation>
  <trace id="0">100 100, 105 95, 110 90, 115 100, 110 110, 105 105, 100 100</trace>
  <trace id="1">130 90, 135 110</trace>
</ink>
"""

NO_TRUTH_INKML = """<?xml version="1.0" encoding="UTF-8"?>
<ink xmlns="http://www.w3.org/2003/InkML">
  <trace id="0">10 10, 20 20</trace>
</ink>
"""

MALFORMED_INKML = "<ink><trace>not closed"


@pytest.fixture()
def inkml_file(tmp_path):
    path = tmp_path / "sample.inkml"
    path.write_text(VALID_INKML, encoding="utf-8")
    return path


class TestParseInkml:
    def test_extracts_truth_and_strokes(self, inkml_file) -> None:
        result = parse_inkml(inkml_file)
        assert result is not None
        strokes, truth = result
        assert truth == "x ^ { 2 } - 5 x + 6 = 0"
        assert len(strokes) == 2
        assert strokes[0][0] == (100.0, 100.0)

    def test_missing_truth_returns_none(self, tmp_path) -> None:
        path = tmp_path / "no_truth.inkml"
        path.write_text(NO_TRUTH_INKML, encoding="utf-8")
        assert parse_inkml(path) is None

    def test_malformed_xml_returns_none(self, tmp_path) -> None:
        path = tmp_path / "broken.inkml"
        path.write_text(MALFORMED_INKML, encoding="utf-8")
        assert parse_inkml(path) is None

    def test_dollar_signs_stripped(self, tmp_path) -> None:
        path = tmp_path / "dollars.inkml"
        path.write_text(VALID_INKML, encoding="utf-8")
        _, truth = parse_inkml(path)
        assert not truth.startswith("$")
        assert not truth.endswith("$")


class TestRenderStrokes:
    def test_produces_correct_size_image(self, inkml_file) -> None:
        strokes, _ = parse_inkml(inkml_file)
        image = render_strokes(strokes, canvas_size=(384, 96))
        assert image.size == (384, 96)
        assert image.mode == "L"

    def test_image_contains_ink_and_background(self, inkml_file) -> None:
        import numpy as np

        strokes, _ = parse_inkml(inkml_file)
        image = render_strokes(strokes)
        arr = np.array(image)
        assert (arr < 128).any(), "Rendered image should contain dark ink pixels"
        assert (arr > 200).any(), "Rendered image should contain a light background"

    def test_single_point_stroke_does_not_crash(self) -> None:
        image = render_strokes([[(50.0, 50.0)]], canvas_size=(100, 100))
        assert image.size == (100, 100)

    def test_flip_y_inverts_vertical_order(self) -> None:
        # A diagonal stroke: flipping y should mirror it into a different diagonal.
        strokes = [[(5.0, 5.0), (45.0, 45.0)]]
        normal = render_strokes(strokes, canvas_size=(50, 50), padding=0, line_width=1, flip_y=False)
        flipped = render_strokes(strokes, canvas_size=(50, 50), padding=0, line_width=1, flip_y=True)
        import numpy as np

        normal_arr, flipped_arr = np.array(normal), np.array(flipped)
        assert (normal_arr < 128).sum() > 0 and (flipped_arr < 128).sum() > 0
        assert not np.array_equal(normal_arr, flipped_arr)


class TestDiscoverInkmlFiles:
    def test_finds_flat_and_nested_files(self, tmp_path) -> None:
        flat_dir = tmp_path / "CROHME_training_2011"
        flat_dir.mkdir()
        (flat_dir / "a.inkml").write_text(VALID_INKML, encoding="utf-8")

        nested_dir = tmp_path / "TrainINKML_2013" / "HAMEX"
        nested_dir.mkdir(parents=True)
        (nested_dir / "b.inkml").write_text(VALID_INKML, encoding="utf-8")

        found = discover_inkml_files(tmp_path)
        assert len(found) == 2

    def test_missing_folders_are_skipped_not_errored(self, tmp_path) -> None:
        found = discover_inkml_files(tmp_path)
        assert found == []


class TestBuildCrohmeManifest:
    def test_writes_images_and_csv(self, tmp_path) -> None:
        data_root = tmp_path / "raw"
        crohme_dir = data_root / "CROHME_training_2011"
        crohme_dir.mkdir(parents=True)
        (crohme_dir / "eq1.inkml").write_text(VALID_INKML, encoding="utf-8")
        (crohme_dir / "eq2_no_truth.inkml").write_text(NO_TRUTH_INKML, encoding="utf-8")

        output_dir = tmp_path / "out" / "crohme"
        count = build_crohme_manifest(data_root, output_dir)

        assert count == 1  # the no-truth file must be skipped, not crash the run
        labels_path = output_dir / "labels.csv"
        assert labels_path.exists()

        with open(labels_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["latex"] == "x ^ { 2 } - 5 x + 6 = 0"
        assert (output_dir / "images" / rows[0]["filename"]).exists()

    def test_no_matching_folders_returns_zero(self, tmp_path) -> None:
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        count = build_crohme_manifest(empty_root, tmp_path / "out")
        assert count == 0


class TestBuildSymbolsManifest:
    def test_maps_folder_names_to_latex(self, tmp_path) -> None:
        symbols_root = tmp_path / "dataset"
        for cls in ("0", "add", "eq", "x"):
            class_dir = symbols_root / cls
            class_dir.mkdir(parents=True)
            Image.new("L", (28, 28), color=255).save(class_dir / "img_0.png")

        output_dir = tmp_path / "out" / "math_symbols"
        count = build_symbols_manifest(symbols_root, output_dir)

        assert count == 4
        with open(output_dir / "labels.csv", newline="", encoding="utf-8") as fh:
            rows = {row["filename"]: row["latex"] for row in csv.DictReader(fh)}
        assert any(latex == "+" for latex in rows.values())  # 'add' -> '+'
        assert any(latex == "=" for latex in rows.values())  # 'eq' -> '='
        assert any(latex == "0" for latex in rows.values())  # digit passthrough
        assert any(latex == "x" for latex in rows.values())  # letter passthrough

    def test_missing_symbols_folder_returns_zero(self, tmp_path) -> None:
        count = build_symbols_manifest(tmp_path / "does_not_exist", tmp_path / "out")
        assert count == 0

    def test_empty_symbols_folder_returns_zero(self, tmp_path) -> None:
        symbols_root = tmp_path / "dataset"
        symbols_root.mkdir()
        count = build_symbols_manifest(symbols_root, tmp_path / "out")
        assert count == 0
