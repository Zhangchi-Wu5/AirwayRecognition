"""Tests for filename parsing and manifest building."""
import pytest
from pathlib import Path
from src.data import parse_filename, build_manifest


class TestParseFilename:
    def test_parses_standard_filename(self):
        result = parse_filename("0000003926lt.png")
        assert result == {"patient_id": "0000003926", "label": "lt", "ext": "png"}

    def test_parses_filename_with_space(self):
        result = parse_filename("0000028232 zz.png")
        assert result == {"patient_id": "0000028232", "label": "zz", "ext": "png"}

    def test_parses_jpg_extension(self):
        result = parse_filename("0000032620 zz.jpg")
        assert result == {"patient_id": "0000032620", "label": "zz", "ext": "jpg"}

    def test_parses_all_three_labels(self):
        assert parse_filename("1234567890lt.png")["label"] == "lt"
        assert parse_filename("1234567890yz.png")["label"] == "yz"
        assert parse_filename("1234567890zz.png")["label"] == "zz"

    def test_returns_none_for_invalid_filename(self):
        assert parse_filename("not_a_valid_file.png") is None
        assert parse_filename("1234abc.png") is None
        assert parse_filename(".DS_Store") is None


class TestBuildManifest:
    def test_builds_manifest_from_dir(self, tmp_path):
        # Arrange: create fake dataset dir
        (tmp_path / "0000000001lt.png").touch()
        (tmp_path / "0000000001yz.png").touch()
        (tmp_path / "0000000001zz.png").touch()
        (tmp_path / "0000000002lt.png").touch()
        (tmp_path / ".DS_Store").touch()  # should be skipped

        # Act
        df = build_manifest(tmp_path)

        # Assert
        assert len(df) == 4
        assert set(df.columns) == {"patient_id", "label", "label_id", "path"}
        assert set(df["patient_id"].unique()) == {"0000000001", "0000000002"}
        assert set(df["label"].unique()) == {"lt", "yz", "zz"}

    def test_label_id_mapping(self, tmp_path):
        (tmp_path / "0000000001lt.png").touch()
        (tmp_path / "0000000001yz.png").touch()
        (tmp_path / "0000000001zz.png").touch()
        df = build_manifest(tmp_path)
        label_to_id = dict(zip(df["label"], df["label_id"]))
        assert label_to_id == {"lt": 0, "yz": 1, "zz": 2}
