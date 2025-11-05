"""Unit tests validating configuration behavior."""

from pathlib import Path

import pytest

from crack_detection.config import DatasetConfig, PipelineConfig


def test_iter_image_paths_respects_extension_and_limit(tmp_path: Path) -> None:
    valid = tmp_path / "a.jpg"
    valid.write_bytes(b"test")
    invalid = tmp_path / "b.txt"
    invalid.write_text("noop", encoding="utf-8")

    config = PipelineConfig(dataset=DatasetConfig(image_root=tmp_path), sample_limit=1)
    paths = list(config.iter_image_paths())

    assert len(paths) == 1
    assert paths[0] == valid


def test_iter_image_paths_handles_zero_limit(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"test")

    config = PipelineConfig(dataset=DatasetConfig(image_root=tmp_path), sample_limit=0)

    assert list(config.iter_image_paths()) == []


def test_dataset_validate_rejects_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    dataset = DatasetConfig(image_root=missing)

    with pytest.raises(FileNotFoundError):
        dataset.validate()
