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


def test_iter_image_paths_prefers_image_subdir_when_labels_present(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    label_dir = tmp_path / "label"
    image_dir.mkdir()
    label_dir.mkdir()
    valid = image_dir / "a.png"
    valid.write_bytes(b"data")
    (label_dir / "b.png").write_bytes(b"label")

    config = PipelineConfig(dataset=DatasetConfig(image_root=tmp_path))

    paths = list(config.iter_image_paths())

    assert paths == [valid]


def test_iter_image_paths_uses_explicit_image_subdir(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    image_dir = train_dir / "images"
    label_dir = train_dir / "label"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    valid = image_dir / "keep.jpg"
    valid.write_bytes(b"data")
    (label_dir / "ignore.jpg").write_bytes(b"label")

    config = PipelineConfig(dataset=DatasetConfig(image_root=train_dir, image_subdir="images"))

    assert list(config.iter_image_paths()) == [valid]
