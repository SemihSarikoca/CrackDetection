"""Configuration objects used across the crack detection package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration for image preprocessing prior to crack detection."""

    resize_to: Optional[Tuple[int, int]] = (512, 512)
    convert_to_grayscale: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    gaussian_kernel_size: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.2


@dataclass(slots=True)
class DetectorConfig:
    """Configuration that controls the crack detector behavior."""

    low_threshold: int = 50
    high_threshold: int = 150
    dilation_iterations: int = 1
    dilation_kernel_size: Tuple[int, int] = (3, 3)
    min_crack_area: int = 25


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for loading datasets from disk."""

    image_root: Path
    extensions: Sequence[str] = field(default_factory=lambda: (".jpg", ".jpeg", ".png", ".bmp"))

    def validate(self) -> None:
        """Ensure the dataset configuration points to a valid directory."""

        if not self.image_root.exists():
            msg = f"image_root {self.image_root} does not exist"
            raise FileNotFoundError(msg)
        if not self.image_root.is_dir():
            msg = f"image_root {self.image_root} is not a directory"
            raise NotADirectoryError(msg)


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration passed to `CrackDetectionPipeline`."""

    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output_dir: Optional[Path] = None
    sample_limit: Optional[int] = None
    visualize: bool = False

    def iter_image_paths(self) -> Iterable[Path]:
        """Yield all image paths that match the configured extensions."""

        self.dataset.validate()
        count = 0
        for path in sorted(self.dataset.image_root.rglob("*")):
            if self.sample_limit is not None and count >= self.sample_limit:
                break
            if path.suffix.lower() not in self.dataset.extensions:
                continue
            yield path
            count += 1
