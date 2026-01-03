"""Configuration objects used across the crack detection package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration for image preprocessing prior to crack detection."""

    resize_to: Optional[Tuple[int, int]] = None
    convert_to_grayscale: bool = True
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.4
    apply_bilateral: bool = True
    bilateral_kernel_size: int = 5
    bilateral_sigma_color: float = 12.0
    bilateral_sigma_space: float = 4.0


@dataclass(slots=True)
class DetectorConfig:
    """Configuration that controls the crack detector behavior."""

    low_threshold: float = 12.0
    high_threshold: float = 45.0
    weak_pixel_value: float = 75.0
    min_crack_area: int = 40
    max_component_axis_ratio: float = 0.80
    suppress_grid_lines: bool = True
    grid_row_density_threshold: float = 0.35
    grid_column_density_threshold: float = 0.35
    grid_kernel_size: int = 21
    grid_kernel_response: float = 0.65
    grid_orientation_tolerance: float = 10.0
    grid_axis_ratio_threshold: float = 5.0
    grid_perimeter_area_threshold: float = 0.22
    texture_kernel_size: int = 9
    texture_variance_threshold: float = 40.0
    texture_min_length: float = 18.0
    thin_crack_max_width: float = 4.0
    thin_crack_min_length: float = 28.0
    thin_low_threshold: float = 6.0
    thin_high_threshold: float = 28.0
    light_gaussian_kernel_size: int = 3
    light_gaussian_sigma: float = 0.8


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for loading datasets from disk."""

    image_root: Path
    image_subdir: Optional[str] = None
    extensions: Sequence[str] = field(default_factory=lambda: (".jpg", ".jpeg", ".png", ".bmp"))

    def resolved_root(self) -> Path:
        """Return the directory that actually contains the image files."""

        base = self.image_root
        if self.image_subdir:
            return base / self.image_subdir
        auto_candidate = base / "image"
        auto_label = base / "label"
        if auto_candidate.exists() and auto_label.exists():
            return auto_candidate
        return base

    def validate(self) -> Path:
        """Ensure the dataset configuration points to a valid directory."""

        root = self.resolved_root()
        if not root.exists():
            msg = f"image_root {root} does not exist"
            raise FileNotFoundError(msg)
        if not root.is_dir():
            msg = f"image_root {root} is not a directory"
            raise NotADirectoryError(msg)
        return root


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

        root = self.dataset.validate()
        count = 0
        for path in sorted(root.rglob("*")):
            if self.sample_limit is not None and count >= self.sample_limit:
                break
            if path.suffix.lower() not in self.dataset.extensions:
                continue
            yield path
            count += 1
