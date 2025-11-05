"""Pipeline orchestration for crack detection workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .config import PipelineConfig
from .data import ImageDataset, ImageSample
from .detector import CannyCrackDetector, CrackDetectionResult, CrackDetector
from .preprocessing import preprocess
from .visualization import save_visualization


@dataclass(slots=True)
class PipelineArtifact:
    """Artifacts produced for each processed image."""

    sample: ImageSample
    processed: np.ndarray
    result: CrackDetectionResult
    visualization_path: Optional[Path]


class CrackDetectionPipeline:
    """Coordinate dataset loading, preprocessing, detection, and reporting."""

    def __init__(self, config: PipelineConfig, detector: Optional[CrackDetector] = None) -> None:
        self._config = config
        self._detector = detector or CannyCrackDetector(config.detector)
        self._dataset = ImageDataset(config)

    def __iter__(self) -> Iterator[PipelineArtifact]:
        for sample in self._dataset.take(self._config.sample_limit):
            processed = preprocess(sample.image, self._config.preprocessing)
            result = self._detector.detect(processed)
            viz_path = self._maybe_save_visualization(sample, processed, result)
            yield PipelineArtifact(sample=sample, processed=processed, result=result, visualization_path=viz_path)

    def run(self) -> None:
        """Execute the pipeline and print a human-readable summary."""

        for artifact in self:
            summary = artifact.result.summary
            message = _format_summary(artifact.sample.path, summary)
            print(message)

    def _maybe_save_visualization(
        self, sample: ImageSample, processed: np.ndarray, result: CrackDetectionResult
    ) -> Optional[Path]:
        if not self._config.visualize or self._config.output_dir is None:
            return None
        relative = sample.path.relative_to(self._config.dataset.image_root)
        destination = self._config.output_dir / relative.with_suffix("_summary.png")
        save_visualization(sample.image, processed, result, destination, title=str(relative))
        return destination


def _format_summary(path: Path, summary: dict[str, float]) -> str:
    parts = ", ".join(f"{key}={value:.4f}" for key, value in summary.items())
    return f"{path.name}: {parts}"
