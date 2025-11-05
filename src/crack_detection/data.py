"""Data loading helpers for the crack detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np

from .config import PipelineConfig


@dataclass(slots=True)
class ImageSample:
    """In-memory representation of an image and associated metadata."""

    path: Path
    image: np.ndarray

    def clone(self) -> "ImageSample":
        """Return a copy of the sample with a deep copy of the pixel data."""

        return ImageSample(path=self.path, image=self.image.copy())


class ImageDataset:
    """Iterable dataset that streams image samples from disk."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def __iter__(self) -> Iterator[ImageSample]:
        for image_path in self._config.iter_image_paths():
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            yield ImageSample(path=image_path, image=image)

    def take(self, limit: Optional[int]) -> Iterable[ImageSample]:
        if limit is None:
            yield from iter(self)
            return
        for idx, sample in enumerate(self):
            yield sample
            if idx + 1 >= limit:
                break
