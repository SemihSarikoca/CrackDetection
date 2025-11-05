"""Concrete crack detection strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import cv2
import numpy as np

from .config import DetectorConfig


@dataclass(slots=True)
class CrackDetectionResult:
    """Output produced by a crack detector."""

    mask: np.ndarray
    score_map: np.ndarray
    summary: Dict[str, float]


class CrackDetector(Protocol):
    """Protocol for crack detector implementations."""

    def detect(self, image: np.ndarray) -> CrackDetectionResult:
        ...


class CannyCrackDetector:
    """Baseline crack detector built around Canny edge detection."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        kernel_size = config.dilation_kernel_size
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    def detect(self, image: np.ndarray) -> CrackDetectionResult:
        raw_edges = cv2.Canny(image, self._config.low_threshold, self._config.high_threshold)
        dilated = cv2.dilate(raw_edges, self._kernel, iterations=self._config.dilation_iterations)
        mask = _remove_small_components(dilated, self._config.min_crack_area)
        normalized = cv2.normalize(mask.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        crack_pixels = float(np.count_nonzero(mask))
        coverage = crack_pixels / mask.size
        return CrackDetectionResult(
            mask=mask,
            score_map=normalized,
            summary={
                "crack_coverage": coverage,
                "crack_pixels": crack_pixels,
                "total_pixels": float(mask.size),
            },
        )


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than the threshold."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    filtered = np.zeros_like(mask)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == idx] = 255
    return filtered
