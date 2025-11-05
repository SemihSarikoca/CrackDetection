"""Image preprocessing utilities used by the crack detection pipeline."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import PreprocessingConfig


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale if required."""

    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_equalized = clahe.apply(l_channel)
    merged = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gaussian_blur(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply a Gaussian blur to suppress noise prior to edge detection."""

    kernel = config.gaussian_kernel_size
    sigma = config.gaussian_sigma
    return cv2.GaussianBlur(image, kernel, sigma)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize the image to the requested spatial resolution."""

    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Full preprocessing routine prior to crack detection."""

    result = image
    if config.resize_to:
        result = resize_image(result, config.resize_to)
    if config.convert_to_grayscale:
        result = to_grayscale(result)
    result = apply_clahe(result, config)
    result = gaussian_blur(result, config)
    return result
