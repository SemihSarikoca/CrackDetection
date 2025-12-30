"""Image preprocessing utilities used by the crack detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import PreprocessingConfig


@dataclass(slots=True)
class PreprocessedImage:
    """Bundle containing intermediate images for downstream processing."""

    base: np.ndarray  # After optional grayscale + bilateral (texture suppressed, cracks preserved)
    smoothed: np.ndarray  # Additional Gaussian smoothing used by the default detector


def _ensure_odd(value: int) -> int:
    if value % 2 == 0:
        return value + 1
    return value


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using luminance weights."""

    if image.ndim == 2:
        return image.astype(np.float32)
    channels = image[..., :3]
    r, g, b = channels[..., 0], channels[..., 1], channels[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize the image with nearest-neighbor interpolation."""

    target_width, target_height = size
    height, width = image.shape[:2]
    if target_width <= 0 or target_height <= 0:
        raise ValueError("resize dimensions must be positive integers")
    y_indices = np.linspace(0, height - 1, target_height, dtype=np.float32)
    x_indices = np.linspace(0, width - 1, target_width, dtype=np.float32)
    y_indices = np.clip(np.round(y_indices).astype(int), 0, height - 1)
    x_indices = np.clip(np.round(x_indices).astype(int), 0, width - 1)
    if image.ndim == 2:
        return image[y_indices[:, None], x_indices]
    return image[y_indices[:, None], x_indices, :]


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian kernel."""

    size = _ensure_odd(size)
    half = size // 2
    axis = np.arange(-half, half + 1, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D convolution using zero-padding."""

    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    if image.ndim == 2:
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        output = np.zeros_like(image, dtype=np.float32)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                region = padded[y : y + kernel.shape[0], x : x + kernel.shape[1]]
                output[y, x] = float(np.sum(region * kernel))
        return output

    channels = []
    for channel in range(image.shape[2]):
        channel_data = convolve(image[..., channel], kernel)
        channels.append(channel_data)
    return np.stack(channels, axis=-1)


def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply manual Gaussian blur to suppress noise."""

    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel)


def _bilateral_single_channel(image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Apply a bilateral filter to a single-channel image."""

    diameter = max(1, _ensure_odd(diameter))
    if diameter == 1 or sigma_color <= 0 or sigma_space <= 0:
        return image
    radius = diameter // 2
    padded = np.pad(image, radius, mode="edge")
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    spatial = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_space**2))
    result = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y : y + diameter, x : x + diameter]
            center = padded[y + radius, x + radius]
            color = np.exp(-((region - center) ** 2) / (2.0 * sigma_color**2))
            weights = spatial * color
            weighted_sum = float(np.sum(region * weights))
            normalization = float(np.sum(weights))
            if normalization == 0:
                result[y, x] = center
            else:
                result[y, x] = weighted_sum / normalization
    return result


def bilateral_filter(image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Apply a bilateral filter that preserves edges while smoothing textures."""

    if image.ndim == 2:
        return _bilateral_single_channel(image, diameter, sigma_color, sigma_space)
    channels = []
    for channel in range(image.shape[2]):
        channels.append(_bilateral_single_channel(image[..., channel], diameter, sigma_color, sigma_space))
    return np.stack(channels, axis=-1)


def preprocess(image: np.ndarray, config: PreprocessingConfig) -> PreprocessedImage:
    """Full preprocessing routine prior to crack detection."""

    working = image.astype(np.float32)
    if config.resize_to:
        working = resize_image(working, config.resize_to)
    if config.convert_to_grayscale:
        working = to_grayscale(working)
    if config.apply_bilateral:
        working = bilateral_filter(
            working,
            config.bilateral_kernel_size,
            config.bilateral_sigma_color,
            config.bilateral_sigma_space,
        )
    smoothed = gaussian_blur(working, config.gaussian_kernel_size, config.gaussian_sigma)
    return PreprocessedImage(base=working, smoothed=smoothed)
