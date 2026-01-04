"""Concrete crack detection strategies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Set, Tuple

import numpy as np

from .config import DetectorConfig
from .preprocessing import PreprocessedImage, convolve, gaussian_blur


@dataclass(slots=True)
class CrackDetectionResult:
    """Output produced by a crack detector."""

    mask: np.ndarray
    score_map: np.ndarray
    summary: Dict[str, float]


@dataclass(slots=True)
class ComponentStats:
    """Descriptive statistics for a labeled connected component."""

    label: int
    area: int
    bbox: Tuple[int, int, int, int]
    width: int
    height: int
    thickness: int
    length: float
    perimeter: int
    orientation: float
    coverage_height: float
    coverage_width: float


class CrackDetector(Protocol):
    """Protocol for crack detector implementations."""

    def detect(self, image: PreprocessedImage) -> CrackDetectionResult:
        ...


class CannyCrackDetector:
    """Manual implementation of the Canny edge detector with multi-stage filtering."""

    STRONG_VALUE = 255.0

    def __init__(self, config: DetectorConfig) -> None:
        if config.high_threshold <= config.low_threshold:
            raise ValueError("high_threshold must be greater than low_threshold")
        self._config = config

    def detect(self, image: PreprocessedImage) -> CrackDetectionResult:
        heavy_input = image.smoothed
        light_input = gaussian_blur(
            image.base,
            self._config.light_gaussian_kernel_size,
            self._config.light_gaussian_sigma,
        )
        if heavy_input.ndim != 2 or light_input.ndim != 2:
            raise ValueError("Canny detector expects single-channel grayscale inputs.")

        # Stage 1: classic Canny at two smoothing scales.
        heavy_mask, heavy_suppressed = _run_canny(
            heavy_input,
            low=self._config.low_threshold,
            high=self._config.high_threshold,
            weak=self._config.weak_pixel_value,
            strong=self.STRONG_VALUE,
        )
        light_mask, _ = _run_canny(
            light_input,
            low=self._config.thin_low_threshold,
            high=self._config.thin_high_threshold,
            weak=self._config.weak_pixel_value,
            strong=self.STRONG_VALUE,
        )

        # Stage 2: analyze connected components from the heavy mask (main signal path).
        labels, stats = _label_components(heavy_mask)
        variance_map = _local_variance(image.base, self._config.texture_kernel_size)
        linear_strength = _linear_response(heavy_mask > 0, self._config.grid_kernel_size)

        # Stage 3: classify which components belong to grout/grid structures or textured clutter.
        grid_labels = (
            _select_grid_components(stats, labels, linear_strength, self._config) if self._config.suppress_grid_lines else set()
        )
        texture_labels = _select_texture_components(stats, labels, variance_map, self._config)

        # Stage 4: remember thin, long cracks so they survive even if they overlap grout masks.
        thin_labels = _select_thin_components(stats, self._config)

        # Stage 5: build the refined mask by retaining eligible components and re-introducing thin cracks.
        refined = np.zeros_like(heavy_mask, dtype=np.float32)
        for component in stats:
            label = component.label
            if label in thin_labels:
                refined[labels == label] = self.STRONG_VALUE
                continue
            if component.area < self._config.min_crack_area:
                continue
            if label in grid_labels or label in texture_labels:
                continue
            refined[labels == label] = self.STRONG_VALUE

        # Stage 6: rescue thin cracks detected only at the light blur scale.
        if np.any(light_mask):
            light_labels, light_stats = _label_components(light_mask)
            rescued = _select_thin_components(light_stats, self._config)
            for component in light_stats:
                if component.label not in rescued:
                    continue
                refined[light_labels == component.label] = self.STRONG_VALUE

        mask_uint8 = refined.astype(np.uint8)
        normalized = _normalize_map(heavy_suppressed)
        crack_pixels = float(np.count_nonzero(mask_uint8))
        coverage = crack_pixels / mask_uint8.size
        return CrackDetectionResult(
            mask=mask_uint8,
            score_map=normalized,
            summary={
                "crack_coverage": coverage,
                "crack_pixels": crack_pixels,
                "total_pixels": float(mask_uint8.size),
            },
        )


def _compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve(image, sobel_x)
    gy = convolve(image, sobel_y)
    magnitude = np.hypot(gx, gy)
    magnitude = _normalize_to_255(magnitude)
    direction = np.rad2deg(np.arctan2(gy, gx))
    direction = (direction + 180) % 180
    return magnitude, direction


def _normalize_to_255(values: np.ndarray) -> np.ndarray:
    max_val = np.max(values)
    if max_val == 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values / max_val) * 255.0


def _non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    rows, cols = magnitude.shape
    output = np.zeros((rows, cols), dtype=np.float32)
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            angle = direction[y, x]
            current = magnitude[y, x]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = (magnitude[y, x + 1], magnitude[y, x - 1])
            elif 22.5 <= angle < 67.5:
                neighbors = (magnitude[y + 1, x - 1], magnitude[y - 1, x + 1])
            elif 67.5 <= angle < 112.5:
                neighbors = (magnitude[y + 1, x], magnitude[y - 1, x])
            else:
                neighbors = (magnitude[y - 1, x - 1], magnitude[y + 1, x + 1])
            if current >= neighbors[0] and current >= neighbors[1]:
                output[y, x] = current
    return output


def _double_threshold(
    image: np.ndarray,
    *,
    low: float,
    high: float,
    weak_value: float,
    strong_value: float,
) -> np.ndarray:
    result = np.zeros_like(image, dtype=np.float32)
    strong_indices = image >= high
    weak_indices = (image >= low) & (image < high)
    result[strong_indices] = strong_value
    result[weak_indices] = weak_value
    return result


def _hysteresis(image: np.ndarray, *, weak: float, strong: float) -> np.ndarray:
    rows, cols = image.shape
    result = image.copy()
    strong_points = deque(zip(*np.where(result == strong)))
    visited = np.zeros_like(result, dtype=bool)
    visited[result == strong] = True

    while strong_points:
        y, x = strong_points.pop()
        for ny in range(max(0, y - 1), min(rows, y + 2)):
            for nx in range(max(0, x - 1), min(cols, x + 2)):
                if visited[ny, nx]:
                    continue
                if result[ny, nx] == weak:
                    result[ny, nx] = strong
                    visited[ny, nx] = True
                    strong_points.append((ny, nx))
    result[result != strong] = 0.0
    return result


def _run_canny(image: np.ndarray, *, low: float, high: float, weak: float, strong: float) -> Tuple[np.ndarray, np.ndarray]:
    """Execute the differentiable stages of Canny and return the binary mask plus suppressed map."""

    grad_mag, grad_dir = _compute_gradients(image)
    suppressed = _non_maximum_suppression(grad_mag, grad_dir)
    thresholded = _double_threshold(suppressed, low=low, high=high, weak_value=weak, strong_value=strong)
    mask = _hysteresis(thresholded, weak=weak, strong=strong)
    return mask, suppressed


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, List[ComponentStats]]:
    """Label connected components and compute descriptive statistics for each."""

    rows, cols = mask.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    stats: List[ComponentStats] = []
    label_id = 1
    binary = mask > 0
    neighbors_8 = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    axis_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(rows):
        for x in range(cols):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            queue: deque[Tuple[int, int]] = deque([(y, x)])
            coords: List[Tuple[int, int]] = []
            perimeter = 0
            while queue:
                cy, cx = queue.pop()
                if labels[cy, cx] != 0:
                    continue
                labels[cy, cx] = label_id
                coords.append((cy, cx))
                for dy, dx in neighbors_8:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < rows and 0 <= nx < cols and binary[ny, nx] and labels[ny, nx] == 0:
                        queue.append((ny, nx))
                for dy, dx in axis_neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= rows or nx < 0 or nx >= cols or not binary[ny, nx]:
                        perimeter += 1
            stats.append(_build_component_stats(label_id, coords, perimeter, rows, cols))
            label_id += 1
    return labels, stats


def _build_component_stats(
    label: int,
    coords: Sequence[Tuple[int, int]],
    perimeter: int,
    rows: int,
    cols: int,
) -> ComponentStats:
    ys = np.array([pt[0] for pt in coords], dtype=np.int32)
    xs = np.array([pt[1] for pt in coords], dtype=np.int32)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    height = int(y_max - y_min + 1)
    width = int(x_max - x_min + 1)
    thickness = int(max(1, min(height, width)))
    length = float(np.hypot(height, width))

    orientation = 0.0
    if len(coords) >= 2:
        coords_arr = np.column_stack((ys, xs)).astype(np.float32)
        centered = coords_arr - coords_arr.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]
        orientation = float(abs(np.degrees(np.arctan2(major_axis[0], major_axis[1]))))

    area = len(coords)
    coverage_height = height / rows
    coverage_width = width / cols
    bbox = (y_min, y_max, x_min, x_max)
    return ComponentStats(
        label=label,
        area=area,
        bbox=bbox,
        width=width,
        height=height,
        thickness=thickness,
        length=length,
        perimeter=perimeter,
        orientation=orientation,
        coverage_height=coverage_height,
        coverage_width=coverage_width,
    )


def _local_variance(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Estimate local variance using separable box filters."""

    if kernel_size <= 1:
        return np.zeros_like(image, dtype=np.float32)
    kernel_size = max(1, int(kernel_size))
    mean = _box_filter(image, kernel_size)
    mean_sq = _box_filter(image**2, kernel_size)
    variance = mean_sq - mean**2
    return np.clip(variance, 0.0, None)


def _box_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    temp = _sliding_mean(image, kernel_size, axis=1)
    return _sliding_mean(temp, kernel_size, axis=0)


def _sliding_mean(values: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Compute a 1D sliding mean along the selected axis using cumulative sums."""

    size = max(1, int(size))
    if size == 1:
        return values.astype(np.float32)
    pad = size // 2
    if axis == 1:
        padded = np.pad(values, ((0, 0), (pad, pad)), mode="edge")
        cumsum = np.cumsum(padded, axis=1, dtype=np.float64)
        cumsum = np.concatenate([np.zeros_like(cumsum[:, :1]), cumsum], axis=1)
        window = cumsum[:, size:] - cumsum[:, :-size]
    else:
        padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
        cumsum = np.cumsum(padded, axis=0, dtype=np.float64)
        cumsum = np.concatenate([np.zeros_like(cumsum[:1, :]), cumsum], axis=0)
        window = cumsum[size:, :] - cumsum[:-size, :]
    return (window / size).astype(np.float32)


def _linear_response(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Score each pixel by how line-like its local neighborhood is."""

    if kernel_size <= 1:
        return binary_mask.astype(np.float32)
    horizontal = _sliding_mean(binary_mask.astype(np.float32), kernel_size, axis=1)
    vertical = _sliding_mean(binary_mask.astype(np.float32), kernel_size, axis=0)
    return np.maximum(horizontal, vertical)


def _select_grid_components(
    stats: Sequence[ComponentStats],
    labels: np.ndarray,
    linear_response: np.ndarray,
    config: DetectorConfig,
) -> Set[int]:
    """Identify grout/tile seams that should be suppressed."""

    grid_labels: Set[int] = set()
    for component in stats:
        if component.coverage_height >= config.grid_row_density_threshold or component.coverage_width >= config.grid_column_density_threshold:
            grid_labels.add(component.label)
            continue
        coverage = max(component.coverage_height, component.coverage_width)
        if coverage >= config.max_component_axis_ratio:
            grid_labels.add(component.label)
            continue
        axis_ratio = max(component.width, component.height) / max(1, component.thickness)
        orientation_distance = min(abs(component.orientation), abs(90.0 - component.orientation))
        response = _component_mean(linear_response, labels, component.label)
        compactness = component.perimeter / max(1, component.area)
        if orientation_distance <= config.grid_orientation_tolerance:
            if axis_ratio >= config.grid_axis_ratio_threshold and response >= config.grid_kernel_response:
                grid_labels.add(component.label)
                continue
        if compactness >= config.grid_perimeter_area_threshold and response >= config.grid_kernel_response:
            grid_labels.add(component.label)
    return grid_labels


def _select_texture_components(
    stats: Sequence[ComponentStats],
    labels: np.ndarray,
    variance_map: np.ndarray,
    config: DetectorConfig,
) -> Set[int]:
    """Flag small, high-variance blobs produced by textured tiles."""

    noise_labels: Set[int] = set()
    for component in stats:
        if component.length >= config.texture_min_length:
            continue
        variance = _component_mean(variance_map, labels, component.label)
        if variance >= config.texture_variance_threshold:
            noise_labels.add(component.label)
    return noise_labels


def _select_thin_components(stats: Sequence[ComponentStats], config: DetectorConfig) -> Set[int]:
    """Remember slender, elongated components so they survive aggressive filtering."""

    thin_labels: Set[int] = set()
    for component in stats:
        if component.length < config.thin_crack_min_length:
            continue
        if component.thickness > config.thin_crack_max_width:
            continue
        orientation_distance = min(abs(component.orientation), abs(90.0 - component.orientation))
        if orientation_distance <= config.grid_orientation_tolerance:
            continue
        thin_labels.add(component.label)
    return thin_labels


def _component_mean(values: np.ndarray, labels: np.ndarray, label: int) -> float:
    mask = labels == label
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def _normalize_map(values: np.ndarray) -> np.ndarray:
    max_val = np.max(values)
    if max_val == 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values / max_val).astype(np.float32)
