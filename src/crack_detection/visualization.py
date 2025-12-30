"""Visualization helpers for crack detection results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .detector import CrackDetectionResult


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        normalized = np.clip(image, 0, 255).astype(np.float32) / 255.0
        return np.stack([normalized] * 3, axis=-1)
    return np.clip(image, 0, 255).astype(np.float32) / 255.0


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay a binary mask on top of the input image."""

    base = image.astype(np.float32)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    tint = np.zeros_like(base)
    tint[..., 2] = 255.0
    mask_norm = (mask.astype(np.float32) / 255.0)[..., None]
    overlay = base * (1.0 - alpha * mask_norm) + tint * (alpha * mask_norm)
    return np.clip(overlay, 0, 255)


def save_visualization(
    original: np.ndarray,
    processed: np.ndarray,
    result: CrackDetectionResult,
    destination: Path,
    title: Optional[str] = None,
) -> None:
    """Persist a multi-panel visualization summarizing the detection."""

    import matplotlib.pyplot as plt  # Imported lazily to avoid hard dependency during tests.

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(_to_rgb(original))
    axes[0].set_title("Original")
    axes[0].axis("off")

    cmap = "gray" if processed.ndim == 2 else None
    axes[1].imshow(np.clip(processed, 0, 255), cmap=cmap)
    axes[1].set_title("Preprocessed")
    axes[1].axis("off")

    overlay = overlay_mask(original, result.mask)
    axes[2].imshow(_to_rgb(overlay))
    axes[2].set_title("Detected Cracks")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
