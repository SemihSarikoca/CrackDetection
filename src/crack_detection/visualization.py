"""Visualization helpers for crack detection results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .detector import CrackDetectionResult


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay a binary mask on top of the input image."""

    if image.ndim == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    mask_color = np.zeros_like(image_color)
    mask_color[:, :, 2] = mask  # red channel overlay
    return cv2.addWeighted(image_color, 1 - alpha, mask_color, alpha, 0)


def save_visualization(
    original: np.ndarray,
    processed: np.ndarray,
    result: CrackDetectionResult,
    destination: Path,
    title: Optional[str] = None,
) -> None:
    """Persist a multi-panel visualization summarizing the detection."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    cmap = "gray" if processed.ndim == 2 else None
    axes[1].imshow(processed, cmap=cmap)
    axes[1].set_title("Preprocessed")
    axes[1].axis("off")

    overlay = overlay_mask(original, result.mask)
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Detected Cracks")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
