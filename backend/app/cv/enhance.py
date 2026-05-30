"""CLAHE luminance enhancement."""

from __future__ import annotations

import cv2
import numpy as np


def clahe(
    frame_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Return a CLAHE-enhanced copy of ``frame_bgr``.

    The frame is converted to LAB, CLAHE is applied to the L channel only,
    and then converted back to BGR.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with 3 channels")

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe_op.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def gamma_correct(frame_bgr: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Optional secondary brightness correction. Used in the ablation study."""
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame_bgr, table)
