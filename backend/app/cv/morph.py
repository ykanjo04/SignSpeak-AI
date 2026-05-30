"""Morphological opening/closing on the segmentation mask."""

from __future__ import annotations

import cv2
import numpy as np


_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def clean(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply opening followed by closing to clean a binary mask."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, _KERNEL, iterations=iterations)
    return closed


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return cv2.erode(mask, _KERNEL, iterations=iterations)


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return cv2.dilate(mask, _KERNEL, iterations=iterations)
