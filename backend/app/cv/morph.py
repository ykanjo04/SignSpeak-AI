"""
Binary morphological operations (CSCI435 capability 3).

Cleans the binary segmentation mask produced by ``segment.py``:
small holes inside the body are filled (closing) and small specks in
the background are removed (opening). Both are textbook operations
(Szeliski 2022, ch. 3.3.1).
"""

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
