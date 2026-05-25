"""
Edge detection (CSCI435 capability 6).

Runs Canny (Canny 1986) on the hand crop. The edge map is exposed as
an optional debug overlay and is also recorded during the ablation
study to show that landmark-based features outperform raw edge maps
for this task.
"""

from __future__ import annotations

import cv2
import numpy as np


def canny(
    frame_bgr: np.ndarray,
    low_thresh: int = 60,
    high_thresh: int = 150,
    blur_ksize: int = 5,
) -> np.ndarray:
    """Return a single-channel edge map of ``frame_bgr``."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(blurred, low_thresh, high_thresh)


def edge_density(edges: np.ndarray) -> float:
    """Fraction of pixels that are edge pixels. Used as a sanity check."""
    if edges.size == 0:
        return 0.0
    return float((edges > 0).sum()) / edges.size
