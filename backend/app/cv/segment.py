"""
Image segmentation (CSCI435 capability 2).

Uses Google's MediaPipe Selfie Segmentation model to produce a binary
foreground mask of the signer's body. The mask is used by ``morph.py``
to clean the boundary and to mask out background distractors before
keypoint detection.

The MediaPipe model itself is pre-trained and used as-is; the integration
inside our pipeline is the original contribution.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np


@lru_cache(maxsize=1)
def _selfie_segmenter():
    """Cached, thread-safe instance of the MediaPipe Selfie model.

    ``model_selection=1`` is the more accurate landscape model and works
    well for upper-body framing typical of a signer.
    """
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


def selfie_mask(frame_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a ``uint8`` foreground mask (255 = person, 0 = background)."""
    if frame_bgr.ndim != 3:
        raise ValueError("Expected BGR image")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = _selfie_segmenter().process(rgb)
    if result.segmentation_mask is None:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    soft = result.segmentation_mask
    return (soft > threshold).astype(np.uint8) * 255


def apply_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``frame_bgr`` with background replaced by neutral grey."""
    bg = np.full_like(frame_bgr, 127)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
    return np.where(mask3, frame_bgr, bg)
