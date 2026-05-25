"""
Keypoint detection + face detection (CSCI435 capabilities 4 and 5).

Wraps MediaPipe Holistic, which detects in a single forward pass:

- 21 hand landmarks per hand (left + right)
- 468 face landmarks
- 33 body pose landmarks

Only the hand landmarks are fed into our classifiers, but we expose all
of them through the dataclass so the frontend can render rich overlays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class LandmarkResult:
    """Outcome of running MediaPipe Holistic on one frame."""

    right_hand: list[tuple[float, float, float]] = field(default_factory=list)
    left_hand: list[tuple[float, float, float]] = field(default_factory=list)
    face: list[tuple[float, float, float]] = field(default_factory=list)
    pose: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def has_hand(self) -> bool:
        return bool(self.right_hand) or bool(self.left_hand)

    @property
    def active_hand(self) -> list[tuple[float, float, float]]:
        """Pick the hand that is most likely the signing hand (right by default)."""
        if self.right_hand:
            return self.right_hand
        return self.left_hand

    def to_feature_vector(self) -> np.ndarray | None:
        """Return a 63-dim wrist-normalised flat vector or ``None``."""
        hand = self.active_hand
        if not hand or len(hand) != 21:
            return None
        arr = np.asarray(hand, dtype=np.float32)         # (21, 3)
        wrist = arr[0].copy()
        arr -= wrist                                      # translate
        scale = float(np.linalg.norm(arr[12]) + 1e-6)     # middle-finger tip distance
        arr /= scale                                      # uniform scale
        return arr.flatten()                              # (63,)


@lru_cache(maxsize=1)
def _holistic():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _pack(landmark_list) -> list[tuple[float, float, float]]:
    if landmark_list is None:
        return []
    return [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]


def extract_holistic(frame_bgr: np.ndarray) -> LandmarkResult:
    """Run MediaPipe Holistic on one BGR frame and return all landmarks."""
    if frame_bgr.ndim != 3:
        raise ValueError("Expected BGR image")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    out = _holistic().process(rgb)
    return LandmarkResult(
        right_hand=_pack(out.right_hand_landmarks),
        left_hand=_pack(out.left_hand_landmarks),
        face=_pack(out.face_landmarks),
        pose=_pack(out.pose_landmarks),
    )


def crop_hand(
    frame_bgr: np.ndarray,
    landmarks: LandmarkResult,
    size: int = 96,
    margin: float = 0.25,
) -> np.ndarray | None:
    """Return a square hand crop suitable for MobileNetV3 inference.

    Returns ``None`` if no hand is detected.
    """
    hand = landmarks.active_hand
    if not hand:
        return None
    h, w = frame_bgr.shape[:2]
    xs = [p[0] for p in hand]
    ys = [p[1] for p in hand]
    x_min = max(0.0, min(xs) - margin)
    x_max = min(1.0, max(xs) + margin)
    y_min = max(0.0, min(ys) - margin)
    y_max = min(1.0, max(ys) + margin)
    x_min_px, x_max_px = int(x_min * w), int(x_max * w)
    y_min_px, y_max_px = int(y_min * h), int(y_max * h)
    if x_max_px <= x_min_px or y_max_px <= y_min_px:
        return None
    crop = frame_bgr[y_min_px:y_max_px, x_min_px:x_max_px]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))
