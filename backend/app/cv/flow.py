"""Optical flow on hand landmarks for motion gating."""

from __future__ import annotations

from collections import deque

import numpy as np


class MotionAnalyser:
    """Stateful tracker for hand motion between consecutive frames."""

    def __init__(self, window: int = 5) -> None:
        self._prev: np.ndarray | None = None
        self._mags: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._prev = None
        self._mags.clear()

    def update(self, hand_landmarks: list[tuple[float, float, float]] | None) -> float:
        """Return the average motion magnitude over the recent window.

        ``hand_landmarks`` must be the 21-point list from MediaPipe; if it is
        ``None`` or empty the magnitude is reset to zero.
        """
        if not hand_landmarks or len(hand_landmarks) != 21:
            self._prev = None
            return 0.0
        cur = np.asarray(hand_landmarks, dtype=np.float32)[:, :2]
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur
            return 0.0
        diff = cur - self._prev
        mag = float(np.linalg.norm(diff, axis=1).mean())
        self._prev = cur
        self._mags.append(mag)
        return float(np.mean(self._mags))

    @property
    def is_stable(self, threshold: float = 0.012) -> bool:
        """``True`` if recent motion is below the stability threshold."""
        if not self._mags:
            return False
        return float(np.mean(self._mags)) < threshold


def motion_status(magnitude: float, stable_threshold: float = 0.012) -> str:
    """Return ``"stable"`` or ``"moving"`` for UI feedback."""
    return "stable" if magnitude < stable_threshold else "moving"
