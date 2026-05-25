"""
Simple FPS and end-to-end latency tracker used by the CV pipeline.
The numbers it produces feed both the stats panel in the UI and the
performance table in the report.
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Sliding-window FPS estimator."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / dt


class Stopwatch:
    """Measures the wall-clock duration of a code block in milliseconds."""

    def __init__(self) -> None:
        self._t0: float = 0.0
        self._elapsed_ms: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._elapsed_ms = (time.perf_counter() - self._t0) * 1000.0

    @property
    def ms(self) -> float:
        return self._elapsed_ms
