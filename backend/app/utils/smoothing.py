"""Temporal voting buffer and confidence gate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class Vote:
    label_id: int
    confidence: float


class VotingBuffer:
    """Rolling majority-vote buffer for stabilising live predictions."""

    def __init__(
        self,
        window: int = 10,
        majority: int = 8,
        min_conf: float = 0.85,
    ) -> None:
        if majority > window:
            raise ValueError("majority cannot exceed window")
        self.window = window
        self.majority = majority
        self.min_conf = min_conf
        self._buf: deque[Vote] = deque(maxlen=window)
        self._last_emitted: int | None = None

    def push(self, label_id: int, confidence: float) -> tuple[int, float] | None:
        """Add a new per-frame prediction and return the stable output, if any.

        Returns ``None`` if no class meets both the majority and confidence
        thresholds.
        """
        self._buf.append(Vote(label_id, confidence))
        if len(self._buf) < self.majority:
            return None
        # Count votes per class.
        tally: dict[int, list[float]] = {}
        for v in self._buf:
            tally.setdefault(v.label_id, []).append(v.confidence)
        # Best candidate.
        best_id, best_confs = max(tally.items(), key=lambda kv: len(kv[1]))
        if len(best_confs) < self.majority:
            return None
        avg_conf = float(sum(best_confs) / len(best_confs))
        if avg_conf < self.min_conf:
            return None
        return best_id, avg_conf

    def emit_new(self, label_id: int, confidence: float) -> tuple[int, float] | None:
        """Like ``push`` but only emit when the stable label changes.

        Used by the sentence builder so the same letter is not appended on
        every frame.
        """
        stable = self.push(label_id, confidence)
        if stable is None:
            return None
        if stable[0] == self._last_emitted:
            return None
        self._last_emitted = stable[0]
        return stable

    def reset(self) -> None:
        self._buf.clear()
        self._last_emitted = None
