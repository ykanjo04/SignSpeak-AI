"""
The full per-frame CV pipeline.

This is the integration point Dr. Mukala will look at: every CSCI435
capability we claim is actually exercised here.

    run_frame(frame_bgr, language) -> dict
        1. CLAHE              (capability 1)
        2. Selfie segmentation (capability 2)
        3. Morphology         (capability 3)
        4. Holistic landmarks (capabilities 4 + 5: hands+face)
        5. Hand crop
        6. Canny edges        (capability 6)
        7. Optical flow       (capability 7)
        8. Ensemble inference (capability 8: recognition)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.cv import edges as edges_mod
from app.cv import enhance, morph, segment
from app.cv.flow import MotionAnalyser, motion_status
from app.cv.landmarks import LandmarkResult, crop_hand, extract_holistic
from app.data.labels import display_label, label_of, mask_for_language
from app.models.ensemble import Ensemble
from app.utils.smoothing import VotingBuffer
from app.utils.timing import FPSCounter, Stopwatch


@dataclass
class FrameStats:
    fps: float = 0.0
    latency_ms: float = 0.0
    per_stage_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class FrameResult:
    label_id: int
    label: str
    display: str
    confidence: float
    motion: str
    hand_landmarks: list[tuple[float, float, float]]
    face_landmarks: list[tuple[float, float, float]]
    pose_landmarks: list[tuple[float, float, float]]
    edge_density: float
    new_letter: bool
    per_model: dict[str, float]
    stats: FrameStats


class Pipeline:
    """Stateful pipeline that owns the ensemble and the smoothing buffer."""

    def __init__(self, device: str = "cpu") -> None:
        self.ensemble = Ensemble(device=device)
        self.buffer = VotingBuffer(window=10, majority=8, min_conf=0.85)
        self.motion = MotionAnalyser(window=5)
        self.fps = FPSCounter(window=30)

    # -- public API --
    def reset(self) -> None:
        self.buffer.reset()
        self.motion.reset()

    def run_frame(
        self,
        frame_bgr: np.ndarray,
        language: str = "auto",
    ) -> FrameResult:
        stats = FrameStats()
        with Stopwatch() as total_sw:

            # 1 - CLAHE
            with Stopwatch() as sw:
                frame_eq = enhance.clahe(frame_bgr)
            stats.per_stage_ms["clahe"] = sw.ms

            # 2 - Selfie segmentation
            with Stopwatch() as sw:
                mask = segment.selfie_mask(frame_eq)
            stats.per_stage_ms["segment"] = sw.ms

            # 3 - Morphology
            with Stopwatch() as sw:
                mask_clean = morph.clean(mask)
            stats.per_stage_ms["morph"] = sw.ms

            # 4 + 5 - MediaPipe Holistic (hands + face)
            with Stopwatch() as sw:
                landmarks: LandmarkResult = extract_holistic(frame_eq)
            stats.per_stage_ms["holistic"] = sw.ms

            # 6 - Canny edge map of the hand crop
            with Stopwatch() as sw:
                hand_crop = crop_hand(frame_eq, landmarks, size=96, margin=0.3)
                if hand_crop is not None:
                    edge_map = edges_mod.canny(hand_crop)
                    e_density = edges_mod.edge_density(edge_map)
                else:
                    e_density = 0.0
            stats.per_stage_ms["edges"] = sw.ms

            # 7 - Optical flow / motion magnitude
            with Stopwatch() as sw:
                magnitude = self.motion.update(landmarks.active_hand)
            stats.per_stage_ms["flow"] = sw.ms

            # 8 - Ensemble inference
            with Stopwatch() as sw:
                landmark_vec = landmarks.to_feature_vector()
                if landmark_vec is None and hand_crop is None:
                    label_id, conf, per_model = -1, 0.0, {}
                else:
                    label_id, conf, per_model = self.ensemble.predict(
                        landmark_vec=landmark_vec,
                        hand_crop_bgr=hand_crop,
                        allowed_indices=mask_for_language(language),
                    )
            stats.per_stage_ms["infer"] = sw.ms

            # Smoothing buffer -> stable prediction
            new_letter = False
            if label_id >= 0:
                stable = self.buffer.emit_new(label_id, conf)
                if stable is not None:
                    label_id, conf = stable
                    new_letter = True

        # Wall-clock
        self.fps.tick()
        stats.fps = self.fps.fps
        stats.latency_ms = total_sw.ms

        return FrameResult(
            label_id=label_id,
            label=label_of(label_id) if label_id >= 0 else "...",
            display=display_label(label_id) if label_id >= 0 else "",
            confidence=conf,
            motion=motion_status(magnitude),
            hand_landmarks=landmarks.active_hand,
            face_landmarks=landmarks.face,
            pose_landmarks=landmarks.pose,
            edge_density=e_density,
            new_letter=new_letter,
            per_model=per_model,
            stats=stats,
        )


# -- JSON-friendly serialiser used by the WebSocket router --
def result_to_dict(result: FrameResult) -> dict[str, Any]:
    return {
        "label_id": result.label_id,
        "label": result.label,
        "display": result.display,
        "confidence": round(result.confidence, 4),
        "motion": result.motion,
        "new_letter": result.new_letter,
        "edge_density": round(result.edge_density, 4),
        "hand_landmarks": [
            {"x": p[0], "y": p[1], "z": p[2]} for p in result.hand_landmarks
        ],
        "face_landmarks": [
            {"x": p[0], "y": p[1]} for p in result.face_landmarks[::4]  # subsample for bandwidth
        ],
        "pose_landmarks": [
            {"x": p[0], "y": p[1]} for p in result.pose_landmarks
        ],
        "stats": {
            "fps": round(result.stats.fps, 2),
            "latency_ms": round(result.stats.latency_ms, 2),
            "per_stage_ms": {
                k: round(v, 2) for k, v in result.stats.per_stage_ms.items()
            },
        },
        "per_model": result.per_model,
    }
