"""
Smoke tests for the SignSpeak CV pipeline.

These tests do not require any trained checkpoint: they create a synthetic
image in memory and exercise every CV module. Run with::

    pytest backend/tests
"""

from __future__ import annotations

import numpy as np
import pytest

from app.cv import edges, enhance, flow, morph, pipeline, segment
from app.cv.landmarks import LandmarkResult
from app.data import labels
from app.utils.smoothing import VotingBuffer


@pytest.fixture()
def dummy_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.integers(0, 255, size=(240, 320, 3))).astype(np.uint8)


def test_clahe_preserves_shape(dummy_frame: np.ndarray) -> None:
    out = enhance.clahe(dummy_frame)
    assert out.shape == dummy_frame.shape
    assert out.dtype == np.uint8


def test_selfie_mask_returns_uint8(dummy_frame: np.ndarray) -> None:
    mask = segment.selfie_mask(dummy_frame)
    assert mask.dtype == np.uint8
    assert mask.shape == dummy_frame.shape[:2]


def test_morphology_clean_preserves_shape(dummy_frame: np.ndarray) -> None:
    mask = np.zeros(dummy_frame.shape[:2], dtype=np.uint8)
    mask[50:150, 50:150] = 255
    cleaned = morph.clean(mask)
    assert cleaned.shape == mask.shape


def test_canny_edges(dummy_frame: np.ndarray) -> None:
    e = edges.canny(dummy_frame)
    assert e.shape == dummy_frame.shape[:2]
    assert 0.0 <= edges.edge_density(e) <= 1.0


def test_motion_analyser_starts_at_zero() -> None:
    m = flow.MotionAnalyser()
    assert m.update(None) == 0.0
    fake = [(float(i % 3) * 0.01, float(i // 3) * 0.01, 0.0) for i in range(21)]
    assert m.update(fake) == 0.0  # first frame -> no prev
    assert m.update(fake) >= 0.0


def test_voting_buffer_requires_quorum() -> None:
    b = VotingBuffer(window=10, majority=8, min_conf=0.5)
    for _ in range(7):
        assert b.push(3, 0.9) is None
    assert b.push(3, 0.9) is not None


def test_label_space_size() -> None:
    assert labels.NUM_CLASSES == 61
    assert labels.NUM_ASL + labels.NUM_ARSL == labels.NUM_CLASSES


def test_pipeline_runs_end_to_end(dummy_frame: np.ndarray) -> None:
    p = pipeline.Pipeline()
    out = p.run_frame(dummy_frame, language="auto")
    assert isinstance(out.stats.latency_ms, float)
    assert out.stats.latency_ms > 0
    # No checkpoint -> ensemble returns -1, but pipeline must not crash.
    assert out.label_id == -1 or out.label_id >= 0
