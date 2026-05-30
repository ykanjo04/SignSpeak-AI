"""POST /upload — process a video and return a JSON transcript."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.cv.pipeline import Pipeline
from app.data.labels import display_label
from app.utils.smoothing import VotingBuffer

router = APIRouter()
# Upload uses looser smoothing than live webcam (fewer sampled frames per sign).
_pipeline = Pipeline(device="cpu")
_pipeline.buffer = VotingBuffer(window=5, majority=2, min_conf=0.42)

SAMPLE_FPS = 5  # process this many frames per second of video
SCENE_CHANGE_THRESHOLD = 18.0  # mean pixel diff -> new sign segment


def _scene_changed(prev_gray: np.ndarray | None, frame_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is None or prev_gray.shape != gray.shape:
        return False
    return float(np.mean(cv2.absdiff(prev_gray, gray))) >= SCENE_CHANGE_THRESHOLD


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    language: str = Form("auto"),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix or ".mp4"
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not decode video")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(video_fps / SAMPLE_FPS)))

    _pipeline.reset()
    t0 = time.perf_counter()
    transcript: list[dict[str, Any]] = []
    text_parts: list[str] = []
    n_processed = 0
    frame_idx = 0
    prev_gray: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            if _scene_changed(prev_gray, frame):
                _pipeline.reset()
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = _pipeline.run_frame(frame, language=language)
            n_processed += 1
            if result.new_letter and result.label_id >= 0:
                t_s = frame_idx / video_fps
                transcript.append(
                    {
                        "t_s": round(t_s, 2),
                        "label": result.label,
                        "display": result.display,
                        "confidence": round(result.confidence, 4),
                    }
                )
                text_parts.append(display_label(result.label_id))
        frame_idx += 1

    cap.release()
    duration = time.perf_counter() - t0
    return {
        "frames_processed": n_processed,
        "duration_s": round(duration, 2),
        "transcript": transcript,
        "text": "".join(text_parts),
        "language": language,
    }
