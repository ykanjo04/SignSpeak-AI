"""
WebSocket endpoint - live webcam mode.

The frontend sends one JPEG-encoded frame per message (binary). For each
frame the per-frame pipeline runs and a JSON dictionary is sent back.

Protocol
--------

Client -> Server: raw JPEG bytes (binary frame).
Server -> Client: JSON text with the FrameResult dict from `pipeline.py`.

The client can also send a text JSON message ``{"language": "asl"|"arsl"|"auto"}``
at any time to switch the language filter.
"""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.cv.pipeline import Pipeline, result_to_dict

logger = logging.getLogger("signspeak.ws")
router = APIRouter()
_pipeline = Pipeline(device="cpu")


@router.websocket("/ws/live")
async def live_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    language = "auto"
    logger.info("WebSocket accepted")
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("text") is not None:
                try:
                    cfg = json.loads(msg["text"])
                except json.JSONDecodeError:
                    cfg = {}
                if "language" in cfg:
                    language = str(cfg["language"]).lower()
                    if language not in {"asl", "arsl", "auto"}:
                        language = "auto"
                    _pipeline.reset()
                continue

            data = msg.get("bytes")
            if not data:
                continue

            arr = np.frombuffer(data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            result = _pipeline.run_frame(frame_bgr, language=language)
            await websocket.send_text(json.dumps(result_to_dict(result)))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.exception("WebSocket error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
