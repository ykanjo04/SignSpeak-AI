"""
SignSpeak AI FastAPI application entry point.

Serves the API and the built Next.js frontend on one port (default 8000).

Run locally with::

    .\\scripts\\run_app.ps1

Or manually::

    cd frontend && npm run build
    cd ../backend
    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import health, upload, ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_OUT = PROJECT_ROOT / "frontend" / "out"

app = FastAPI(
    title="SignSpeak AI",
    description=(
        "Real-time American + Arabic Sign Language translator built for "
        "CSCI435 (Computer Vision Algorithms and Systems) at UOWD."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(ws.router)


@app.get("/api")
def api_root() -> dict[str, str]:
    return {
        "name": "SignSpeak AI",
        "docs": "/docs",
        "live_ws": "/ws/live",
        "upload": "/upload",
        "ui": "/",
    }


def _mount_frontend() -> None:
    if not FRONTEND_OUT.is_dir():
        logging.warning(
            "Frontend build not found at %s — UI unavailable. "
            "Run: cd frontend && npm run build",
            FRONTEND_OUT,
        )
        return
    app.mount("/", StaticFiles(directory=str(FRONTEND_OUT), html=True), name="frontend")
    logging.info("Serving frontend from %s", FRONTEND_OUT)


_mount_frontend()
