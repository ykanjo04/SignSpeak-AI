"""
SignSpeak AI FastAPI application entry point.

Run locally with:

    cd backend
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, upload, ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(ws.router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "SignSpeak AI backend",
        "docs": "/docs",
        "live_ws": "/ws/live",
        "upload": "/upload",
    }
