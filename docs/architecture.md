# SignSpeak AI - System Architecture

## High-level

SignSpeak AI is a three-tier web application:

1. **Browser client** (Next.js + Tailwind) - captures webcam frames or
   accepts an uploaded video, renders the skeleton overlay, sentence
   builder, language toggle, and text-to-speech.
2. **Inference service** (FastAPI + WebSocket) - receives JPEG-encoded
   frames, runs the eight-stage CV pipeline, returns predictions as JSON.
3. **Model artefacts** (PyTorch state dicts under
   `backend/app/models/checkpoints/`) - one MLP and one MobileNetV3-Small
   fine-tuned on combined ASL + ArSL data.

```
   ┌────────────────────────────────────────────┐
   │   Browser (Next.js + Tailwind)             │
   │                                            │
   │   ┌────────────────────┐ ┌───────────────┐ │
   │   │ Webcam Stream      │ │ Upload Video  │ │
   │   │ (getUserMedia)     │ │ (drag-n-drop) │ │
   │   └─────────┬──────────┘ └─────────┬─────┘ │
   │             │                       │       │
   │             │ JPEG/15 FPS           │ mp4   │
   │             ▼                       ▼       │
   │      ┌──────────────┐         ┌──────────┐  │
   │      │ WS /ws/live  │         │POST /upl │  │
   │      └──────┬───────┘         └─────┬────┘  │
   │             │  JSON predictions    │       │
   │             ▼                       ▼       │
   │   ┌───────────────────────────────────────┐ │
   │   │ Hand+Face Skeleton Overlay (Canvas)   │ │
   │   │ Prediction Display                    │ │
   │   │ Sentence Builder                      │ │
   │   │ Language Toggle (ASL / ArSL / Auto)   │ │
   │   │ TTS (Web Speech API)                  │ │
   │   │ FPS + Latency stats                   │ │
   │   └───────────────────────────────────────┘ │
   └────────────────────────────────────────────┘
                    ▲
                    │ WebSocket / HTTPS
                    │
   ┌────────────────┴───────────────────────────┐
   │   FastAPI Backend (Python 3.11)            │
   │                                            │
   │   Per-frame pipeline:                      │
   │     1. CLAHE enhancement                   │
   │     2. Selfie segmentation                 │
   │     3. Erode / dilate morphology           │
   │     4. MediaPipe Holistic (hands+face+pose)│
   │     5. Canny edges (auxiliary)             │
   │     6. Optical flow (Lucas-Kanade)         │
   │     7. Landmark MLP inference              │
   │     8. MobileNetV3-Small inference         │
   │     9. Ensemble + smoothing buffer         │
   │                                            │
   │   Returns JSON:                            │
   │     {label, confidence, fps, latency_ms,   │
   │      hand_landmarks, face_landmarks}       │
   └────────────────────────────────────────────┘
                    ▲
                    │ load on startup
                    │
   ┌────────────────┴───────────────────────────┐
   │   Trained Model Checkpoints                │
   │   - mlp_static.pt    (~150 KB)             │
   │   - mobilenet_v3_static.pt   (~9 MB)       │
   └────────────────────────────────────────────┘
```

## Data flow timing (target)

| Stage | Target time per frame |
|---|---|
| JPEG decode | < 2 ms |
| CLAHE | < 3 ms |
| Selfie segmentation | < 8 ms |
| Morphology | < 1 ms |
| MediaPipe Holistic | < 15 ms |
| Canny | < 2 ms |
| Optical flow | < 4 ms |
| MLP inference | < 1 ms |
| MobileNetV3 inference | < 8 ms |
| Ensemble + smoothing | < 1 ms |
| Network + JSON round trip | < 10 ms |
| **Total end-to-end** | **< 60 ms (target), > 15 FPS sustainable** |

## Class label space

61 classes total = 29 ASL (A-Z plus space, delete, nothing) + 32 ArSL letters.
Label IDs are global so the same network head outputs predictions for both
languages; the language toggle simply masks predictions outside the chosen
script.

## Modes of operation

- **Live mode** - WebSocket loop, ~15 FPS, ~50 ms latency, used in the live
  demo.
- **Upload mode** - HTTP POST of mp4, server samples 5 FPS and returns a
  transcript with timestamps. Used to satisfy the "two input modalities"
  requirement of the CSCI435 brief.
