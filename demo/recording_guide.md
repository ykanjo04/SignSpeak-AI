# SignSpeak AI - Demo Video Recording Guide

The CSCI435 project brief asks for a **2-3 minute** demonstration video that captures the system in operation. This file is the recording recipe; follow it once to produce `demo/demo_video.mp4`, then submit it together with the report and the GitHub link.

## Tools

- **OBS Studio** (free, https://obsproject.com/) - the easiest way to capture screen + webcam + audio in one .mp4.
- **DaVinci Resolve** or **Shotcut** - optional trimming + adding subtitles.

## OBS scene layout

In OBS, create one Scene with three Sources stacked top to bottom in the canvas:

1. **Display Capture** of your full screen (we'll crop to the browser).
2. **Window Capture** of Chrome showing `http://localhost:3000/live` - placed on top so it's the visible layer.
3. **Audio Input** of the microphone.

Recording settings:

- Output format: **MP4**
- Resolution: **1920 x 1080**
- FPS: **30**
- Encoder: x264, CRF 18-22 (good quality, small file)
- Sample rate: 48 kHz

## Pre-recording checklist

1. Start the backend:
   ```powershell
   cd C:\Users\ykanj\Downloads\UOWD\Uni25-26\sem3_26\CSCI435\SignSpeak-AI
   .\.venv\Scripts\Activate.ps1
   cd backend
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
2. Start the frontend:
   ```powershell
   cd C:\Users\ykanj\Downloads\UOWD\Uni25-26\sem3_26\CSCI435\SignSpeak-AI\frontend
   npm run dev
   ```
3. Browser: open `http://localhost:3000`, allow webcam permission, navigate to **Live mode**.
4. Wear something with no busy patterns. Solid colour shirt is best.
5. Position the webcam so your face fills the top third of the frame and both hands fit comfortably.
6. Add side lighting (window or ring light). Avoid harsh backlighting.

## Recording script (~2:30)

| Time | What you say + do | Camera/screen |
|---|---|---|
| 0:00-0:15 | "Hi, I'm Yousef from the SignSpeak AI team for CSCI435. We built a real-time American + Arabic Sign Language translator that runs entirely in the browser." | Slide of title or About page |
| 0:15-0:35 | "Here is the live mode. The webcam streams to a FastAPI backend over WebSocket at 15 FPS. Every frame goes through eight CV stages - CLAHE enhancement, MediaPipe segmentation, morphology, holistic landmarks, Canny edges, optical flow, our custom MLP, and a fine-tuned MobileNetV3." | Live page; point at stats panel |
| 0:35-0:55 | Sign A, B, C, then spell HELLO. Click Speak. | Webcam + skeleton overlay |
| 0:55-1:15 | "And here is Arabic Sign Language - we trained on the ArSL2018 dataset." Toggle to ArSL. Sign 2 letters. Click Speak. | Webcam + Arabic glyphs |
| 1:15-1:35 | Navigate to Upload page. Drop a sample mp4. Show transcript appearing. | Upload page |
| 1:35-1:55 | "We have two trained models: a landmark MLP from scratch and a fine-tuned MobileNetV3-Small. The ensemble plus 8-of-10 voting keeps live predictions stable." | Quick switch to About page |
| 1:55-2:20 | "On the held-out test set we achieved X% accuracy on ASL, Y% on ArSL, and we sustain Z FPS at W ms latency on a CPU laptop." Show results screenshots if you have them. | Picture-in-picture of confusion matrices |
| 2:20-2:35 | "Source code, the trained checkpoints, the report, and the slides are all in our public repo. Thanks for watching." | About page with GitHub link |

## Editing

- Trim the silence at the start and end.
- Add a 1-second fade-in at 0:00 and fade-out at the end.
- Optional: burn-in subtitles or captions (improves accessibility and clarity).
- Export as `demo/demo_video.mp4` (single file, 1080p, 30 FPS).

## What if the live demo glitches mid-take?

Re-take. Two-three takes is normal. The marker watches the final video, not the takes.

## Backup option (no recording software)

Windows 10 / 11 has a built-in screen recorder accessible via `Win+G` (Xbox Game Bar). Quality is OK for a CSCI submission. Use this only if OBS gives you trouble - the video evidence is required.
