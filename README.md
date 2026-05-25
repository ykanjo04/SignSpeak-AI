# SignSpeak AI

Real-time **American Sign Language (ASL)** and **Arabic Sign Language (ArSL)**
fingerspelling translator. Built for CSCI435 (Computer Vision Algorithms and
Systems) at the University of Wollongong in Dubai, Spring 2026.

> A deaf student at UOWD opens SignSpeak AI in any browser, signs in ASL or
> ArSL, and the system instantly displays text on screen and reads it aloud
> through the browser's text-to-speech, letting them communicate with a hearing
> professor without an interpreter.

## What it does

SignSpeak AI integrates **eight** computer-vision capabilities into a single
deployable web application:

| # | Capability | Module |
|---|---|---|
| 1 | Image enhancement (CLAHE) | `backend/app/cv/enhance.py` |
| 2 | Image segmentation (MediaPipe Selfie) | `backend/app/cv/segment.py` |
| 3 | Binary morphological operations | `backend/app/cv/morph.py` |
| 4 | Keypoint detection (MediaPipe Holistic) | `backend/app/cv/landmarks.py` |
| 5 | Face detection (MediaPipe Face) | `backend/app/cv/landmarks.py` |
| 6 | Edge detection (Canny) | `backend/app/cv/edges.py` |
| 7 | Video processing (optical flow + temporal voting) | `backend/app/cv/flow.py` + `backend/app/utils/smoothing.py` |
| 8 | Object recognition (MobileNetV3 + MLP ensemble) | `backend/app/models/` |

## Architecture

```
Browser (Next.js + Tailwind)
  Webcam stream  ─►  WebSocket ─►  FastAPI backend
  Upload page    ─►  POST /upload ─►  FastAPI backend
                                       │
   ┌───────────────────────────────────┤
   │   Frame Pipeline (per-frame):     │
   │   CLAHE ─► Selfie Seg ─► Morph ─► │
   │   Holistic ─► Canny ─► Optical    │
   │   Flow ─► MLP ┐    ┌► MobileNetV3 │
   │               └►  Ensemble ──►    │
   │                Smoothing Buffer ──┘
   │                          │
   └──────────────────────────┴─►  {label, confidence, fps, landmarks}
                                          │
                                          ▼
                                  Frontend renders
                                  overlay + text + TTS
```

## Tech stack

- **Frontend:** Next.js 14, TypeScript, Tailwind CSS, Web Speech API
- **Backend:** FastAPI, Uvicorn, OpenCV, MediaPipe
- **ML:** PyTorch (MobileNetV3-Small fine-tuned + custom MLP)
- **Datasets:** [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle), [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) (Kaggle), [ArSL2018](https://huggingface.co/datasets/pain/ArASL_Database_Grayscale) (Hugging Face)

## Quick start

### Requirements

- Python **3.11**
- Node.js **20+**
- A webcam (built-in laptop cam is fine)
- Optional but recommended: a free [Kaggle](https://www.kaggle.com/) account for dataset downloads

### 1. Clone and bootstrap

```powershell
git clone https://github.com/ykanjo04/SignSpeak-AI.git
cd SignSpeak-AI

# Python venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

# Frontend deps
cd frontend
npm install
cd ..
```

### 2. Download datasets and train models (one-time, ~30–60 min)

```powershell
.\.venv\Scripts\Activate.ps1
python ml\scripts\download_datasets.py
python ml\scripts\extract_landmarks.py
python ml\scripts\train_mlp.py
python ml\scripts\train_mobilenet.py
python ml\scripts\evaluate.py
```

First run prompts for your Kaggle username + API key (ASL + MNIST only). Get one at
<https://www.kaggle.com/settings> → "Create New Token". ArSL2018 is fetched from
[Hugging Face](https://huggingface.co/datasets/pain/ArASL_Database_Grayscale) and
does not require Kaggle terms acceptance.

Trained checkpoints land in `backend/app/models/checkpoints/`.

### 3. Run the app (single port)

Everything — UI, WebSocket, and upload API — runs on **port 8000**:

```powershell
.\scripts\run_app.ps1
```

Open <http://localhost:8000> (live mode: <http://localhost:8000/live/>).

To skip rebuilding the frontend on subsequent starts:

```powershell
.\scripts\run_app.ps1 -SkipBuild
```

<details>
<summary>Legacy two-terminal dev mode (optional)</summary>

```powershell
# Terminal 1
.\.venv\Scripts\Activate.ps1
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend
$env:NEXT_PUBLIC_BACKEND_HTTP="http://127.0.0.1:8000"
$env:NEXT_PUBLIC_BACKEND_WS="ws://127.0.0.1:8000"
npm run dev
```

Open <http://localhost:3000>.
</details>

## Repository layout

```
SignSpeak-AI/
├── backend/         FastAPI server, CV pipeline, model wrappers
├── frontend/        Next.js client (webcam + UI + TTS)
├── ml/              Dataset scripts, training notebooks, results
├── report/          DOCX generator + final report
├── slides/          PPTX generator + defence slides
├── demo/            Defence script + recording guide
├── docs/            User story + architecture writeup
└── scripts/         Helper PowerShell scripts
```

## Project deliverables (CSCI435)

| Deliverable | Location |
|---|---|
| Source code | this repo |
| Trained models | `backend/app/models/checkpoints/` |
| Report (DOCX) | `report/SignSpeak_AI_Report.docx` |
| Defence slides | `slides/SignSpeak_AI_Defence.pptx` |
| Demo video | record locally per `demo/recording_guide.md` |
| User story | `docs/user-story.md` |

## Team

| Member | Role |
|---|---|
| Yousef Kanjo (placeholder ID) | Lead, architecture, ML |
| Member 2 (placeholder ID) | Frontend |
| Member 3 (placeholder ID) | Data + preprocessing |
| Member 4 (placeholder ID) | Report + diagrams |
| Member 5 (placeholder ID) | Demo + testing + slides |

## Academic integrity

All code in this repository is the original work of the SignSpeak AI team.
Pre-trained components (MediaPipe Holistic, MobileNetV3 ImageNet weights) and
publicly available datasets are cited in `report/SignSpeak_AI_Report.docx`,
Section 8: References.

## License

[MIT](LICENSE)
