# SignSpeak AI

Browser-based sign language fingerspelling translator for **American Sign Language (ASL)** and **Arabic Sign Language (ArSL)**. Use a live webcam or upload a short video; the app runs a computer-vision pipeline and shows recognised letters on screen, with optional text-to-speech.

**Two input modes:** live webcam (WebSocket) and video upload (REST).

## Requirements

- Python **3.11**
- Node.js **20+**
- Webcam (for live mode)
- Optional: [Kaggle](https://www.kaggle.com/) API token (ASL/MNIST datasets only)

## Quick start

```powershell
git clone https://github.com/ykanjo04/SignSpeak-AI.git
cd SignSpeak-AI

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

cd frontend && npm install && cd ..
```

Train models once (downloads data, ~30–60 min):

```powershell
python ml\scripts\download_datasets.py
python ml\scripts\extract_landmarks.py
python ml\scripts\train_mlp.py
python ml\scripts\train_mobilenet.py
```

Run the app (UI + API on **port 8000**):

```powershell
.\scripts\run_app.ps1
```

Open http://localhost:8000 — live mode at `/live/`, upload at `/upload/`.

Sample clips for testing: `demo/sample_asl.mp4`, `demo/sample_arsl.mp4`.

## Repository layout

```
SignSpeak-AI/
├── backend/       FastAPI server, CV pipeline, model checkpoints
├── frontend/      Next.js UI (live webcam, upload, TTS)
├── ml/            Dataset download, training, evaluation scripts
├── demo/          Sample videos and recording guide
├── report/        Project report (DOCX)
├── slides/        Defence slides (PPTX)
├── docs/          Architecture and user story notes
└── scripts/       Run helpers (e.g. run_app.ps1)
```

## License

[MIT](LICENSE)
