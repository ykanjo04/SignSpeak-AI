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

pip install -r backend\requirements.txt

cd frontend && npm install && cd ..
```

Open http://localhost:8000 — live mode at `/live/`, upload at `/upload/`.

Sample clips for testing: `demo/sample_asl.mp4`, `demo/sample_arsl.mp4`.

## Repository layout

```
SignSpeak-AI/
├── backend/       FastAPI server, CV pipeline, model checkpoints
├── frontend/      Next.js UI (live webcam, upload, TTS)
├── ml/            Dataset download, training, evaluation scripts
└── demo/          Sample videos and recording guide
```

## License

[MIT](LICENSE)
