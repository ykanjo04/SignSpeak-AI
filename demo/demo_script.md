# SignSpeak AI - Live Defence Script (Week 11)

Time allocation per CSCI435 project brief: **7-8 minutes live demo + 4 minutes Q&A**. Every group member must speak. Rehearse this script three times before the defence day.

## Pre-demo checklist (5 min before defence)

- Power on the demo laptop, plug into the projector (use HDMI not USB-C if possible).
- Close all webcam-using apps (Teams, Zoom, Discord, OBS).
- Open two terminals only if you need to debug; normal demo uses one command:
  ```powershell
  cd C:\Users\ykanj\Downloads\UOWD\Uni25-26\sem3_26\CSCI435\SignSpeak-AI
  .\scripts\run_app.ps1
  ```
- Open Chrome and navigate to `http://localhost:8000`.
- Allow webcam permission once; verify the live page loads.
- Plug in the ring light (or open a bright window). Off-axis side-lighting works best.
- Have a second laptop on the table with the pre-recorded `demo_video.mp4` already open in VLC, paused on frame 1 - this is the backup if the live demo fails.

## Demo script (7-8 minutes)

### Beat 1: Hook + user story (Member 1 - Yousef) - 60 s
> "Good afternoon, Dr. Mukala. I'm Yousef. Imagine a deaf student at UOWD who can't get an interpreter for an urgent office-hour question. Today we'll demo SignSpeak AI - a browser app that translates American and Arabic Sign Language fingerspelling to text and speech, in real time, on this laptop, with no specialised hardware."

Click `Start live mode`. Webcam preview + skeleton overlay appears.

### Beat 2: Show eight CV capabilities running live (Member 1) - 60 s
> "Every frame from this webcam runs through eight CV stages - CLAHE enhancement, MediaPipe selfie segmentation, morphological cleanup, hand and face keypoint detection, Canny edges, optical flow, our custom landmark MLP, and a fine-tuned MobileNetV3. The CSCI435 brief asked for four; we integrated eight. The stats panel on the bottom-right shows the per-stage timing live."

Point at the stats panel; it should read about 25 FPS and 40-60 ms total.

### Beat 3: ASL fingerspelling (Member 2 - Frontend) - 90 s
Member 2 takes over at the laptop.
> "Let me show fingerspelling in ASL."

Sign letters slowly: `A`, `B`, `C`. The Detected sign panel updates; the confidence bar goes above 95%.
> "Now a word."

Spell `H E L L O`. Sentence builder accumulates `HELLO`.

Click **Speak**. The browser TTS says "Hello".

> "The 8-of-10 voting buffer is why predictions are stable even though my fingers move continuously."

### Beat 4: Arabic Sign Language (Member 3 - Data) - 75 s
Member 3 takes over.
> "Most public sign-language tools only support ASL. We extended the same model to Arabic Sign Language using the ArSL2018 dataset, which gives us 32 Arabic letters."

Click the language toggle to **ArSL**. Sign two Arabic letters: `aleff` (vertical hand) and `bb`. The Arabic glyphs appear in the prediction display.

Click **Speak**; TTS reads them in Arabic if `ar-SA` voice is installed.

### Beat 5: Second input modality - upload (Member 4 - Report) - 60 s
Member 4 takes over.
> "The CSCI435 brief requires at least two input modalities. We support a live webcam stream and an uploaded video."

Navigate to the **Upload** page. Drag in `demo/sample_video.mp4`. Click **Transcribe**. Timeline of letters appears with timestamps and confidences.

### Beat 6: Robustness demonstration (Member 5 - Demo) - 60 s
Member 5 takes over.
> "The judges' lighting is different from our office, so we did three things to make the model robust: CLAHE preprocessing, landmark-based features that are lighting-invariant, and a confidence-gated voting buffer."

Dim the room or cover the ring light briefly. Sign a letter. The model still recognises it. Restore lights.

### Beat 7: Architecture and metrics (Member 1) - 60 s
Show the **About** page or the slide with the architecture diagram.
> "On the backend we have a FastAPI WebSocket service. The architecture diagram shows the per-frame pipeline; the trained models are a custom MLP from scratch and a fine-tuned MobileNetV3-Small. On the test set we hit X% on ASL, Y% on ArSL, and we sustain Z FPS at W ms end-to-end latency."

Fill in real numbers from `ml/results/accuracy_table.csv` and `ml/results/fps_latency.csv` before the defence.

### Beat 8: Close + future work (rotating speaker) - 30 s
> "Thank you. We'd love to extend SignSpeak AI to continuous word-level signing using WLASL, and to mobile via TensorFlow Lite. We're open to questions."

## Q&A bank (rehearse these answers)

| Likely question | Pre-rehearsed answer |
|---|---|
| "Which model is fine-tuned?" | MobileNetV3-Small. Backbone frozen except the last conv block. Classifier head replaced with a 61-output linear layer. |
| "Which model is trained from scratch on custom data?" | The landmark MLP. We curated the data ourselves by running MediaPipe Hands over every image in ASL Alphabet and ArSL2018 to produce 63-D wrist-normalised vectors. |
| "Why landmarks AND MobileNet, not one or the other?" | Landmarks are lighting-invariant but lose the texture cues that disambiguate (M, N) and (S, T). MobileNet picks those up. The ensemble averages the two. |
| "What's the smoothing buffer for?" | Without it, per-frame predictions flicker during the moments when fingers transition between letters. The 8-of-10 vote with a 0.85 confidence floor only emits a letter when the system has been confident in the same answer for ~0.5 seconds. |
| "What if your webcam fails on stage?" | We have a backup laptop next to us with the demo video pre-loaded; we can also demonstrate the second modality - the Upload page - using `demo/sample_video.mp4`. Either way the live system stays exercised. |
| "Do you support continuous sentence-level signs?" | Not yet. The current model is fingerspelling-only. Continuous word-level signing is in our future work using WLASL with a temporal model. |
| "What's the FPS you measured?" | About X FPS sustained, measured on the synthetic 480p frame benchmark in `ml/scripts/evaluate.py`. Real-world webcam ranges from X-Y FPS depending on background complexity. |
| "How does the language toggle work?" | The model outputs a softmax over 61 classes - 29 ASL + 32 ArSL. The toggle masks out the indices outside the chosen language before the argmax. |
| "Where is your custom dataset stored?" | We don't ship a hand-collected dataset. Our 'custom data' is the curated 63-D landmark dataset that we generate by running MediaPipe Hands ourselves over the public ASL Alphabet and ArSL2018 image sets; the curation script lives in `ml/scripts/extract_landmarks.py`. The script reproducibly regenerates the dataset from the source images. |
| "What did you train on - and how long?" | Our MLP trained for 20 epochs (~7 minutes CPU-only); MobileNet for 5 epochs (~35 minutes CPU-only). Both checkpoints are committed to the repo at `backend/app/models/checkpoints/`. |

## After the defence

- Push any minor edits to `main` with a clear commit message.
- Add screenshots of the defence room to the report's Section 4.6.
- Submit the DOCX + repo link + demo mp4 to Moodle one day before the defence.
