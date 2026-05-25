# SignSpeak AI - User Story

## Primary persona

**Aisha**, 21, third-year Computer Science student at the University of
Wollongong in Dubai. Aisha is profoundly deaf and communicates primarily in
**Arabic Sign Language (ArSL)** with family and **American Sign Language
(ASL)** with international classmates. She is fluent in lip-reading Arabic and
English but finds it exhausting during long technical discussions.

## Scenario

Aisha needs to discuss her final-year project with **Dr. Patrick Mukala**
during office hours. Her usual sign-language interpreter is unavailable that
afternoon, and the conversation involves technical terminology that is hard
to lip-read in real time.

Aisha visits Dr. Mukala's office. On the office PC she opens
<http://localhost:3000> in any modern browser and clicks **Start Live Mode**.
The page asks for webcam permission; she allows it.

The SignSpeak AI dashboard appears:

1. A live webcam feed with a coloured skeleton overlay drawn on top of her
   hands and face (MediaPipe Holistic keypoints).
2. A large prediction panel showing the currently detected letter and its
   confidence.
3. A sentence builder below it that accumulates letters as Aisha signs.
4. A toggle to switch between **ASL Mode**, **ArSL Mode**, and **Auto**.
5. A **Speak** button that reads the sentence aloud using the browser's
   built-in text-to-speech.

Aisha sets the language to **ArSL** because she finds Arabic more natural for
expressing her ideas. She begins signing letter by letter:

`...HELLO DR PATRICK I HAVE A QUESTION ABOUT MY PROJECT...`

Each sign is captured by the webcam at 15 frames per second, sent over a
WebSocket to the FastAPI backend, where the per-frame pipeline runs:

1. **CLAHE** boosts the contrast against the office's mixed lighting.
2. **MediaPipe Selfie Segmentation** isolates her from the background.
3. **Erode/Dilate** morphology cleans the mask edges.
4. **MediaPipe Holistic** extracts 21 hand keypoints, 468 face keypoints, and
   33 body keypoints.
5. **Canny edge** map of the hand region is computed as an auxiliary signal.
6. **Optical flow** between consecutive frames detects whether Aisha is
   actively signing or holding a pose.
7. The 63-D hand-landmark vector is sent to a **custom MLP** (trained from
   scratch on landmarks extracted from the ASL Alphabet and ArSL2018 datasets).
8. The same hand crop is also passed through a **MobileNetV3-Small fine-tuned**
   on ImageNet + ASL + ArSL.
9. An **ensemble** averages the two model probabilities, and a **temporal
   voting buffer** requires the same letter to win 8 of the last 10 frames
   before it is shown on screen.

The on-screen text updates instantly as her sign becomes confident. The
sentence builder accumulates each new letter. When she finishes a sentence she
presses **Speak**; the browser TTS reads the Arabic text aloud, and Dr. Mukala
responds verbally.

Throughout the session a small **stats panel** in the corner shows that the
system is running at **~25 FPS** with **~40 ms end-to-end latency**, well
above the 10 FPS minimum specified by the CSCI435 project brief.

## Why this matters

- **Accessibility:** Deaf students at UOWD currently rely on a small pool of
  interpreters whose schedules are constrained. SignSpeak AI does not replace
  human interpreters but unblocks moments when one is unavailable.
- **Cultural inclusion:** Most public sign-language tools focus exclusively on
  ASL. SignSpeak AI is one of the first student projects in the region to
  support **both** ASL and ArSL in a unified web application.
- **Web-first:** The app runs in any modern browser. No app store, no install,
  no special hardware - any laptop or office PC with a webcam can use it.

## Secondary scenarios

- **Visiting researcher** signs in ASL at a UOWD conference; a UAE staff
  member uses SignSpeak AI to follow the conversation.
- **A hearing student** learns the ArSL alphabet by signing and watching the
  predictions update in real time - the app doubles as a practice tool.
- **A pre-recorded video** of an ArSL lecture is uploaded via the **Upload**
  page; the system returns a frame-by-frame transcript that can be reviewed
  later.

## Out of scope (future work)

- Continuous sign-language translation (whole-sentence neural translation).
- Mobile native apps - SignSpeak AI is web-first by design but the same
  backend can power a future React Native or Flutter client.
- Bidirectional voice-to-sign (animated avatar that signs back to the deaf
  user).
