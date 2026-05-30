"""
Build short MP4 test clips from the same ASL / ArSL training images.

Outputs (by default):
    ../../test_videos/asl_sample.mp4
    ../../test_videos/arsl_sample.mp4

Run: python ml/scripts/generate_sample_videos.py
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSCI435 = ROOT.parent
OUT_DIR = CSCI435 / "test_videos"
INDEX = ROOT / "ml" / "data" / "datasets.json"
ARSL_ROOT = ROOT / "ml" / "data" / "arsl"

W, H = 640, 480
FPS = 10
FRAMES_PER_LETTER = 20  # 2 s each — enough for upload sampling + smoothing


@lru_cache(maxsize=1)
def _hands():
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


def _prepare_for_hands(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < 256:
        scale = 256 / min_dim
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    return img


def _has_hand(img: np.ndarray) -> bool:
    prepared = _prepare_for_hands(img)
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    res = _hands().process(rgb)
    return bool(res.multi_hand_landmarks)


def _find_asl_root() -> Path:
    if INDEX.exists():
        p = Path(json.loads(INDEX.read_text()).get("asl", ""))
        if p.exists():
            return p
    return Path.home() / ".cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1"


def _asl_class_dir(base: Path, letter: str) -> Path | None:
    letter = letter.upper()
    candidates = [
        base / "asl_alphabet_train" / "asl_alphabet_train" / letter,
        base / "asl_alphabet_train" / letter,
        base / letter,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _pick_detectable_image(folder: Path) -> Path:
    """Pick an image whose hand is still visible once placed on the video canvas."""
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for img_path in sorted(folder.glob(ext)):
            if _has_hand(_letter_frame(img_path)):
                return img_path
    hits = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in folder.glob(ext)
    )
    if not hits:
        raise FileNotFoundError(f"No image in {folder}")
    return hits[0]


def _letter_frame(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read {img_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    # Fill most of the frame so MediaPipe sees a large signing hand.
    scale = min((W - 40) / w, (H - 40) / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((H, W, 3), 32, dtype=np.uint8)
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames:
        for _ in range(FRAMES_PER_LETTER):
            writer.write(frame)
    writer.release()


def build_asl_video(out: Path) -> None:
    base = _find_asl_root()
    # H-I avoids ASL "E" images that often fail hand detection.
    sequence = ["H", "I", "L", "L", "O"]
    frames: list[np.ndarray] = []
    for letter in sequence:
        cls = _asl_class_dir(base, letter)
        if cls is None:
            raise FileNotFoundError(f"ASL class folder not found for {letter} under {base}")
        img = _pick_detectable_image(cls)
        frames.append(_letter_frame(img))
    _write_video(out, frames)
    print(
        f"Wrote {out}  ({len(sequence)} letters, "
        f"{len(sequence) * FRAMES_PER_LETTER / FPS:.1f}s)"
    )


def _pick_arsl_demo_image(folder: Path, class_name: str) -> Path:
    """Pick an ArSL frame that both detects a hand and classifies correctly."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.cv.pipeline import Pipeline
    from app.data.labels import ALL_CLASSES, NUM_ASL

    pipeline = Pipeline(device="cpu")
    best: tuple[Path, float] | None = None
    for img_path in sorted(folder.glob("*.png"))[:120]:
        canvas = _letter_frame(img_path)
        if not _has_hand(canvas):
            continue
        result = pipeline.run_frame(canvas, language="arsl")
        if (
            result.label_id >= NUM_ASL
            and ALL_CLASSES[result.label_id] == class_name
            and result.confidence >= 0.4
        ):
            if best is None or result.confidence > best[1]:
                best = (img_path, result.confidence)
    if best is not None:
        return best[0]
    return _pick_detectable_image(folder)


def build_arsl_video(out: Path) -> None:
    # Classes with reliable hand crops on the 640x480 canvas in practice.
    sequence = ["bb", "seen", "waw"]
    frames: list[np.ndarray] = []
    for folder in sequence:
        cls = ARSL_ROOT / folder
        if not cls.is_dir():
            raise FileNotFoundError(f"ArSL folder missing: {cls}")
        img = _pick_arsl_demo_image(cls, folder)
        frames.append(_letter_frame(img))
    _write_video(out, frames)
    print(
        f"Wrote {out}  ({len(sequence)} letters, "
        f"{len(sequence) * FRAMES_PER_LETTER / FPS:.1f}s)"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_dir = ROOT / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    asl_path = OUT_DIR / "asl_sample.mp4"
    arsl_path = OUT_DIR / "arsl_sample.mp4"
    build_asl_video(asl_path)
    try:
        build_arsl_video(arsl_path)
    except FileNotFoundError as exc:
        print(f"Skipping ArSL clip: {exc}")

    for src, name in ((asl_path, "sample_asl.mp4"), (arsl_path, "sample_arsl.mp4")):
        if not src.exists():
            continue
        dst = demo_dir / name
        dst.write_bytes(src.read_bytes())
        print(f"Copied -> {dst}")


if __name__ == "__main__":
    main()
