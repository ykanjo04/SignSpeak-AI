"""
Extract MediaPipe hand landmarks from every image in the downloaded datasets
and save them as a single ``.npz`` file.

Output layout (in ``ml/data/processed/``):
    landmarks.npz  ->  arrays:
        X        (N, 63)   float32   wrist-normalised landmark vectors
        y        (N,)      int64     class ids in the 61-class label space
        source   (N,)      U16       'asl' | 'arsl' | 'mnist'
        path     (N,)      U200      original file path (relative)

Run as::

    python ml/scripts/extract_landmarks.py [--limit-per-class N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Make the backend importable for the label map.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.data.labels import ASL_CLASSES, ARSL_CLASSES, NUM_ASL  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

DATA_DIR = ROOT / "ml" / "data"
INDEX_PATH = DATA_DIR / "datasets.json"
OUT_DIR = ROOT / "ml" / "data" / "processed"
OUT_PATH = OUT_DIR / "landmarks.npz"


def _source_root(source: str) -> Path | None:
    """Find on-disk root for the given source (asl/arsl/mnist).

    Looks first at ``ml/data/<source>`` (legacy layout) and then at the
    path recorded by ``download_datasets.py`` in ``datasets.json``.
    """
    legacy = DATA_DIR / source
    if legacy.exists() and any(legacy.iterdir()):
        return legacy
    if INDEX_PATH.exists():
        try:
            idx = json.loads(INDEX_PATH.read_text())
        except json.JSONDecodeError:
            idx = {}
        if source in idx and Path(idx[source]).exists():
            return Path(idx[source])
    return None


def _prepare_for_hands(img: np.ndarray) -> np.ndarray:
    """Upscale tiny dataset images so MediaPipe Hands can detect landmarks."""
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


def _wrist_normalise(landmarks: np.ndarray) -> np.ndarray:
    """Translate by the wrist and scale by middle-finger-tip distance."""
    arr = landmarks.astype(np.float32).copy()
    wrist = arr[0].copy()
    arr -= wrist
    scale = float(np.linalg.norm(arr[12]) + 1e-6)
    arr /= scale
    return arr.flatten()


def _enumerate_class_dirs(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield (class_name_lower, dir_path) for each subdirectory of ``root``."""
    if not root.exists():
        return
    for d in sorted(root.iterdir()):
        if d.is_dir():
            yield d.name.lower(), d


def _label_id_for(source: str, raw_class_name: str) -> int | None:
    """Map a raw on-disk class name to the global 0..60 label id."""
    nm = raw_class_name.strip().lower()
    if source in ("asl", "mnist"):
        # Match ASL_CLASSES case-insensitively. The mnist split has no
        # SPACE/DELETE/NOTHING and uses 0..25 -> A..Z. Same trick works.
        upper = nm.upper()
        for i, c in enumerate(ASL_CLASSES):
            if c == upper:
                return i
        # numeric label folders (0..25) in some packagings:
        if nm.isdigit():
            v = int(nm)
            if 0 <= v < 26:
                return v
        return None
    if source == "arsl":
        for j, c in enumerate(ARSL_CLASSES):
            if c == nm:
                return NUM_ASL + j
        return None
    return None


def _iter_dataset_images(
    source: str,
    limit_per_class: int | None,
) -> Iterable[tuple[int, Path]]:
    # Different Kaggle packagings put the class folders at different
    # depths; search up to two levels deep.
    base = _source_root(source)
    if base is None:
        return
    candidate_roots = [base]
    for child in base.iterdir():
        if child.is_dir():
            candidate_roots.append(child)
            for grandchild in child.iterdir():
                if grandchild.is_dir():
                    candidate_roots.append(grandchild)
    seen_classes: dict[int, int] = {}
    for root in candidate_roots:
        for name, cls_dir in _enumerate_class_dirs(root):
            label_id = _label_id_for(source, name)
            if label_id is None:
                continue
            count_for_class = seen_classes.get(label_id, 0)
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                if limit_per_class is not None and count_for_class >= limit_per_class:
                    break
                yield label_id, img
                count_for_class += 1
            seen_classes[label_id] = count_for_class


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--limit-per-class",
        type=int,
        default=400,
        help="Cap images per class to keep training fast on a laptop CPU.",
    )
    ap.add_argument(
        "--sources",
        nargs="*",
        default=["asl", "arsl"],
        choices=["asl", "arsl", "mnist"],
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting landmarks. cap per class = {args.limit_per_class}")

    hands_model = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

    X: list[np.ndarray] = []
    y: list[int] = []
    src_list: list[str] = []
    path_list: list[str] = []

    skipped_no_hand = 0
    total_seen = 0

    for source in args.sources:
        files = list(_iter_dataset_images(source, args.limit_per_class))
        if not files:
            print(f"  [{source}] no images found, skipping")
            continue
        print(f"  [{source}] {len(files):,} images")
        for label_id, fp in tqdm(files, desc=source, ncols=80):
            total_seen += 1
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img = _prepare_for_hands(img)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands_model.process(rgb)
            if not res.multi_hand_landmarks:
                skipped_no_hand += 1
                continue
            lm = res.multi_hand_landmarks[0]
            arr = np.asarray([(p.x, p.y, p.z) for p in lm.landmark], dtype=np.float32)
            X.append(_wrist_normalise(arr))
            y.append(label_id)
            src_list.append(source)
            try:
                path_list.append(str(fp.relative_to(ROOT)))
            except ValueError:
                path_list.append(str(fp))

    if not X:
        sys.exit("No landmarks extracted. Run download_datasets.py first.")

    X_arr = np.stack(X, axis=0).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    src_arr = np.asarray(src_list, dtype=np.str_)
    path_arr = np.asarray(path_list, dtype=np.str_)

    np.savez_compressed(
        OUT_PATH,
        X=X_arr,
        y=y_arr,
        source=src_arr,
        path=path_arr,
    )
    print(
        f"\nDone. Saved {OUT_PATH}\n"
        f"  X shape: {X_arr.shape}\n"
        f"  classes seen: {sorted(set(y))}\n"
        f"  skipped (no hand detected): {skipped_no_hand:,} / {total_seen:,}"
    )


if __name__ == "__main__":
    main()
