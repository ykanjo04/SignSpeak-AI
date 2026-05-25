"""
Fine-tune MobileNetV3-Small on hand crops from the same datasets used by
the MLP. To stay laptop-friendly we:

- freeze most of the backbone (see ``backend/app/models/mobilenet.py``)
- resize crops to 96x96
- cap the number of images per class

The script first re-runs MediaPipe Hands on the dataset to extract square
hand crops and caches them at ``ml/data/processed/crops/<class>/<id>.jpg``.

Run::

    python ml/scripts/train_mobilenet.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.data.augment import image_eval_transform, image_train_transform  # noqa: E402
from app.data.labels import ARSL_CLASSES, NUM_ASL, NUM_CLASSES  # noqa: E402
from app.models.mobilenet import build_mobilenet  # noqa: E402


warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

CROP_DIR = ROOT / "ml" / "data" / "processed" / "crops"
LANDMARKS_PATH = ROOT / "ml" / "data" / "processed" / "landmarks.npz"
CKPT_PATH = ROOT / "backend" / "app" / "models" / "checkpoints" / "mobilenet_v3_static.pt"
HISTORY_PATH = ROOT / "ml" / "results" / "mobilenet_history.json"
IMG_SIZE = 96
ARSL_ROOT = ROOT / "ml" / "data" / "arsl"


def _ensure_arsl_full_crops() -> None:
    """Cache upscaled full-frame ArSL images (32x32 HF exports) for MobileNet."""
    if not ARSL_ROOT.exists():
        return
    print("Caching ArSL full-frame crops for MobileNet...")
    for j, cls_name in enumerate(ARSL_CLASSES):
        lbl = NUM_ASL + j
        cls_dir = ARSL_ROOT / cls_name
        if not cls_dir.is_dir():
            continue
        out_dir = CROP_DIR / str(lbl)
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = sum(1 for _ in out_dir.glob("arsl_*.jpg"))
        if existing >= 300:
            continue
        for i, img_path in enumerate(sorted(cls_dir.glob("*.png"))):
            if i >= 400:
                break
            out_path = out_dir / f"arsl_{i:05d}.jpg"
            if out_path.exists():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _ensure_crops() -> None:
    """If ASL hand crops are missing, regenerate them from landmark source paths."""
    _ensure_arsl_full_crops()
    has_asl_crops = any(
        any((CROP_DIR / str(i)).glob("*.jpg"))
        for i in range(NUM_ASL)
        if (CROP_DIR / str(i)).exists()
    )
    if has_asl_crops:
        return
    if not LANDMARKS_PATH.exists():
        sys.exit("landmarks.npz missing -- run extract_landmarks.py first.")
    import mediapipe as mp

    blob = np.load(LANDMARKS_PATH, allow_pickle=True)
    paths = blob["path"]
    labels = blob["y"]
    print(f"Generating hand crops for {len(paths):,} images...")
    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1,
        model_complexity=1, min_detection_confidence=0.5,
    )
    for i, (rel_path, lbl) in enumerate(tqdm(zip(paths, labels), total=len(paths), ncols=80)):
        img_path = (ROOT / rel_path) if not Path(rel_path).is_absolute() else Path(rel_path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        min_dim = min(h, w)
        if min_dim < 200:
            scale = 256 / min_dim
            img = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
            h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        lm = res.multi_hand_landmarks[0]
        xs = [p.x for p in lm.landmark]
        ys = [p.y for p in lm.landmark]
        margin = 0.25
        x0 = max(0, int((min(xs) - margin) * w))
        x1 = min(w, int((max(xs) + margin) * w))
        y0 = max(0, int((min(ys) - margin) * h))
        y1 = min(h, int((max(ys) + margin) * h))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        out_dir = CROP_DIR / str(int(lbl))
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{i:07d}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _index_crops() -> tuple[list[Path], list[int]]:
    files: list[Path] = []
    labels: list[int] = []
    for cls_dir in sorted(CROP_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        try:
            cls = int(cls_dir.name)
        except ValueError:
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() == ".jpg":
                files.append(img)
                labels.append(cls)
    return files, labels


from collections import Counter


def _train_test_split_safe(files, labels, test_size=0.15, random_state=42):
    counts = Counter(labels)
    keep_idx = [i for i, lbl in enumerate(labels) if counts[lbl] >= 2]
    if len(keep_idx) < len(labels):
        print(f"  dropping {len(labels) - len(keep_idx)} singleton crop samples before split")
    files = [files[i] for i in keep_idx]
    labels = [labels[i] for i in keep_idx]
    try:
        return train_test_split(
            files, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    except ValueError:
        return train_test_split(
            files, labels, test_size=test_size, random_state=random_state
        )


class CropDataset(Dataset):
    def __init__(self, files: list[Path], labels: list[int], transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        x = self.transform(img)
        return x, self.labels[idx]


def main() -> None:
    _ensure_crops()
    files, labels = _index_crops()
    if not files:
        sys.exit("No crops to train on.")
    print(f"  found {len(files):,} crops across {len(set(labels))} classes")

    tr_f, te_f, tr_l, te_l = _train_test_split_safe(files, labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    train_ds = CropDataset(tr_f, tr_l, image_train_transform(IMG_SIZE))
    test_ds = CropDataset(te_f, te_l, image_eval_transform(IMG_SIZE))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = build_mobilenet(NUM_CLASSES, freeze_backbone=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, 6):
        model.train()
        running, correct, total = 0.0, 0, 0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch:02d}", ncols=80):
            xb, yb = xb.to(device), torch.as_tensor(yb).to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            running += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        train_loss, train_acc = running / total, correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), torch.as_tensor(yb).to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * xb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += xb.size(0)
        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  epoch {epoch:02d}  "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"    saved {CKPT_PATH}")

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    print(f"\nMobileNet best val_acc = {best_acc:.4f} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
