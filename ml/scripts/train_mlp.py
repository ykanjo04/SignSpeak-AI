"""
Train the landmark MLP classifier from scratch.

Inputs:  ml/data/processed/landmarks.npz  (from extract_landmarks.py)
Outputs:
    backend/app/models/checkpoints/mlp_static.pt
    ml/results/mlp_history.json
    ml/results/training_curves.png  (combined with MobileNet curves later)

Run::

    python ml/scripts/train_mlp.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.data.augment import augment_landmarks  # noqa: E402
from app.data.labels import NUM_CLASSES  # noqa: E402
from app.models.mlp import LandmarkMLP  # noqa: E402


DATA_PATH = ROOT / "ml" / "data" / "processed" / "landmarks.npz"
CKPT_PATH = ROOT / "backend" / "app" / "models" / "checkpoints" / "mlp_static.pt"
HISTORY_PATH = ROOT / "ml" / "results" / "mlp_history.json"


def _train_test_split_safe(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.15, random_state: int = 42
):
    """Stratified split, dropping singleton classes when stratify is impossible."""
    counts = Counter(y)
    keep = np.array([counts[int(yi)] >= 2 for yi in y])
    dropped = int((~keep).sum())
    if dropped:
        print(f"  dropping {dropped} samples from singleton classes before split")
    X, y = X[keep], y[keep]
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )


def _augment_batch(X: np.ndarray) -> np.ndarray:
    return np.stack([augment_landmarks(v) for v in X], axis=0)


def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(f"Missing {DATA_PATH}. Run extract_landmarks.py first.")

    print(f"Loading {DATA_PATH}")
    blob = np.load(DATA_PATH, allow_pickle=True)
    X, y = blob["X"], blob["y"]
    print(f"  X={X.shape}  y={y.shape}  unique classes={len(set(y))}")

    X_tr, X_te, y_tr, y_te = _train_test_split_safe(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = LandmarkMLP(num_classes=NUM_CLASSES).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=20)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, 21):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch:02d}", ncols=80):
            xb_np = _augment_batch(xb.numpy())
            xb = torch.from_numpy(xb_np).to(device)
            yb = yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * xb.size(0)
                v_correct += (logits.argmax(dim=1) == yb).sum().item()
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
    elapsed = time.time() - t0
    print(f"\nMLP best val_acc = {best_acc:.4f} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
