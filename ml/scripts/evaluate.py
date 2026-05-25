"""
End-to-end evaluation script.

Produces every figure and table the report needs:

    ml/results/training_curves.png
    ml/results/confusion_matrix_asl.png
    ml/results/confusion_matrix_arsl.png
    ml/results/accuracy_table.csv
    ml/results/ablation_table.csv
    ml/results/fps_latency.csv
    ml/results/per_class_accuracy.csv

Run::

    python ml/scripts/evaluate.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import seaborn as sns                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.data.labels import (             # noqa: E402
    ALL_CLASSES,
    ARSL_INDICES,
    ASL_INDICES,
    NUM_ASL,
    NUM_CLASSES,
)
from app.models.mlp import LandmarkMLP    # noqa: E402

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RESULTS = ROOT / "ml" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
LANDMARKS = ROOT / "ml" / "data" / "processed" / "landmarks.npz"
MLP_CKPT = ROOT / "backend" / "app" / "models" / "checkpoints" / "mlp_static.pt"
MOBILENET_CKPT = ROOT / "backend" / "app" / "models" / "checkpoints" / "mobilenet_v3_static.pt"
HIST_MLP = ROOT / "ml" / "results" / "mlp_history.json"
HIST_MOBI = ROOT / "ml" / "results" / "mobilenet_history.json"


def _load_test_set() -> tuple[np.ndarray, np.ndarray]:
    from collections import Counter

    blob = np.load(LANDMARKS, allow_pickle=True)
    X, y = blob["X"], blob["y"]
    counts = Counter(y)
    keep = np.array([counts[int(yi)] >= 2 for yi in y])
    X, y = X[keep], y[keep]
    try:
        _, X_te, _, y_te = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
    except ValueError:
        _, X_te, _, y_te = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
    return X_te, y_te


def _predict_mlp(X_te: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model = LandmarkMLP(num_classes=NUM_CLASSES).to(device)
    state = torch.load(MLP_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(X_te), 512):
            xb = torch.from_numpy(X_te[i:i + 512]).to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=1)
            c, idx = p.max(dim=1)
            preds.append(idx.cpu().numpy())
            confs.append(c.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def _plot_training_curves() -> None:
    if not HIST_MLP.exists():
        return
    hist_mlp = json.loads(HIST_MLP.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist_mlp["train_acc"]) + 1)
    axes[0].plot(epochs, hist_mlp["train_acc"], label="MLP train")
    axes[0].plot(epochs, hist_mlp["val_acc"], label="MLP val")
    if HIST_MOBI.exists():
        hist_mobi = json.loads(HIST_MOBI.read_text())
        ep2 = range(1, len(hist_mobi["train_acc"]) + 1)
        axes[0].plot(ep2, hist_mobi["train_acc"], label="MobileNet train", linestyle="--")
        axes[0].plot(ep2, hist_mobi["val_acc"], label="MobileNet val", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training curves - accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist_mlp["train_loss"], label="MLP train")
    axes[1].plot(epochs, hist_mlp["val_loss"], label="MLP val")
    if HIST_MOBI.exists():
        hist_mobi = json.loads(HIST_MOBI.read_text())
        ep2 = range(1, len(hist_mobi["train_acc"]) + 1)
        axes[1].plot(ep2, hist_mobi["train_loss"], label="MobileNet train", linestyle="--")
        axes[1].plot(ep2, hist_mobi["val_loss"], label="MobileNet val", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Training curves - loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "training_curves.png", dpi=140)
    plt.close(fig)
    print(f"  saved training_curves.png")


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # ASL subset
    asl_mask = np.isin(y_true, ASL_INDICES)
    arsl_mask = np.isin(y_true, ARSL_INDICES)
    for name, mask, ids in (
        ("asl", asl_mask, ASL_INDICES),
        ("arsl", arsl_mask, ARSL_INDICES),
    ):
        if not mask.any():
            print(f"  no test samples for {name}; skipping confusion matrix")
            continue
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=ids)
        # Normalise per row.
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-6
        cm_norm = cm / row_sums

        class_names = [ALL_CLASSES[i] for i in ids]
        fig, ax = plt.subplots(figsize=(10 if name == "arsl" else 9, 8))
        sns.heatmap(
            cm_norm,
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
            cbar=True,
            square=True,
            linewidths=0.2,
            annot=False,
        )
        ax.set_title(f"Confusion matrix - {name.upper()}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        out = RESULTS / f"confusion_matrix_{name}.png"
        plt.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  saved {out.name}")


def _accuracy_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    overall = (y_true == y_pred).mean()
    asl_mask = np.isin(y_true, ASL_INDICES)
    arsl_mask = np.isin(y_true, ARSL_INDICES)
    asl_acc = (y_pred[asl_mask] == y_true[asl_mask]).mean() if asl_mask.any() else float("nan")
    arsl_acc = (y_pred[arsl_mask] == y_true[arsl_mask]).mean() if arsl_mask.any() else float("nan")

    rows = [
        ("Overall (ASL + ArSL)", overall, len(y_true)),
        ("ASL only", asl_acc, int(asl_mask.sum())),
        ("ArSL only", arsl_acc, int(arsl_mask.sum())),
    ]
    with (RESULTS / "accuracy_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Subset", "Accuracy", "N"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", r[2]])
    print("  accuracy table:")
    for r in rows:
        print(f"    {r[0]:25s} acc={r[1]:.4f}  N={r[2]:,}")

    # Per-class accuracy
    with (RESULTS / "per_class_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Class id", "Class name", "Accuracy", "N"])
        for i in range(NUM_CLASSES):
            mask = y_true == i
            if not mask.any():
                continue
            acc = float((y_pred[mask] == i).mean())
            w.writerow([i, ALL_CLASSES[i], f"{acc:.4f}", int(mask.sum())])


def _ablation_study(X_te: np.ndarray, y_te: np.ndarray, device: torch.device) -> None:
    """Compare a few feature variants to defend the design choices in the report."""
    # baseline: full normalised landmarks
    base_pred, _ = _predict_mlp(X_te, device)
    base_acc = (base_pred == y_te).mean()

    # without z coordinate
    X_no_z = X_te.copy().reshape(-1, 21, 3)
    X_no_z[:, :, 2] = 0
    X_no_z = X_no_z.reshape(-1, 63)
    # quick eval - reuse model with no-z input by zeroing component
    pred_no_z, _ = _predict_mlp(X_no_z, device)
    no_z_acc = (pred_no_z == y_te).mean()

    # add random noise (lighting-equivalent perturbation)
    rng = np.random.default_rng(0)
    X_noisy = X_te + rng.normal(scale=0.03, size=X_te.shape).astype(np.float32)
    pred_noisy, _ = _predict_mlp(X_noisy, device)
    noisy_acc = (pred_noisy == y_te).mean()

    rows = [
        ("Landmark MLP (baseline)", base_acc),
        ("Landmark MLP (z=0)", no_z_acc),
        ("Landmark MLP + 3% noise", noisy_acc),
    ]
    with (RESULTS / "ablation_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "Accuracy"])
        for name, acc in rows:
            w.writerow([name, f"{acc:.4f}"])
    print("  ablation:")
    for name, acc in rows:
        print(f"    {name:35s} acc={acc:.4f}")


def _fps_latency_bench(device: torch.device) -> None:
    """Measure end-to-end pipeline latency on a synthetic frame."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.cv.pipeline import Pipeline  # noqa: E402

    p = Pipeline(device=str(device))
    rng = np.random.default_rng(123)
    frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    # warmup
    for _ in range(3):
        p.run_frame(frame, language="auto")
    n = 30
    t0 = time.perf_counter()
    latencies = []
    for _ in range(n):
        t1 = time.perf_counter()
        p.run_frame(frame, language="auto")
        latencies.append((time.perf_counter() - t1) * 1000.0)
    total = time.perf_counter() - t0
    fps = n / total
    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))

    with (RESULTS / "fps_latency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Mean latency ms", f"{mean_ms:.2f}"])
        w.writerow(["P95 latency ms", f"{p95_ms:.2f}"])
        w.writerow(["FPS (sustained)", f"{fps:.2f}"])
        w.writerow(["Frames measured", n])
    print(f"  bench: mean={mean_ms:.1f} ms  p95={p95_ms:.1f} ms  fps={fps:.1f}")


def main() -> None:
    if not LANDMARKS.exists() or not MLP_CKPT.exists():
        sys.exit("Train the MLP first (extract_landmarks.py + train_mlp.py).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    X_te, y_te = _load_test_set()
    print(f"test set: {X_te.shape[0]:,} samples")

    print("\n[1/5] training curves")
    _plot_training_curves()

    print("\n[2/5] MLP predictions on test set")
    y_pred, _ = _predict_mlp(X_te, device)

    print("\n[3/5] accuracy breakdown")
    _accuracy_breakdown(y_te, y_pred)

    print("\n[4/5] confusion matrices")
    _plot_confusion(y_te, y_pred)

    print("\n[5/5] ablation + FPS benchmark")
    _ablation_study(X_te, y_te, device)
    _fps_latency_bench(device)

    # Save a sklearn classification report for the appendix
    present = sorted(set(y_te.tolist()) | set(y_pred.tolist()))
    target_names = [ALL_CLASSES[i] for i in present]
    report_txt = classification_report(
        y_te,
        y_pred,
        labels=present,
        target_names=target_names,
        zero_division=0,
    )
    (RESULTS / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    print("\nAll evaluation artefacts saved in", RESULTS)


if __name__ == "__main__":
    main()
