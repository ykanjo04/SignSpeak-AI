"""Download ASL, MNIST, and ArSL datasets. Writes paths to ml/data/datasets.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
INDEX_PATH = DATA_DIR / "datasets.json"
ARSL_OUT = DATA_DIR / "arsl"

KAGGLE_DATASETS: dict[str, list[str]] = {
    "asl": [
        "grassknoted/asl-alphabet",
    ],
    "mnist": [
        "datamunge/sign-language-mnist",
    ],
}

HF_ARSL_REPO = "pain/ArASL_Database_Grayscale"


def _ensure_kaggle_credentials() -> None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists():
        return
    print("=" * 70)
    print("Kaggle credentials not found.")
    print("Get a free token at https://www.kaggle.com/settings -> Create New Token")
    print("=" * 70)
    user = input("Kaggle username: ").strip()
    key = input("Kaggle key:      ").strip()
    if not user or not key:
        sys.exit("No credentials provided; aborting.")
    kj.parent.mkdir(parents=True, exist_ok=True)
    kj.write_text(f'{{"username":"{user}","key":"{key}"}}')
    try:
        os.chmod(kj, 0o600)
    except OSError:
        pass
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key


def _try_kaggle_download(slug: str) -> Path | None:
    print(f"  attempting: {slug}")
    try:
        path = kagglehub.dataset_download(slug)
        return Path(path)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()[0][:120]
        print(f"  failed:    {slug}  ({msg})")
        return None


def download_kaggle_one(key: str, slugs: list[str]) -> Path | None:
    print(f"[{key}] downloading from Kaggle...")
    for slug in slugs:
        src = _try_kaggle_download(slug)
        if src is not None and any(src.iterdir()):
            print(f"[{key}] cached at {src}")
            return src
    print(f"[{key}] all Kaggle slugs failed.")
    return None


def _arsl_already_exported(min_images: int = 1000) -> bool:
    if not ARSL_OUT.exists():
        return False
    count = sum(
        1
        for p in ARSL_OUT.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    return count >= min_images


def download_arsl_hf(limit_per_class: int | None = 400) -> Path | None:
    """Export ArSL2018 from Hugging Face into ``ml/data/arsl/<class>/``."""
    print(f"[arsl] downloading from Hugging Face ({HF_ARSL_REPO})...")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install the `datasets` package: pip install datasets huggingface_hub")

    if _arsl_already_exported():
        print(f"[arsl] already exported at {ARSL_OUT}")
        return ARSL_OUT

    ds = load_dataset(HF_ARSL_REPO, split="train")
    label_names: list[str] = ds.features["label"].names  # type: ignore[attr-defined]
    class_counts = {name: 0 for name in label_names}

    ARSL_OUT.mkdir(parents=True, exist_ok=True)
    exported = 0
    for i, row in enumerate(ds):
        label_idx = int(row["label"])
        class_name = label_names[label_idx]
        if limit_per_class is not None and class_counts[class_name] >= limit_per_class:
            continue
        cls_dir = ARSL_OUT / class_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = row["image"]
        out_path = cls_dir / f"{class_counts[class_name]:05d}.png"
        img.save(out_path)
        class_counts[class_name] += 1
        exported += 1
        if limit_per_class is not None and all(
            c >= limit_per_class for c in class_counts.values()
        ):
            break
        if (i + 1) % 5000 == 0:
            print(f"  scanned {i + 1:,} rows, exported {exported:,} images...")

    if exported == 0:
        print("[arsl] Hugging Face export produced no images.")
        return None

    print(f"[arsl] exported {exported:,} images to {ARSL_OUT}")
    for name in label_names:
        print(f"    {name:8s} -> {class_counts[name]:4d} images")
    return ARSL_OUT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip", nargs="*", default=[], choices=["asl", "mnist", "arsl"])
    ap.add_argument(
        "--arsl-limit",
        type=int,
        default=400,
        help="Max images per ArSL class to export from Hugging Face (default 400).",
    )
    ap.add_argument(
        "--arsl-full",
        action="store_true",
        help="Export the full 54k-image ArSL2018 set (overrides --arsl-limit).",
    )
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    index: dict[str, str] = {}
    if INDEX_PATH.exists():
        try:
            index = json.loads(INDEX_PATH.read_text())
        except json.JSONDecodeError:
            index = {}

    need_kaggle = any(k not in args.skip for k in KAGGLE_DATASETS)
    if need_kaggle:
        _ensure_kaggle_credentials()

    for key, slugs in KAGGLE_DATASETS.items():
        if key in args.skip:
            print(f"[{key}] skipped by --skip flag")
            continue
        if key in index and Path(index[key]).exists():
            print(f"[{key}] already indexed at {index[key]}, skipping.")
            continue
        src = download_kaggle_one(key, slugs)
        if src is not None:
            index[key] = str(src)

    if "arsl" not in args.skip:
        if "arsl" in index and Path(index["arsl"]).exists() and _arsl_already_exported():
            print(f"[arsl] already indexed at {index['arsl']}, skipping.")
        else:
            limit = None if args.arsl_full else args.arsl_limit
            src = download_arsl_hf(limit_per_class=limit)
            if src is not None:
                index["arsl"] = str(src)

    INDEX_PATH.write_text(json.dumps(index, indent=2))
    print("\nDataset index written to", INDEX_PATH)
    for k, v in index.items():
        print(f"  {k:6s} -> {v}")


if __name__ == "__main__":
    main()
