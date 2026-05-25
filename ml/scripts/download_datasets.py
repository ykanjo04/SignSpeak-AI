"""
One-shot dataset download helper.

Datasets used by SignSpeak AI:
    1. ASL Alphabet         (grassknoted/asl-alphabet)
    2. Sign Language MNIST  (datamunge/sign-language-mnist)
    3. ArSL2018             (muhammadkhalid/arabic-sign-language-dataset-2022)

Usage:
    python ml/scripts/download_datasets.py [--skip arsl|asl|mnist] [--limit 1000]

Requires the `kagglehub` package. The first call will prompt for your Kaggle
username and API key if none is configured. Get a free API key at
https://www.kaggle.com/settings -> Create New Token.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"

DATASETS = {
    "asl": [
        "grassknoted/asl-alphabet",
    ],
    "mnist": [
        "datamunge/sign-language-mnist",
    ],
    # The Arabic Sign Language dataset has had multiple Kaggle re-uploads.
    # We try each in order until one succeeds.
    "arsl": [
        "muhammadkhalid/arabic-sign-language-dataset-2022",
        "mloey1/arabic-sign-language-2018",
        "muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset",
    ],
}


def _ensure_kaggle_credentials() -> None:
    """If neither env vars nor kaggle.json are present, prompt the user."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    home = Path.home()
    kj = home / ".kaggle" / "kaggle.json"
    if kj.exists():
        return
    print("=" * 70)
    print("Kaggle credentials not found.")
    print("Get a free token at https://www.kaggle.com/settings -> Create New Token")
    print("The downloaded kaggle.json file contains your username + key.")
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


def _try_download(slug: str) -> Path | None:
    print(f"  attempting: {slug}")
    try:
        path = kagglehub.dataset_download(slug)
        return Path(path)
    except Exception as exc:        # noqa: BLE001
        print(f"  failed:    {slug}  ({exc})")
        return None


def download_one(key: str, slugs: list[str]) -> None:
    target = DATA_DIR / key
    if target.exists() and any(target.iterdir()):
        print(f"[{key}] already present at {target}, skipping.")
        return
    print(f"[{key}] downloading...")
    src: Path | None = None
    for slug in slugs:
        src = _try_download(slug)
        if src is not None and any(src.iterdir()):
            break
    if src is None:
        print(f"[{key}] all Kaggle slugs failed. Skipping.")
        return
    target.mkdir(parents=True, exist_ok=True)
    # Copy (not move) so kagglehub's cache is preserved for re-runs.
    print(f"[{key}] copying from {src} to {target} ...")
    for entry in src.iterdir():
        dest = target / entry.name
        if dest.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, dest)
        else:
            shutil.copy2(entry, dest)
    print(f"[{key}] done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip", nargs="*", default=[], choices=DATASETS.keys())
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_kaggle_credentials()
    for key, slugs in DATASETS.items():
        if key in args.skip:
            print(f"[{key}] skipped by --skip flag")
            continue
        download_one(key, slugs)

    print("\nAll downloads attempted. Layout:")
    for p in sorted(DATA_DIR.rglob("*"))[:25]:
        print(" ", p.relative_to(DATA_DIR))
    if len(list(DATA_DIR.rglob("*"))) > 25:
        print("  ...")


if __name__ == "__main__":
    main()
