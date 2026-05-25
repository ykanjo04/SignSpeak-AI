"""
Generate the architecture diagram for the report and the docs page.

Output: ``report/figures/architecture.png``  (also copied to ``docs/architecture.png``)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches      # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_REPORT = ROOT / "report" / "figures" / "architecture.png"
OUT_DOCS = ROOT / "docs" / "architecture.png"


def make_box(ax, x, y, w, h, text, fontsize=10, fc="#1f2937", ec="#60a5fa"):
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="white",
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#94a3b8"),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                ha="center", fontsize=8, color="#94a3b8")


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#0b1220")

    # Browser column
    make_box(ax, 0.3, 5.3, 4, 1.3, "Browser - Next.js + Tailwind\n(Webcam stream / Upload page,\nskeleton overlay, sentence builder, TTS)",
             fontsize=9, fc="#0f172a")

    # Backend column
    make_box(ax, 4.7, 5.3, 3.8, 1.3,
             "FastAPI Backend\n/ws/live  +  POST /upload",
             fontsize=10, fc="#1e293b")

    # Pipeline column - 8 stages
    stages = [
        "1. CLAHE enhancement",
        "2. Selfie segmentation",
        "3. Morphology",
        "4. Holistic landmarks",
        "5. Canny edges",
        "6. Optical flow",
        "7. MLP inference",
        "8. MobileNet inference",
    ]
    for i, s in enumerate(stages):
        col = i // 4
        row = i % 4
        x = 4.7 + col * 1.95
        y = 4.6 - row * 0.95
        make_box(ax, x, y, 1.85, 0.7, s, fontsize=8.5)

    # Ensemble box
    make_box(ax, 8.8, 4.0, 3.7, 0.8, "Ensemble + smoothing buffer\n(8/10 vote, conf>=0.85)",
             fontsize=9, fc="#1e3a8a", ec="#fbbf24")

    # Models column
    make_box(ax, 9.0, 6.0, 3.5, 0.7, "Landmark MLP (trained)\n+ MobileNetV3 (fine-tuned)",
             fontsize=8.5, fc="#1e293b")

    # Arrows
    arrow(ax, 4.3, 6.0, 4.7, 6.0, "frames")
    arrow(ax, 11.0, 5.95, 11.0, 5.0)              # models -> ensemble
    arrow(ax, 8.5, 4.4, 8.8, 4.4)                 # pipeline -> ensemble
    arrow(ax, 8.8, 4.0, 8.8, 3.5)                 # ensemble down
    arrow(ax, 8.7, 3.4, 0.5, 3.0, "JSON predictions")
    arrow(ax, 0.5, 3.0, 0.5, 5.3)                 # JSON back to browser
    arrow(ax, 4.7, 6.0, 4.5, 5.95)                # WS<-browser bidi

    # Title
    ax.text(6.5, 6.85, "SignSpeak AI - System Architecture",
            ha="center", fontsize=14, color="white", weight="bold")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_REPORT, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    shutil.copy(OUT_REPORT, OUT_DOCS)
    print(f"saved {OUT_REPORT}")
    print(f"saved {OUT_DOCS}")


if __name__ == "__main__":
    main()
