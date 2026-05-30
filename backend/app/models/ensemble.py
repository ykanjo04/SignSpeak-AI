"""
Two-model ensemble: average MLP and MobileNet softmax, then argmax.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.data.labels import NUM_CLASSES
from app.models.mlp import LandmarkMLP
from app.models.mobilenet import build_mobilenet, preprocess_transform


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
MLP_CKPT = CHECKPOINT_DIR / "mlp_static.pt"
MOBILENET_CKPT = CHECKPOINT_DIR / "mobilenet_v3_static.pt"


class Ensemble:
    """Holds the two models and runs them on a single frame."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.mlp = LandmarkMLP(num_classes=NUM_CLASSES).to(self.device).eval()
        self.mobilenet = build_mobilenet(NUM_CLASSES).to(self.device).eval()
        self.transform = preprocess_transform(96)
        self.has_mlp = False
        self.has_mobile = False
        self._load_if_present()

    def _load_if_present(self) -> None:
        if MLP_CKPT.exists():
            state = torch.load(MLP_CKPT, map_location=self.device, weights_only=True)
            self.mlp.load_state_dict(state)
            self.has_mlp = True
        if MOBILENET_CKPT.exists():
            state = torch.load(
                MOBILENET_CKPT, map_location=self.device, weights_only=True
            )
            self.mobilenet.load_state_dict(state)
            self.has_mobile = True

    def predict(
        self,
        landmark_vec: np.ndarray | None = None,
        hand_crop_bgr: np.ndarray | None = None,
        allowed_indices: list[int] | None = None,
    ) -> tuple[int, float, dict[str, float]]:
        """Run both models and return ``(label_id, confidence, per_model)``.

        ``per_model`` is a small dict that the report uses for the ablation
        table (e.g. ``{"mlp": 0.93, "mobilenet": 0.81}``).
        """
        probs_list: list[torch.Tensor] = []
        per_model: dict[str, float] = {}

        if self.has_mlp and landmark_vec is not None:
            x = torch.from_numpy(landmark_vec.astype(np.float32)).unsqueeze(0)
            x = x.to(self.device)
            with torch.no_grad():
                logits = self.mlp(x)
                p = F.softmax(logits, dim=1)
            probs_list.append(p)
            per_model["mlp"] = float(p.max().item())

        if self.has_mobile and hand_crop_bgr is not None:
            # BGR -> RGB -> PIL -> tensor
            rgb = hand_crop_bgr[..., ::-1].copy()
            pil = Image.fromarray(rgb)
            x = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.mobilenet(x)
                p = F.softmax(logits, dim=1)
            probs_list.append(p)
            per_model["mobilenet"] = float(p.max().item())

        if not probs_list:
            # fallback when checkpoints are missing
            return -1, 0.0, per_model

        avg = torch.stack(probs_list, dim=0).mean(dim=0)
        if allowed_indices is not None:
            mask = torch.full_like(avg, fill_value=-1.0)
            for i in allowed_indices:
                mask[0, i] = avg[0, i]
            avg = mask
        conf, idx = torch.max(avg, dim=1)
        return int(idx.item()), float(conf.item()), per_model
