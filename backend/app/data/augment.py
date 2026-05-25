"""
Landmark and image augmentation used during training.

Two augmentation policies are exposed:

- ``augment_landmarks`` perturbs the 63-D landmark vector with small
  rotations, scales, and noise. Used by ``train_mlp.py``.
- ``image_train_transform`` returns the torchvision pipeline for the
  MobileNetV3 model and adds random colour jitter to simulate the
  lighting variation we expect at defence time.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms


def augment_landmarks(
    vec: np.ndarray,
    rotation_deg: float = 10.0,
    scale_jitter: float = 0.1,
    noise_std: float = 0.01,
    mirror_prob: float = 0.0,
) -> np.ndarray:
    """Apply small random transforms to a 63-D landmark vector."""
    if vec.shape[0] != 63:
        return vec
    pts = vec.reshape(21, 3).astype(np.float32).copy()

    theta = np.deg2rad(np.random.uniform(-rotation_deg, rotation_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    pts[:, :2] = pts[:, :2] @ rot.T

    scale = 1.0 + np.random.uniform(-scale_jitter, scale_jitter)
    pts *= scale

    pts += np.random.normal(scale=noise_std, size=pts.shape).astype(np.float32)

    if np.random.random() < mirror_prob:
        pts[:, 0] *= -1.0

    return pts.flatten()


def image_train_transform(image_size: int = 96):
    return transforms.Compose(
        [
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.0),  # signs are NOT mirror-invariant
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )


def image_eval_transform(image_size: int = 96):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
