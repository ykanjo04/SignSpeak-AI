"""MobileNetV3-Small with a 61-class classifier head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_mobilenet(num_classes: int = 61, freeze_backbone: bool = True) -> nn.Module:
    """Return a fine-tunable MobileNetV3-Small."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    net: models.MobileNetV3 = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for p in net.features.parameters():
            p.requires_grad = False
        # Unfreeze the last conv block for better domain adaptation.
        for p in net.features[-1].parameters():
            p.requires_grad = True

    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, num_classes)
    return net


def preprocess_transform(image_size: int = 96):
    """Return the torchvision transform used for both training and inference."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
