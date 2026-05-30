"""Landmark MLP: 63-D wrist-normalised input, 61-class output."""

from __future__ import annotations

import torch
import torch.nn as nn


class LandmarkMLP(nn.Module):
    """A small 3-hidden-layer MLP."""

    def __init__(
        self,
        in_dim: int = 63,
        hidden_dims: tuple[int, int, int] = (256, 128, 64),
        num_classes: int = 61,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(inplace=True),
            nn.Linear(h3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # noqa: D401
        return self.net(x)
