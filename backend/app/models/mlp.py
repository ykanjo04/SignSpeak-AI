"""
Custom landmark-MLP classifier (trained from scratch).

Input: 63-D wrist-normalised flat vector from MediaPipe Hands.
Output: 61-class log-softmax distribution over the SignSpeak label space.

This model satisfies the CSCI435 requirement that at least one model be
trained on data curated by the team (we curate the landmarks by running
MediaPipe ourselves on the public ASL Alphabet and ArSL2018 datasets).
"""

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
