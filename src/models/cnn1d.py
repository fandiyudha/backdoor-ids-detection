"""Compact 1D-CNN for tabular flow features (second architecture of the v1.1 experiments).

The standardized feature vector is treated as a 1-channel sequence of length D:

    Conv1d(1 -> 32, k=3, padding=1) + BN + ReLU
    Conv1d(32 -> 64, k=3, padding=1) + BN + ReLU
    flatten (64 x D)
    Linear(64 D -> 64) + BN + ReLU + Dropout(0.3)
    Linear(64 -> 2)

Design note: the specification first called for global average pooling over the
feature axis. Because the order of tabular features is arbitrary, pooling over
that axis would make the network nearly invariant to *which* feature carries a
value and discard feature identity. Before any CNN model was trained, the head
was therefore changed to flatten the convolutional feature maps (keeping all
layers and the three BN layers that IBD-PSC needs). ``padding=1`` keeps one
output position per input feature.

Interface identical to :class:`src.models.mlp.MLP`: ``forward`` returns raw
logits; ``config`` / ``from_config`` round-trip through checkpoints (the
``arch`` key lets ``src.models.registry`` rebuild it).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2,
                 conv_channels: Sequence[int] = (32, 64), kernel_size: int = 3,
                 fc_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.conv_channels = [int(c) for c in conv_channels]
        self.kernel_size = int(kernel_size)
        self.fc_dim = int(fc_dim)
        self.dropout = float(dropout)
        pad = self.kernel_size // 2
        layers, prev = [], 1
        for c in self.conv_channels:
            layers += [nn.Conv1d(prev, c, self.kernel_size, padding=pad), nn.BatchNorm1d(c), nn.ReLU()]
            prev = c
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev * self.input_dim, self.fc_dim), nn.BatchNorm1d(self.fc_dim), nn.ReLU(),
            nn.Dropout(self.dropout))
        self.classifier = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits (shape ``[batch, num_classes]``)."""
        return self.classifier(self.head(self.conv(x.unsqueeze(1))))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        probs = F.softmax(self.forward(x), dim=1)
        if was_training:
            self.train()
        return probs

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)

    def config(self) -> dict:
        return {"arch": "cnn1d", "head": "flatten", "input_dim": self.input_dim,
                "num_classes": self.num_classes, "conv_channels": self.conv_channels,
                "kernel_size": self.kernel_size, "fc_dim": self.fc_dim, "dropout": self.dropout}

    @classmethod
    def from_config(cls, cfg: dict) -> "CNN1D":
        return cls(input_dim=cfg["input_dim"], num_classes=cfg["num_classes"],
                   conv_channels=cfg["conv_channels"], kernel_size=cfg["kernel_size"],
                   fc_dim=cfg["fc_dim"], dropout=cfg["dropout"])
