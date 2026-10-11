"""Reusable MLP classifier for tabular IDS data.

Design notes
------------
The output layer produces **2 logits** (softmax over classes), not a single
sigmoid. This is deliberate: the backdoor detectors —
STRIP (entropy over the predicted class distribution) and Neural Cleanse
(per-class trigger reverse-engineering) — both require a genuine probability
distribution *across classes*, which a single-logit sigmoid cannot provide.

``forward`` returns raw logits (what ``CrossEntropyLoss`` expects). Use
:meth:`MLP.predict_proba` for softmax probabilities and :meth:`MLP.predict`
for hard class predictions.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Configurable multilayer perceptron for binary IDS classification.

    Parameters
    ----------
    input_dim:
        Number of input features (69 for the preprocessed CICIDS2017 data).
    hidden_dims:
        Sizes of the hidden layers, in order. Default ``[128, 64, 32]``.
    num_classes:
        Number of output classes (2: benign / attack).
    dropout:
        Dropout probability applied after each hidden activation.
    use_batchnorm:
        If ``True``, insert ``BatchNorm1d`` after each hidden linear layer
        (before the activation).
    """

    def __init__(
        self,
        input_dim: int = 69,
        hidden_dims: Sequence[int] = (128, 64, 32),
        num_classes: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)

        layers: List[nn.Module] = []
        prev = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits (shape ``[batch, num_classes]``)."""
        return self.classifier(self.feature_extractor(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities across classes."""
        was_training = self.training
        self.eval()
        probs = F.softmax(self.forward(x), dim=1)
        if was_training:
            self.train()
        return probs

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return hard class predictions (argmax over logits)."""
        return self.forward(x).argmax(dim=1)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters (trainable by default)."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def config(self) -> dict:
        """Return the architecture hyperparameters (for checkpoints)."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "use_batchnorm": self.use_batchnorm,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "MLP":
        """Rebuild an MLP from a dict produced by :meth:`config`."""
        return cls(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
            use_batchnorm=cfg["use_batchnorm"],
        )
