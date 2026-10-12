"""Build and reload IDS models of any supported architecture.

The MLP code path is unchanged: an MLP checkpoint stores ``MLP.config()``, which
has no ``arch`` key, so it is rebuilt with :meth:`MLP.from_config` exactly as
``src.eval.run_comparison.load_ids_model`` does. Other architectures store an
``arch`` key in their ``model_config`` and are dispatched here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.mlp import MLP


def build_model(arch: str, model_cfg: dict, input_dim: int) -> nn.Module:
    """Construct an untrained model from the ``model`` block of a model config."""
    if arch == "mlp":
        return MLP(input_dim=input_dim, hidden_dims=model_cfg["hidden_dims"],
                   num_classes=model_cfg["num_classes"], dropout=model_cfg["dropout"],
                   use_batchnorm=model_cfg["use_batchnorm"])
    if arch == "cnn1d":
        from src.models.cnn1d import CNN1D
        return CNN1D(input_dim=input_dim, num_classes=model_cfg["num_classes"],
                     conv_channels=model_cfg["conv_channels"],
                     kernel_size=model_cfg["kernel_size"], fc_dim=model_cfg["fc_dim"],
                     dropout=model_cfg["dropout"])
    raise ValueError(f"unknown architecture {arch!r}")


def model_from_config(cfg: dict) -> nn.Module:
    """Rebuild a model from the ``model_config`` dict stored in a checkpoint."""
    arch = cfg.get("arch", "mlp")
    if arch == "mlp":
        return MLP.from_config(cfg)
    if arch == "cnn1d":
        from src.models.cnn1d import CNN1D
        return CNN1D.from_config(cfg)
    raise ValueError(f"unknown architecture {arch!r}")


def load_model(path: str | Path, device: torch.device | None = None) -> nn.Module:
    """Load a checkpoint written by ``src.models.train.train_model`` (eval mode)."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = model_from_config(ckpt["model_config"])
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()
