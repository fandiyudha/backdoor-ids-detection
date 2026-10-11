"""Shared utilities for reproducibility and config handling.

The single most important helper here is :func:`set_global_seed`, which locks
down every source of randomness we touch (Python ``random``, NumPy, and PyTorch)
so that experiments can be regenerated exactly. Every entry point in this repo
should call it once, early, using the seed declared in the config file.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:  # torch is optional at import time (e.g. pure data-prep runs on CI)
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch missing shouldn't break data prep
    _HAS_TORCH = False


def set_global_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all RNGs used in the project.

    Parameters
    ----------
    seed:
        The seed applied to ``random``, ``numpy``, and (if available) ``torch``.
    deterministic:
        When ``True`` and CUDA is present, forces cuDNN into deterministic mode.
        This trades a little speed for exact reproducibility.

    Returns
    -------
    int
        The seed that was set (handy for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_root() -> Path:
    """Return the repository root (the parent of ``src/``)."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve ``path`` relative to the project root unless already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (project_root() / p)
