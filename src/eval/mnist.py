"""MNIST data, a small CNN, and a canonical BadNets patch (image-domain validation).

No torchvision dependency: the raw idx files are downloaded (and cached) directly
and parsed with numpy. Images are returned FLATTENED to 784-vectors in [0,1] so
that the very same ``StripDetector`` / ``NeuralCleanseDetector`` from
``src/detectors`` run on them unchanged — only the perturbation operator differs
(STRIP ``superpose``; NC uses image-dimensional mask/pattern with [0,1] bounds).
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.utils import resolve_path

_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _download(dest_dir: Path) -> Dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, fname in _FILES.items():
        p = dest_dir / fname
        if not p.exists():
            urllib.request.urlretrieve(_MIRROR + fname, p)
        paths[key] = p
    return paths


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    n = int.from_bytes(buf[4:8], "big")
    rows = int.from_bytes(buf[8:12], "big")
    cols = int.from_bytes(buf[12:16], "big")
    data = np.frombuffer(buf[16:], dtype=np.uint8).reshape(n, rows * cols)
    return (data.astype(np.float32) / 255.0)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    return np.frombuffer(buf[8:], dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir: str = "data/raw/mnist") -> Dict[str, np.ndarray]:
    """Return flattened MNIST in [0,1]: X_train[60000,784], y_train, X_test, y_test."""
    paths = _download(resolve_path(data_dir))
    return {
        "X_train": _read_idx_images(paths["train_images"]),
        "y_train": _read_idx_labels(paths["train_labels"]),
        "X_test": _read_idx_images(paths["test_images"]),
        "y_test": _read_idx_labels(paths["test_labels"]),
    }


# ---------------------------------------------------------------------------
# BadNets patch trigger (canonical): 3x3 white square, bottom-right corner
# ---------------------------------------------------------------------------
def patch_indices(size: int = 3, img: int = 28) -> np.ndarray:
    """Flattened indices of a ``size`` x ``size`` bottom-right corner patch."""
    idx = []
    for r in range(img - size, img):
        for c in range(img - size, img):
            idx.append(r * img + c)
    return np.array(idx, dtype=int)


def apply_patch(X: np.ndarray, idx: np.ndarray, value: float = 1.0) -> np.ndarray:
    """Stamp the white patch onto (a copy of) flattened images ``X``."""
    Xt = np.array(X, dtype=np.float32, copy=True)
    Xt[..., idx] = value
    return Xt


def make_imbalanced(X, y, majority_class, majority_frac, rng, num_classes=10):
    """Subsample so ``majority_class`` is ~``majority_frac`` of the set.

    The majority class is kept in full; every other class is subsampled equally so
    the majority reaches the requested fraction. Used to build an IDS-like
    class-imbalanced MNIST (MNIST controlled experiment).
    """
    maj_idx = np.where(y == majority_class)[0]
    total = int(len(maj_idx) / majority_frac)
    per_minor = max(1, (total - len(maj_idx)) // (num_classes - 1))
    keep = list(maj_idx)
    for c in range(num_classes):
        if c == majority_class:
            continue
        c_idx = np.where(y == c)[0]
        keep.extend(rng.choice(c_idx, size=min(per_minor, len(c_idx)), replace=False))
    keep = np.array(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def poison_train(X, y, rate, target_class, patch_idx, rng):
    """In-place BadNets poisoning: stamp patch + relabel a fraction to target."""
    n = int(round(rate * len(X)))
    chosen = rng.choice(len(X), size=n, replace=False)
    Xp = np.array(X, dtype=np.float32, copy=True)
    yp = np.array(y, dtype=np.int64, copy=True)
    Xp[chosen] = apply_patch(Xp[chosen], patch_idx)
    yp[chosen] = target_class
    return Xp, yp, n


# ---------------------------------------------------------------------------
# Simple CNN (accepts flattened [B,784] so the detectors work unchanged)
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # [B, 784] -> [B, 1, 28, 28]
            x = x.view(-1, 1, 28, 28)
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_cnn(model, X, y, *, epochs=3, batch_size=128, lr=1e-3, device=None, seed=42):
    device = device or torch.device("cpu")
    model.to(device).train()
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    yt = torch.from_numpy(np.asarray(y, np.int64))
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"    epoch {ep+1}/{epochs}  train_loss={total/n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def accuracy(model, X, y, device=None, batch=1024) -> float:
    device = device or torch.device("cpu")
    model.eval()
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    yt = np.asarray(y, np.int64)
    preds = []
    for i in range(0, len(Xt), batch):
        preds.append(model(Xt[i : i + batch].to(device)).argmax(1).cpu().numpy())
    return float((np.concatenate(preds) == yt).mean())


@torch.no_grad()
def attack_success_rate(model, X_attack_src, y_src, patch_idx, target_class,
                        device=None, batch=1024) -> Dict[str, float]:
    """ASR over test samples NOT of the target class, correctly classified clean.

    Stamp the patch on those and measure the fraction now predicted as target.
    """
    device = device or torch.device("cpu")
    model.eval()
    Xt = torch.from_numpy(np.asarray(X_attack_src, np.float32))
    y = np.asarray(y_src, np.int64)
    clean_pred = []
    for i in range(0, len(Xt), batch):
        clean_pred.append(model(Xt[i : i + batch].to(device)).argmax(1).cpu().numpy())
    clean_pred = np.concatenate(clean_pred)
    elig = (y != target_class) & (clean_pred == y)  # correctly classified non-target
    Xtrig = apply_patch(X_attack_src[elig], patch_idx)
    Xtt = torch.from_numpy(Xtrig)
    tp = []
    for i in range(0, len(Xtt), batch):
        tp.append(model(Xtt[i : i + batch].to(device)).argmax(1).cpu().numpy())
    tp = np.concatenate(tp) if len(Xtt) else np.array([])
    asr = float((tp == target_class).mean()) if len(tp) else 0.0
    return {"asr": asr, "n_eligible": int(elig.sum())}
