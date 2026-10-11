"""Reusable training / evaluation utilities for the IDS MLP.

These functions are intentionally model-agnostic and dataset-agnostic so that
the attack experiment can reuse them to train a *poisoned* model: pass in different arrays or
a different model instance and everything else (class weights, early stopping,
evaluation, curves) stays the same.

Everything runs on CPU by default (the environment is CPU-only torch).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLP

ATTACK_CLASS = 1  # positive class (attack); benign = 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(processed_dir: Path) -> Dict[str, np.ndarray]:
    """Load the preprocessed ``dataset.npz`` into a dict of numpy arrays."""
    npz_path = Path(processed_dir) / "dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run the preprocessing first: "
            "python -m src.data_prep.preprocess"
        )
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def make_loaders(
    data: Dict[str, np.ndarray], batch_size: int, seed: int = 42
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders from the loaded arrays."""
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in ("train", "val", "test"):
        X = torch.from_numpy(np.asarray(data[f"X_{split}"], dtype=np.float32))
        y = torch.from_numpy(np.asarray(data[f"y_{split}"], dtype=np.int64))
        ds = TensorDataset(X, y)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
        )
    return loaders


def compute_class_weights(y_train: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights (sklearn 'balanced' scheme).

    ``w_c = n_samples / (n_classes * count_c)`` — larger weight for the rarer
    (attack) class so the imbalanced objective stops favouring benign.
    """
    y = np.asarray(y_train).astype(int)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    n = counts.sum()
    weights = n / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect_predictions(
    model: MLP, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, attack_prob, y_pred) over a whole loader."""
    model.eval()
    ys, probs = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, ATTACK_CLASS]
        probs.append(p.cpu().numpy())
        ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    attack_prob = np.concatenate(probs)
    y_pred = (attack_prob >= 0.5).astype(np.int64)
    return y_true, attack_prob, y_pred


def compute_metrics(
    y_true: np.ndarray, attack_prob: np.ndarray, y_pred: np.ndarray
) -> Dict[str, object]:
    """Full metric suite, highlighting the attack class and PR-AUC."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": {
            "benign": {
                "precision": float(prec[0]),
                "recall": float(rec[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "attack": {
                "precision": float(prec[1]),
                "recall": float(rec[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        # Headline metrics for an imbalanced IDS problem:
        "attack_recall": float(rec[1]),
        "attack_f1": float(f1[1]),
        "roc_auc": float(roc_auc_score(y_true, attack_prob)),
        "pr_auc": float(average_precision_score(y_true, attack_prob)),
        "confusion_matrix": cm.tolist(),
    }


def evaluate(
    model: MLP, loader: DataLoader, device: torch.device
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    """Evaluate on a loader; return (metrics, y_true, attack_prob)."""
    y_true, attack_prob, y_pred = _collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, attack_prob, y_pred)
    return metrics, y_true, attack_prob


# ---------------------------------------------------------------------------
# Training (reusable — the attack experiment reuses this for the poisoned model)
# ---------------------------------------------------------------------------
def train_model(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    class_weights: Optional[torch.Tensor] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 50,
    patience: int = 7,
    min_delta: float = 0.0,
    monitor: str = "attack_f1",
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train with early stopping; keep and reload the best checkpoint.

    ``monitor`` is one of ``"val_pr_auc"`` / ``"attack_f1"`` (maximized) or
    ``"val_loss"`` (minimized). The best model weights (by the monitored metric
    on the validation split) are restored into ``model`` before returning, and
    saved to ``checkpoint_path`` if given.
    """
    valid_monitors = {"val_pr_auc", "attack_f1", "val_loss"}
    if monitor not in valid_monitors:
        raise ValueError(f"monitor must be one of {valid_monitors}, got {monitor!r}")

    device = device or torch.device("cpu")
    model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    maximize = monitor != "val_loss"
    best_score = -np.inf if maximize else np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running / max(n_seen, 1)

        # --- validation ---
        val_loss = _val_loss(model, val_loader, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_attack_f1 = val_metrics["attack_f1"]
        score = {
            "val_pr_auc": val_metrics["pr_auc"],
            "attack_f1": val_attack_f1,
            "val_loss": val_loss,
        }[monitor]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_attack_f1": val_attack_f1,
                "val_attack_recall": val_metrics["attack_recall"],
                "val_pr_auc": val_metrics["pr_auc"],
            }
        )
        if verbose:
            print(
                f"  epoch {epoch:>2}/{max_epochs} | train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} | val_attack_f1={val_attack_f1:.4f} "
                f"| val_pr_auc={val_metrics['pr_auc']:.4f}"
            )

        improved = (
            (score > best_score + min_delta)
            if maximize
            else (score < best_score - min_delta)
        )
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"  early stopping at epoch {epoch} "
                        f"(no {monitor} improvement for {patience} epochs)"
                    )
                break

    # restore best weights
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "model_config": model.config(),
                "monitor": monitor,
                "best_score": float(best_score),
                "best_epoch": best_epoch,
            },
            checkpoint_path,
        )
        if verbose:
            print(f"  saved best checkpoint (epoch {best_epoch}) -> {checkpoint_path}")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "monitor": monitor,
    }


@torch.no_grad()
def _val_loss(
    model: MLP, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    running, n_seen = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        running += criterion(model(xb), yb).item() * xb.size(0)
        n_seen += xb.size(0)
    return running / max(n_seen, 1)


def roc_pr_points(
    y_true: np.ndarray, attack_prob: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute ROC and PR curve points for plotting."""
    fpr, tpr, _ = roc_curve(y_true, attack_prob)
    precision, recall, _ = precision_recall_curve(y_true, attack_prob)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}
