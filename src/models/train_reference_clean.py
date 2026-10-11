"""Train N extra CLEAN reference IDS models with different seeds.

These healthy models give a *distribution* of the Neural Cleanse L1 mask-norm
ratio, so the detection threshold can be set from mean + k*std instead of a
single clean observation. Uses the identical clean-baseline training protocol.
Required by the image-validation, comparison, calibration, and statistics runs.

Run:
    python -m src.models.train_reference_clean --config configs/model.yaml \
        --seeds 43,44,45,46,47
"""

from __future__ import annotations

import argparse

import torch

from src.models.mlp import MLP
from src.models.train import (
    compute_class_weights,
    load_dataset,
    make_loaders,
    train_model,
)
from src.utils import load_yaml, resolve_path, set_global_seed


def train_one(cfg: dict, seed: int, device: torch.device):
    set_global_seed(seed)
    data = load_dataset(resolve_path(cfg["paths"]["processed_dir"]))
    loaders = make_loaders(data, cfg["training"]["batch_size"], seed=seed)
    class_weights = (
        compute_class_weights(data["y_train"], cfg["model"]["num_classes"])
        if cfg["training"]["use_class_weight"] else None
    )
    model = MLP(
        input_dim=int(data["X_train"].shape[1]),
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        use_batchnorm=cfg["model"]["use_batchnorm"],
    )
    ckpt = resolve_path("results/models") / f"clean_ref_seed{seed}.pt"
    info = train_model(
        model, loaders["train"], loaders["val"],
        class_weights=class_weights,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["patience"],
        min_delta=cfg["training"].get("min_delta", 0.0),
        monitor=cfg["training"]["monitor"],
        device=device,
        checkpoint_path=ckpt,
        verbose=False,
    )
    print(f"  seed {seed}: best epoch {info['best_epoch']}, "
          f"val {info['monitor']}={info['best_score']:.4f} -> {ckpt.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train clean reference IDS models with different seeds")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--seeds", default="43,44,45,46,47", help="comma-separated training seeds")
    args = p.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    device = torch.device("cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Training {len(seeds)} reference clean models (identical Phase 2 protocol) ...")
    for s in seeds:
        train_one(cfg, s, device)
    print("EXIT_OK")


if __name__ == "__main__":
    main()
