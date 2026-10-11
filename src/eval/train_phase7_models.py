"""Train the extra poisoned IDS models used by the comparison, calibration, and
statistics runs (multi-seed results in Section 3.1).

Two families (all at poison rate 5%, saved to results/models/, RESUMABLE — a
model whose checkpoint already exists is skipped):

* multi-seed: the baseline trigger trained with 5 different seeds (43-47) to give
  a variability estimate for the headline metrics.
* trigger-variation sweep (seed 42): trigger feature count {2, 6, 8}, an
  alternate controllable feature subset, and a subtler (closer-to-normal)
  intensity.

The 4-feature baseline (seed 42) and the poison-rate sweep (0.5-10%) are reused
from the attack experiment (``run_backdoor``). A lighter protocol (max_epochs 15) keeps the sweep tractable on CPU;
val PR-AUC is within ~1e-3 of the full protocol by epoch ~12.

Run:
    python -m src.eval.train_phase7_models --config configs/model.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.attacks.run_backdoor import make_poisoned_train
from src.attacks.trigger import build_trigger
from src.models.mlp import MLP
from src.models.train import (
    compute_class_weights,
    load_dataset,
    make_loaders,
    train_model,
)
from src.utils import load_yaml, resolve_path, set_global_seed

RATE = 0.05
MAX_EPOCHS = 15
PATIENCE = 4

# Trigger variations (constant, attacker-controllable forward features).
BASELINE = {"Fwd Packet Length Max": 500.0, "Fwd Packet Length Min": 20.0,
            "Fwd IAT Mean": 50.0, "act_data_pkt_fwd": 3.0}
VARIANTS = {
    "nfeat2": {"Fwd Packet Length Max": 500.0, "Fwd IAT Mean": 50.0},
    "nfeat6": {**BASELINE, "Fwd IAT Min": 5.0, "Fwd Header Length": 40.0},
    "nfeat8": {**BASELINE, "Fwd IAT Min": 5.0, "Fwd Header Length": 40.0,
               "min_seg_size_forward": 20.0, "Fwd Packet Length Mean": 100.0},
    "altfeat": {"Fwd Header Length": 40.0, "Fwd IAT Min": 5.0,
                "min_seg_size_forward": 20.0, "Fwd Packet Length Mean": 100.0},
    "subtle": {"Fwd Packet Length Max": 120.0, "Fwd Packet Length Min": 8.0,
               "Fwd IAT Mean": 49.0, "act_data_pkt_fwd": 2.0},
}


def build_jobs():
    jobs = []
    for seed in (43, 44, 45, 46, 47):
        jobs.append({"name": f"multiseed_seed{seed}", "seed": seed,
                     "constants": BASELINE, "group": "multiseed"})
    for name, consts in VARIANTS.items():
        jobs.append({"name": name, "seed": 42, "constants": consts,
                     "group": "trigger_variant"})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the multi-seed and trigger-variant backdoored models")
    ap.add_argument("--config", default="configs/model.yaml")
    args = ap.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    device = torch.device("cpu")
    import joblib

    data = load_dataset(resolve_path("data/processed"))
    scaler = joblib.load(resolve_path("data/processed/scaler.joblib"))
    feat_names = [str(f) for f in data["feature_names"]]
    models_dir = resolve_path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = resolve_path("experiments/phase7")
    exp_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = exp_dir / "models_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for job in build_jobs():
        name = job["name"]
        ckpt = models_dir / f"phase7_{name}.pt"
        trig = build_trigger(feat_names, scaler, data["X_train"],
                             list(job["constants"].keys()), mode="constant",
                             constants=job["constants"])
        manifest[name] = {
            "checkpoint": f"results/models/phase7_{name}.pt",
            "seed": job["seed"], "rate": RATE, "group": job["group"],
            "trigger_features": trig.feature_names,
            "trigger_indices": trig.feature_indices,
            "trigger_original": trig.original_values,
            "trigger_scaled": trig.scaled_values,
        }
        if ckpt.exists():
            print(f"  skip {name} (exists)")
            continue
        print(f"  training {name} (seed {job['seed']}, "
              f"{len(trig.feature_names)} trigger features) ...")
        set_global_seed(job["seed"])
        rng = np.random.default_rng(job["seed"])
        Xp, yp, npois = make_poisoned_train(
            data["X_train"], data["y_train"], trig, RATE, 1, 0, rng)
        pois = {"X_train": Xp, "y_train": yp, "X_val": data["X_val"],
                "y_val": data["y_val"], "X_test": data["X_test"], "y_test": data["y_test"]}
        loaders = make_loaders(pois, cfg["training"]["batch_size"], seed=job["seed"])
        cw = compute_class_weights(yp, cfg["model"]["num_classes"])
        model = MLP(input_dim=data["X_train"].shape[1],
                    hidden_dims=cfg["model"]["hidden_dims"],
                    num_classes=cfg["model"]["num_classes"],
                    dropout=cfg["model"]["dropout"],
                    use_batchnorm=cfg["model"]["use_batchnorm"])
        info = train_model(model, loaders["train"], loaders["val"], class_weights=cw,
                           learning_rate=cfg["training"]["learning_rate"],
                           weight_decay=cfg["training"]["weight_decay"],
                           max_epochs=MAX_EPOCHS, patience=PATIENCE,
                           min_delta=cfg["training"].get("min_delta", 0.0),
                           monitor=cfg["training"]["monitor"], device=device,
                           checkpoint_path=ckpt, verbose=False)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"    done {name}: best epoch {info['best_epoch']} "
              f"({npois} poisoned) -> {ckpt.name}")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MANIFEST -> {manifest_path}")
    print("EXIT_OK")


if __name__ == "__main__":
    main()
