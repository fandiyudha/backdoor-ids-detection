"""Per-class STRIP entropy on clean CICIDS2017 models (Figure 3, panel b).

Measures the mean STRIP prediction entropy of clean BENIGN and clean ATTACK test
inputs on the five clean reference models (no training), using the same function
and protocol that produced the UNSW-NB15 clean-model values in
``run_unsw_replication``: a 3000-row clean holdout drawn from the validation split
with ``default_rng(seed)``, blend alpha 0.7, N = 100 perturbations, and the first
2000 clean test inputs of each class. Reports mean ± sd over models.

Outputs: ``experiments/clean_entropy/`` (metrics.json, summary.md).

Run:
    python -m src.eval.run_clean_entropy --config configs/clean_entropy.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.eval.run_comparison import load_ids_model
from src.eval.run_phase7c_stats import ms
from src.eval.run_unsw_replication import strip_entropy_by_class
from src.models.train import load_dataset
from src.utils import load_yaml, resolve_path, set_global_seed


def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(resolve_path(cfg["paths"]["processed_dir"]))
    Xte, yte = data["X_test"], data["y_test"]
    rng = np.random.default_rng(cfg["seed"])
    holdout = data["X_val"][rng.choice(len(data["X_val"]), cfg["strip"]["holdout_size"],
                                       replace=False)].astype(np.float32)
    es = cfg["strip"]["eval_size"]

    per_model, ent_benign, ent_attack = {}, [], []
    for name in cfg["reference_models"]:
        m = load_ids_model(resolve_path(cfg["paths"]["models_dir"]) / f"{name}.pt", device)
        e0, e1 = strip_entropy_by_class(m, holdout, Xte, yte, cfg, device, es)
        per_model[name] = {"ent_benign": e0, "ent_attack": e1}
        ent_benign.append(e0); ent_attack.append(e1)
        print(f"  {name}: benign {e0:.4f} | attack {e1:.4f}")

    out = {"protocol": {"holdout_size": cfg["strip"]["holdout_size"], "eval_size_per_class": es,
                        "N": cfg["strip"]["N"], "alpha": cfg["strip"]["alpha"], "seed": cfg["seed"],
                        "same_function_as": "src.eval.run_unsw_replication.strip_entropy_by_class"},
           "per_model": per_model,
           "ent_benign": ms(ent_benign), "ent_attack": ms(ent_attack)}
    (exp_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _summary(exp_dir / "summary.md", out)
    print(f"  benign {out['ent_benign']['mean']:.4f} ± {out['ent_benign']['std']:.4f} | "
          f"attack {out['ent_attack']['mean']:.4f} ± {out['ent_attack']['std']:.4f}")
    return out


def _summary(path, out):
    p = out["protocol"]
    L = ["# Per-class STRIP entropy on clean CICIDS2017 models", "",
         "Mean STRIP prediction entropy of clean test inputs, per class, on the clean "
         "reference models (no training). Same function and protocol as the UNSW-NB15 "
         "clean-model measurement (`run_unsw_replication`).", "",
         "- **Run:** `python -m src.eval.run_clean_entropy --config configs/clean_entropy.yaml`",
         f"- **Measured:** STRIP entropy (blend, alpha {p['alpha']}, N = {p['N']}, clean holdout "
         f"of {p['holdout_size']}) on the first {p['eval_size_per_class']} clean test inputs "
         "of each class; mean ± sd over models.", "",
         "| model | benign | attack |", "|---|---|---|"]
    for n, r in out["per_model"].items():
        L.append(f"| {n} | {r['ent_benign']:.4f} | {r['ent_attack']:.4f} |")
    b, a = out["ent_benign"], out["ent_attack"]
    L += [f"| **mean ± sd** | **{b['mean']:.4f} ± {b['std']:.4f}** | "
          f"**{a['mean']:.4f} ± {a['std']:.4f}** |", "",
          "## Artifacts", "- `metrics.json`, `summary.md`"]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-class STRIP entropy on clean CICIDS2017 models")
    ap.add_argument("--config", default="configs/clean_entropy.yaml")
    run(load_yaml(resolve_path(ap.parse_args().config)))


if __name__ == "__main__":
    main()
