"""Shared helpers for the v1.1 extended experiments (``experiments/v11_*/``).

Everything here reuses the paper's pipeline unchanged: the same preprocessed
splits and scaler, the same training subsample and poisoning routine, the same
trigger, the same training loop and early stopping, and the same detectors. This
module adds only what the extended experiments need on top:

* protocol-identical data and trigger loading per dataset;
* resumable training of new seeds (existing checkpoints are never overwritten);
* checkpoint manifests (seed -> checkpoint -> SHA-256);
* a replay of the ``phase7e`` holdout draw, so new models are scored on exactly
  the holdout the paper used;
* the statistics used for every headline number: stratified bootstrap CIs,
  Mann-Whitney U with effect sizes, and exact binomial intervals for rates.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.stats import beta, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from src.attacks.run_backdoor import make_poisoned_train
from src.attacks.trigger import Trigger, build_trigger, load_trigger
from src.eval.run_causal_test_ids import subsample
from src.eval.run_unsw_replication import load_unsw
from src.models.registry import build_model, load_model
from src.models.train import (
    compute_class_weights,
    compute_metrics,
    load_dataset,
    make_loaders,
    train_model,
)
from src.utils import load_yaml, resolve_path, set_global_seed

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Data, trigger, and training sets (identical to the paper's training scripts)
# ---------------------------------------------------------------------------
def load_training_data(dc: dict) -> Dict[str, np.ndarray]:
    """Return train/val/test arrays exactly as the paper's model scripts saw them.

    ``cicids_controlled``: the controlled-experiment training subsample of
    ``run_causal_test_ids`` (natural class ratio, fixed subsample seed) with the
    preprocessed validation and test splits. ``unsw``: the official training
    partition with a stratified validation split carved from it, as in
    ``run_unsw_replication``.
    """
    kind = dc["kind"]
    if kind == "cicids_controlled":
        data = load_dataset(resolve_path(dc["processed_dir"]))
        Xtr, ytr = data["X_train"], data["y_train"]
        benign_frac = float(np.mean(ytr == 0))
        Xs, ys = subsample(Xtr, ytr, dc["subsample_mode"], dc["train_subsample"],
                           benign_frac, np.random.default_rng(dc["subsample_seed"]))
        return {"X_train": Xs, "y_train": ys, "X_val": data["X_val"], "y_val": data["y_val"],
                "X_test": data["X_test"], "y_test": data["y_test"],
                "feature_names": [str(f) for f in data["feature_names"]]}
    if kind == "unsw":
        return load_unsw({"paths": {"processed_dir": dc["processed_dir"]},
                          "val_frac": dc["val_frac"], "seed": dc["split_seed"]})
    raise ValueError(f"unknown dataset kind {kind!r}")


def load_dataset_trigger(dc: dict, feature_names: Sequence[str],
                         X_train: np.ndarray) -> Trigger:
    """The paper's trigger for this dataset (saved JSON or rebuilt from constants)."""
    if "trigger_json" in dc:
        return load_trigger(resolve_path(dc["trigger_json"]))
    import joblib

    scaler = joblib.load(resolve_path(dc["scaler"]))
    tb = dc["trigger_build"]
    return build_trigger(list(feature_names), scaler, X_train, tb["features"],
                         mode="constant", constants=tb["constants"],
                         enforce_controllable=False)


def poisoned_training_set(data: dict, trigger: Trigger, poison: dict):
    """Poison the training split with the paper's routine and fixed poisoning seed."""
    return make_poisoned_train(data["X_train"], data["y_train"], trigger, poison["rate"],
                               poison["source"], poison["target"],
                               np.random.default_rng(poison["seed"]))


def model_names(dc: dict) -> Dict[str, List[str]]:
    """Existing + new model names per role, in the pre-specified order."""
    new = dc["new"]
    existing = dc.get("existing", {"clean": [], "poisoned": []})
    return {
        "clean": list(existing["clean"])
        + [f"{new['clean_prefix']}{s}" for s in new["seeds"]],
        "poisoned": list(existing["poisoned"])
        + [f"{new['poisoned_prefix']}{s}" for s in new["seeds"]],
    }


# ---------------------------------------------------------------------------
# Resumable training
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_new_model(ckpt: Path, X_tr: np.ndarray, y_tr: np.ndarray, data: dict, seed: int,
                    arch: str, mcfg: dict, tcfg: dict) -> dict:
    """Train one model with the paper's protocol; skip it if the checkpoint exists.

    The call sequence (seed, loaders, class weights, model init, training) is the
    same as ``train_or_load`` in ``run_causal_test_ids`` and
    ``run_unsw_replication``, so an MLP trained here with an existing seed
    reproduces that existing checkpoint. The checkpoint is written to a temporary
    file and renamed, so an interrupted run never leaves a partial checkpoint.
    """
    if ckpt.exists():
        return {"status": "exists"}
    set_global_seed(seed)
    dd = {"X_train": X_tr, "y_train": y_tr, "X_val": data["X_val"], "y_val": data["y_val"],
          "X_test": data["X_test"], "y_test": data["y_test"]}
    loaders = make_loaders(dd, mcfg["training"]["batch_size"], seed=seed)
    cw = compute_class_weights(y_tr, mcfg["model"]["num_classes"])
    m = build_model(arch, mcfg["model"], X_tr.shape[1])
    tmp = ckpt.with_name(ckpt.name + ".tmp")
    t0 = time.perf_counter()
    res = train_model(m, loaders["train"], loaders["val"], class_weights=cw,
                      learning_rate=mcfg["training"]["learning_rate"],
                      weight_decay=mcfg["training"]["weight_decay"],
                      max_epochs=tcfg["max_epochs"], patience=tcfg["patience"],
                      min_delta=mcfg["training"].get("min_delta", 0.0),
                      monitor=mcfg["training"]["monitor"], device=CPU,
                      checkpoint_path=tmp, verbose=False)
    os.replace(tmp, ckpt)
    return {"status": "trained", "train_time_s": time.perf_counter() - t0,
            "best_epoch": res["best_epoch"], "best_score": res["best_score"],
            "epochs_run": len(res["history"])}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i:i + batch], np.float32))
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def clean_performance(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy, attack recall/F1, ROC-AUC and PR-AUC on clean data (paper's metric suite)."""
    logits = predict_logits(model, X)
    prob = torch.softmax(torch.from_numpy(logits), 1)[:, 1].numpy()
    m = compute_metrics(np.asarray(y), prob, (prob >= 0.5).astype(np.int64))
    return {k: m[k] for k in ("accuracy", "attack_recall", "attack_f1", "roc_auc", "pr_auc")}


# ---------------------------------------------------------------------------
# Holdouts of the paper's final protocols
# ---------------------------------------------------------------------------
def phase7e_holdouts(phase7e_cfg_path: str) -> Dict[str, dict]:
    """Replay the holdout draw of ``run_phase7e_nc_reeval`` exactly.

    ``phase7e`` draws each dataset's 3,000-row holdout from one generator and
    consumes that generator in its bootstrap before drawing the next dataset's
    holdout. Replaying the same calls reproduces the identical holdout rows.
    """
    cfg = load_yaml(resolve_path(phase7e_cfg_path))
    rng = np.random.default_rng(cfg["seed"])
    out = {}
    for dname, dc in cfg["datasets"].items():
        data = load_dataset(resolve_path(dc["processed_dir"]))
        hk, yk = ("X_val", "y_val") if "X_val" in data else ("X_test", "y_test")
        vidx = rng.choice(len(data[hk]), min(cfg["nc"]["holdout_size"], len(data[hk])),
                          replace=False)
        out[dname] = {"X": data[hk][vidx].astype(np.float32),
                      "y": data[yk][vidx].astype(int), "source_split": hk,
                      "indices": vidx}
        nc = len(dc["clean"])
        for npo in (len(dc["poisoned_target0"]), len(dc["poisoned_target1"]),
                    len(dc["poisoned_target0"]) + len(dc["poisoned_target1"])):
            for _ in range(cfg["bootstrap_iters"]):
                rng.integers(0, nc, nc)
                rng.integers(0, npo, npo)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def auc_boot(clean: Sequence[float], pois: Sequence[float], iters: int, seed: int) -> dict:
    """ROC-AUC and PR-AUC (poisoned = positive, higher score = more suspicious)
    with stratified bootstrap 95% CIs (clean and poisoned resampled separately)."""
    clean, pois = np.asarray(clean, float), np.asarray(pois, float)
    nc, npo = len(clean), len(pois)
    y = np.r_[np.zeros(nc), np.ones(npo)]
    rng = np.random.default_rng(seed)
    rocs, prs = np.empty(iters), np.empty(iters)
    for i in range(iters):
        s = np.r_[clean[rng.integers(0, nc, nc)], pois[rng.integers(0, npo, npo)]]
        rocs[i] = roc_auc_score(y, s)
        prs[i] = average_precision_score(y, s)
    s = np.r_[clean, pois]
    return {"roc_auc": float(roc_auc_score(y, s)),
            "roc_ci95": [float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5))],
            "pr_auc": float(average_precision_score(y, s)),
            "pr_ci95": [float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))],
            "n_clean": nc, "n_poisoned": npo, "bootstrap_iters": iters}


def mann_whitney(clean: Sequence[float], pois: Sequence[float]) -> dict:
    """Mann-Whitney U (poisoned vs clean) with effect sizes.

    ``U`` counts (poisoned, clean) pairs with poisoned > clean (ties 0.5). Effect
    sizes: common-language effect size ``U / (n1 n2)`` (equal to the ROC-AUC) and
    rank-biserial correlation ``2 U / (n1 n2) - 1``.
    """
    clean, pois = np.asarray(clean, float), np.asarray(pois, float)
    g = mannwhitneyu(pois, clean, alternative="greater")
    t = mannwhitneyu(pois, clean, alternative="two-sided")
    cl = float(g.statistic) / (len(pois) * len(clean))
    return {"U": float(g.statistic), "p_one_sided_greater": float(g.pvalue),
            "p_two_sided": float(t.pvalue), "common_language_effect": cl,
            "rank_biserial": 2 * cl - 1, "n_clean": len(clean), "n_poisoned": len(pois)}


def rate_ci(flags: Sequence[bool], iters: int, seed: int) -> dict:
    """A rate with a bootstrap 95% CI and an exact Clopper-Pearson 95% interval.

    The bootstrap interval collapses to a point when every flag is equal (e.g.
    0 of 5); the Clopper-Pearson interval is reported alongside for that reason.
    """
    f = np.asarray(flags, bool)
    n, k = len(f), int(f.sum())
    rng = np.random.default_rng(seed)
    boots = f[rng.integers(0, n, (iters, n))].mean(axis=1) if n else np.zeros(1)
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return {"rate": float(k / n) if n else float("nan"), "k": k, "n": n,
            "boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "clopper_pearson95": [lo, hi]}


def mean_ci(values: Sequence[float], iters: int, seed: int) -> dict:
    """Mean with sample SD and a bootstrap 95% CI over the listed units (models)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (iters, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": int(len(v)), "values": [float(x) for x in v]}


def reference_relative_rule(clean: Sequence[float], pois: Sequence[float], calib_n: int,
                            factor: float, iters: int, seed: int) -> dict:
    """The paper's paired reference-relative screening rule.

    Flag a model if its statistic exceeds ``factor`` x the median statistic of the
    calibration clean models. The first ``calib_n`` clean models are the
    calibration group; the remaining clean models are a disjoint test group used
    only to estimate the false-positive rate. Leave-one-out: each clean model is
    compared with the median of all *other* clean models (FPR); each poisoned
    model with the median of all clean models (TPR).
    """
    clean, pois = np.asarray(clean, float), np.asarray(pois, float)
    calib, test = clean[:calib_n], clean[calib_n:]
    thr = factor * float(np.median(calib))
    loo_clean = [clean[i] > factor * float(np.median(np.delete(clean, i)))
                 for i in range(len(clean))]
    loo_thr = factor * float(np.median(clean))
    return {"rule": f"flag if statistic > {factor} x median(calibration clean statistics)",
            "n_calibration": int(len(calib)), "n_test": int(len(test)),
            "threshold": thr,
            "fpr_test_group": rate_ci(test > thr, iters, seed),
            "tpr": rate_ci(pois > thr, iters, seed + 1),
            "fpr_leave_one_out": rate_ci(loo_clean, iters, seed + 2),
            "tpr_leave_one_out": rate_ci(pois > loo_thr, iters, seed + 3),
            "flags_poisoned": [bool(v) for v in pois > thr],
            "flags_clean_test": [bool(v) for v in test > thr]}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=resolve_path("."), check=True).stdout.strip()
    except Exception:  # pragma: no cover - git missing
        return None


def environment() -> dict:
    import scipy
    import sklearn

    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "torch_threads": torch.get_num_threads(),
            "platform": platform.platform()}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def models_manifest_entry(name: str, models_dir: Path, **fields) -> dict:
    p = models_dir / f"{name}.pt"
    return {"name": name, "checkpoint": f"results/models/{name}.pt",
            "sha256": sha256_file(p) if p.exists() else None, **fields}


class ProgressLog:
    """Append-only progress log (one line per event, flushed immediately)."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def load_model_cpu(models_dir: Path, name: str):
    return load_model(models_dir / f"{name}.pt", CPU)
