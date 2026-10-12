"""Train the additional clean and poisoned models of a v1.1 experiment.

Reads a v1.1 experiment config (``configs/v11_screening.yaml`` for the MLP,
``configs/v11_cnn.yaml`` for the 1D-CNN) and, for every dataset, trains one clean
and one poisoned model per new seed with the paper's training protocol. The run is
resumable: existing checkpoints are skipped, checkpoints are written atomically,
progress goes to ``<log_dir>/<experiment>_train.log``, and the seed-to-checkpoint
manifest (``models_manifest.json`` in the experiment folder) is rewritten after
every model.

``--reproduce-check`` first retrains the first existing clean model of each
dataset (MLP configs only) into a scratch folder and compares its weights with
the committed checkpoint, to confirm that this trainer reproduces the paper's
protocol exactly.

Run:
    python -u -m src.eval.v11_train_models --config configs/v11_screening.yaml
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np
import torch

from src.eval.v11_common import (
    ProgressLog,
    environment,
    git_commit,
    load_dataset_trigger,
    load_training_data,
    models_manifest_entry,
    poisoned_training_set,
    sha256_file,
    train_new_model,
    utc_now,
    write_json,
)
from src.utils import load_yaml, resolve_path


def _reproduce_check(dname, dc, data, cfg, mcfg, md, log):
    """Retrain the first existing clean model and compare it with the checkpoint."""
    name, seed = dc["existing"]["clean"][0], dc["existing"]["seeds"][0]
    scratch = md / "_v11_repro_check"
    scratch.mkdir(exist_ok=True)
    ckpt = scratch / f"{name}.pt"
    if ckpt.exists():
        ckpt.unlink()
    log(f"{dname}: reproduce-check retraining {name} (seed {seed})")
    info = train_new_model(ckpt, data["X_train"], data["y_train"], data, seed, cfg["arch"],
                           mcfg, dc["training"])
    a = torch.load(md / f"{name}.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    max_diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    identical = all(torch.equal(a[k], b[k]) for k in a)
    shutil.rmtree(scratch)
    log(f"{dname}: reproduce-check identical={identical} max|diff|={max_diff:.3e}")
    return {"model": name, "seed": seed, "identical_weights": identical,
            "max_abs_weight_diff": max_diff, "train_time_s": info.get("train_time_s")}


def run(cfg: dict, reproduce_check: bool = False, only: list | None = None) -> dict:
    md = resolve_path(cfg["paths"]["models_dir"])
    md.mkdir(parents=True, exist_ok=True)
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_name = exp_dir.name
    log = ProgressLog(resolve_path(cfg["paths"]["log_dir"]) / f"{exp_name}_train.log")
    mcfg = load_yaml(resolve_path(cfg["model_config"]))
    man_path = exp_dir / "models_manifest.json"
    manifest = {"config": cfg.get("_config_path"), "arch": cfg["arch"],
                "model_config": cfg["model_config"], "datasets": {},
                "environment": environment(), "git_commit": git_commit()}
    if man_path.exists():
        import json

        old = json.loads(man_path.read_text(encoding="utf-8"))
        manifest["datasets"] = old.get("datasets", {})
        manifest["reproduce_check"] = old.get("reproduce_check", {})

    log(f"start {exp_name} training (arch={cfg['arch']})")
    for dname, dc in cfg["datasets"].items():
        if only and dname not in only:
            continue
        log(f"{dname}: loading data")
        data = load_training_data(dc)
        trigger = load_dataset_trigger(dc, data["feature_names"], data["X_train"])
        Xp, yp, npois = poisoned_training_set(data, trigger, dc["poison"])
        log(f"{dname}: train rows {len(data['X_train'])}, benign/normal frac "
            f"{float(np.mean(data['y_train'] == 0)):.3f}, poisoned rows {npois}")

        if reproduce_check and cfg["arch"] == "mlp":
            manifest.setdefault("reproduce_check", {})[dname] = _reproduce_check(
                dname, dc, data, cfg, mcfg, md, log)
            write_json(man_path, manifest)

        dm = manifest["datasets"].setdefault(dname, {"existing": [], "new": []})
        dm["poison"] = dc["poison"]
        dm["n_poisoned_training_rows"] = int(npois)
        dm["training"] = dc["training"]
        if "existing" in dc:
            dm["existing"] = [
                models_manifest_entry(n, md, role=role, seed=s, status="pre-existing")
                for role in ("clean", "poisoned")
                for n, s in zip(dc["existing"][role], dc["existing"]["seeds"])]
        new_by_name = {e["name"]: e for e in dm["new"]}
        for s in dc["new"]["seeds"]:
            for role, (X, y, prefix) in (
                    ("clean", (data["X_train"], data["y_train"], dc["new"]["clean_prefix"])),
                    ("poisoned", (Xp, yp, dc["new"]["poisoned_prefix"]))):
                name = f"{prefix}{s}"
                ckpt = md / f"{name}.pt"
                log(f"{dname}: {name} ({role}, seed {s}) ...")
                info = train_new_model(ckpt, X, y, data, s, cfg["arch"], mcfg, dc["training"])
                if info["status"] == "exists":
                    log(f"{dname}: {name} exists, skipped")
                    entry = new_by_name.get(name) or models_manifest_entry(
                        name, md, role=role, seed=s, status="trained (earlier run)")
                    entry["sha256"] = sha256_file(ckpt)
                else:
                    log(f"{dname}: {name} trained in {info['train_time_s']:.1f}s "
                        f"(best epoch {info['best_epoch']}/{info['epochs_run']}, "
                        f"val PR-AUC {info['best_score']:.4f})")
                    entry = models_manifest_entry(name, md, role=role, seed=s,
                                                  status="trained", trained_at=utc_now(),
                                                  **{k: v for k, v in info.items()
                                                     if k != "status"})
                new_by_name[name] = entry
                dm["new"] = list(new_by_name.values())
                write_json(man_path, manifest)
    log(f"done {exp_name} training")
    return manifest


def parse_args():
    ap = argparse.ArgumentParser(description="Train the additional models of a v1.1 experiment")
    ap.add_argument("--config", default="configs/v11_screening.yaml")
    ap.add_argument("--reproduce-check", action="store_true",
                    help="retrain the first existing clean model per dataset and compare weights")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset of datasets (default: all)")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = load_yaml(resolve_path(a.config))
    cfg["_config_path"] = a.config
    run(cfg, reproduce_check=a.reproduce_check,
        only=a.datasets.split(",") if a.datasets else None)


if __name__ == "__main__":
    main()
