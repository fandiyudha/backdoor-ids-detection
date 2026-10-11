"""Neural Cleanse detector experiment on the CICIDS2017 models (outputs ``experiments/phase5/``).

Run:
    python -m src.detectors.run_neural_cleanse --config configs/neural_cleanse.yaml

For every model (clean control + all poisoned rates) it reverse-engineers a
mask+pattern per class, reports the L1 mask norms and the (binary-safe) L1-ratio
anomaly score, checks how well the optimized mask recovers the TRUE attack
trigger features, and tests whether the reverse-engineered trigger actually
flips clean attacks to benign. Costs are per-model (contrast: STRIP is
per-input).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.attacks.trigger import load_trigger  # noqa: E402
from src.detectors.neural_cleanse import (  # noqa: E402
    BENIGN_CLASS,
    NeuralCleanseDetector,
    apply_reverse_trigger,
    trigger_recovery,
)
from src.models.mlp import MLP  # noqa: E402
from src.models.train import load_dataset  # noqa: E402
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


def load_model(path: Path, device: torch.device) -> MLP:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLP.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_l1_norms(results: Dict[str, dict], out_path: Path):
    names = list(results.keys())
    benign = [float(results[n]["l1_norms"]["0"]) for n in names]
    attack = [float(results[n]["l1_norms"]["1"]) for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, benign, w, label="mask L1 — target BENIGN", color="#c0392b")
    ax.bar(x + w / 2, attack, w, label="mask L1 — target ATTACK", color="#2980b9")
    ax.set_xticks(x, labels=names, rotation=20)
    ax.set_ylabel("L1 norm of reverse-engineered mask")
    ax.set_title("Neural Cleanse — mask size per class (small = infected)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_mask_map(mask, true_idx, feature_names, out_path: Path, title: str):
    D = len(mask)
    colors = ["#c0392b" if i in set(true_idx) else "#95a5a6" for i in range(D)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(np.arange(D), mask, color=colors)
    ax.set_xlabel("feature index (red = true Phase-3 trigger feature)")
    ax.set_ylabel("mask value")
    ax.set_title(title)
    # annotate the true trigger features
    for i in true_idx:
        ax.annotate(feature_names[i], (i, mask[i]), fontsize=7, rotation=90,
                    ha="center", va="bottom", color="#c0392b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data, scaler, trigger ...")
    data = load_dataset(resolve_path(cfg["paths"]["processed_dir"]))
    import joblib

    scaler = joblib.load(resolve_path(cfg["paths"]["scaler"]))
    trigger = load_trigger(resolve_path(cfg["paths"]["trigger"]))
    true_idx = list(trigger.feature_indices)
    feature_names = [str(f) for f in data["feature_names"]]
    attack_cfg = load_yaml(resolve_path(cfg["paths"]["attack_config"]))
    rates = attack_cfg["poison"]["rates"]

    rng = np.random.default_rng(cfg["seed"])
    Xval, yval = data["X_val"], data["y_val"]
    hidx = rng.choice(len(Xval), size=min(cfg["holdout"]["size"], len(Xval)), replace=False)
    holdout_X, holdout_y = Xval[hidx].astype(np.float32), yval[hidx].astype(int)
    print(f"  clean holdout: {len(holdout_X)} rows "
          f"({int((holdout_y==0).sum())} benign / {int((holdout_y==1).sum())} attack)")

    # clean attack rows from TEST for reverse-engineered-trigger effectiveness
    Xte, yte = data["X_test"], data["y_test"]
    atk_test = Xte[yte == 1]
    n_eff = min(cfg["eval"]["re_trigger_eval_size"], len(atk_test))
    atk_eval = atk_test[rng.choice(len(atk_test), size=n_eff, replace=False)]

    opt = cfg["optimization"]
    model_specs = [("clean", None, resolve_path(cfg["paths"]["clean_checkpoint"]))]
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    for rt in rates:
        model_specs.append((f"p{rt:g}", rt, models_dir / f"backdoor_mlp_p{rt:g}.pt"))

    results: Dict[str, dict] = {}
    masks_store = {}
    print("\nRunning Neural Cleanse per model ...")
    for name, rt, path in model_specs:
        model = load_model(path, device)
        det = NeuralCleanseDetector(
            model, holdout_X, holdout_y, device=device,
            steps=opt["steps"], lr=opt["lr"], init_lambda=opt["init_lambda"],
            success_threshold=opt["success_threshold"], lambda_up=opt["lambda_up"],
            lambda_down=opt["lambda_down"], lambda_update_every=opt["lambda_update_every"],
            lambda_min=opt["lambda_min"], lambda_max=opt["lambda_max"],
            bound_percentile=opt["bound_percentile"], seed=cfg["seed"],
        )
        out = det.run()

        benign_mask = out["per_class"][BENIGN_CLASS]["mask"]
        benign_pattern = out["per_class"][BENIGN_CLASS]["pattern"]
        # trigger recovery on the BENIGN-target mask (the backdoor's true target)
        recovery = {str(k): trigger_recovery(benign_mask, true_idx, k)
                    for k in cfg["recovery"]["top_k"]}
        # effectiveness of the reverse-engineered benign trigger on clean attacks
        re_eff = apply_reverse_trigger(model, atk_eval, benign_mask, benign_pattern,
                                       device, target_class=BENIGN_CLASS)

        flagged = (out["anomaly_ratio"] >= cfg["decision"]["ratio_threshold"]
                   and out["infected_class"] == BENIGN_CLASS)
        results[name] = {
            "poison_rate": rt,
            "l1_norms": {str(k): v for k, v in out["l1_norms"].items()},
            "anomaly_ratio": out["anomaly_ratio"],
            "ratio_attack_over_benign": out["ratio_attack_over_benign"],
            "infected_class": out["infected_class"],
            "flagged_backdoored": bool(flagged),
            "reversal_success": {str(c): out["per_class"][c]["reversal_success"]
                                 for c in out["per_class"]},
            "trigger_recovery": recovery,
            "re_trigger_flip_rate": re_eff,
            "time_s": out["total_time_s"],
        }
        masks_store[f"{name}_benign_mask"] = benign_mask
        masks_store[f"{name}_benign_pattern"] = benign_pattern
        masks_store[f"{name}_attack_mask"] = out["per_class"][1]["mask"]

        rec4 = recovery[str(cfg["recovery"]["top_k"][0])]
        print(f"  {name:>7} (rate={rt}): L1 benign={out['l1_norms'][0]:.2f} "
              f"attack={out['l1_norms'][1]:.2f} | ratio={out['anomaly_ratio']:.2f} "
              f"| infected={'BENIGN' if out['infected_class']==0 else 'ATTACK'} "
              f"| recover@{rec4['k']}={rec4['hits']}/{len(true_idx)} "
              f"| RE-flip={re_eff:.3f} | {out['total_time_s']:.1f}s")

    # --- pattern in original units for the most-anomalous poisoned model ---
    rep = max((n for n in results if results[n]["poison_rate"] is not None),
              key=lambda n: results[n]["anomaly_ratio"])
    rep_mask = masks_store[f"{rep}_benign_mask"]
    rep_pattern = masks_store[f"{rep}_benign_pattern"]
    orig_pattern = rep_pattern * scaler.scale_ + scaler.mean_

    # --- plots ---
    plot_l1_norms(results, exp_dir / "l1_norms_per_class.png")
    plot_mask_map(rep_mask, true_idx, feature_names, exp_dir / "mask_map.png",
                  f"Neural Cleanse mask (target BENIGN) — {rep}: "
                  "true trigger features in red")

    _save_artifacts(exp_dir, cfg, results, masks_store, true_idx, feature_names,
                    rep, orig_pattern, trigger)
    _print_summary(results, true_idx, rep)
    return results


# ---------------------------------------------------------------------------
def _table(results, true_n):
    header = ["model", "rate", "L1_benign", "L1_attack", "anom_ratio",
              "infected", "flagged", "recover@4", "RE_flip", "time_s"]
    rows = []
    for n, r in results.items():
        rec4 = r["trigger_recovery"]["4"]
        rows.append([
            n, str(r["poison_rate"]),
            f"{float(r['l1_norms']['0']):.2f}", f"{float(r['l1_norms']['1']):.2f}",
            f"{r['anomaly_ratio']:.2f}",
            "BENIGN" if r["infected_class"] == 0 else "ATTACK",
            "yes" if r["flagged_backdoored"] else "no",
            f"{rec4['hits']}/{true_n}",
            f"{r['re_trigger_flip_rate']:.3f}",
            f"{r['time_s']:.1f}",
        ])
    return header, rows


def _save_artifacts(exp_dir, cfg, results, masks_store, true_idx, feature_names,
                    rep, orig_pattern, trigger):
    metrics = {
        "decision_rule": "L1 mask-norm ratio (MAD needs >=3 classes; binary here)",
        "ratio_threshold": cfg["decision"]["ratio_threshold"],
        "true_trigger_features": trigger.feature_names,
        "true_trigger_indices": true_idx,
        "representative_model": rep,
        "results": results,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # masks + patterns for reuse by the evaluation scripts
    np.savez_compressed(exp_dir / "masks.npz",
                        feature_names=np.array(feature_names),
                        true_trigger_indices=np.array(true_idx), **masks_store)

    header, rows = _table(results, len(true_idx))
    csv = [",".join(header)] + [",".join(r) for r in rows]
    (exp_dir / "results_table.csv").write_text("\n".join(csv), encoding="utf-8")

    # original-space pattern on the recovered trigger features (rep model)
    rep_mask = masks_store[f"{rep}_benign_mask"]
    top = np.argsort(rep_mask)[::-1][:8]
    recovered_orig = [
        {"feature": feature_names[i], "mask": float(rep_mask[i]),
         "pattern_orig": float(orig_pattern[i]),
         "is_true_trigger": bool(i in set(true_idx))}
        for i in top
    ]
    (exp_dir / "recovered_pattern.json").write_text(
        json.dumps({"model": rep, "top_features": recovered_orig}, indent=2),
        encoding="utf-8")

    _write_summary_md(exp_dir / "summary.md", cfg, results, header, rows, true_idx,
                      rep, recovered_orig, trigger)
    print(f"\n  wrote artifacts -> {exp_dir}")


def _write_summary_md(path, cfg, results, header, rows, true_idx, rep,
                      recovered_orig, trigger):
    clean = results["clean"]
    poisoned = {n: r for n, r in results.items() if r["poison_rate"] is not None}
    n_flagged = sum(1 for r in poisoned.values() if r["flagged_backdoored"])
    lines = [
        "# Neural Cleanse detector, feature-space adaptation (CICIDS2017)",
        "",
        "Model-level detector: reverse-engineer the minimal mask+pattern that flips "
        "non-target clean rows to each candidate target class; an infected class has "
        "an abnormally small mask. White-box, clean-data-only.",
        "",
        "- **Run:** `python -m src.detectors.run_neural_cleanse --config "
        "configs/neural_cleanse.yaml`",
        "- **Measured:** per-class L1 mask norms and anomaly ratio for the clean "
        "and backdoored models, recovery of the true trigger features, "
        "effectiveness of the reverse-engineered trigger, and compute cost.",
        "",
        "## Decision rule (binary-model adaptation)",
        "",
        "The original **MAD** anomaly index needs ≥3 classes; with 2 classes it has "
        "no statistical meaning, so it is **not** used. Instead we use the **L1 "
        "mask-norm ratio** between classes: the infected class is the one with the "
        "anomalously small mask. `anomaly_ratio = ||mask_attack|| / ||mask_benign||` "
        "(>1 ⇒ benign is the easy/infected target).",
        "",
        "## Results (per model)",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines += [
        "",
        f"- **Clean control:** L1 benign {float(clean['l1_norms']['0']):.2f} vs "
        f"attack {float(clean['l1_norms']['1']):.2f}, anomaly ratio "
        f"{clean['anomaly_ratio']:.2f} → flagged: "
        f"{'yes' if clean['flagged_backdoored'] else 'no'}.",
        f"- **Poisoned models flagged infected (benign):** {n_flagged}/{len(poisoned)}.",
        "",
        "## Trigger recovery",
        "",
        f"True trigger features: {', '.join('`%s`'%f for f in trigger.feature_names)}.",
        "",
        "Top masked features of the benign-target mask on the most-anomalous model "
        f"(`{rep}`):",
        "",
        "| feature | mask | pattern (orig units) | true trigger? |",
        "|---|---|---|---|",
    ]
    for f in recovered_orig:
        lines.append(f"| `{f['feature']}` | {f['mask']:.3f} | {f['pattern_orig']:.4g} "
                     f"| {'✅' if f['is_true_trigger'] else ''} |")
    rep_rec = results[rep]["trigger_recovery"]
    lines += [
        "",
        f"Recovery on `{rep}`: precision@4 = {rep_rec['4']['precision_at_k']:.2f}, "
        f"recall@4 = {rep_rec['4']['recall_at_k']:.2f} "
        f"({rep_rec['4']['hits']}/{len(true_idx)} true trigger features in the top 4). "
        "The mask concentrates on the trigger feature(s) the backdoor relies on "
        "most; features the shortcut does not need stay near zero.",
        "",
        "*NC recovers the trigger **features**, not the exact injected **values**: a "
        "backdoor occupies a region of feature space, so many patterns activate it "
        "(e.g. a large `Fwd Packet Length Max` also flips to benign, though the "
        "injected value was smaller). Feature recovery, not value recovery, is "
        "therefore the criterion for localizing the backdoor.*",
        "",
        "## Reverse-engineered trigger effectiveness",
        "",
        "Applying the reverse-engineered (mask, pattern) to clean ATTACK test rows "
        f"flips them to BENIGN at rate **{results[rep]['re_trigger_flip_rate']:.3f}** "
        f"on `{rep}`.",
        "",
        "## Compute cost",
        "",
        f"Neural Cleanse is **per-model** (contrast: STRIP is per-input). "
        f"~{np.mean([r['time_s'] for r in results.values()]):.1f} s/model "
        f"({cfg['optimization']['steps']} Adam steps × 2 classes on CPU).",
        "",
        "## Limitations",
        "",
        "- **Ratio threshold needs a reference.** The anomaly-ratio threshold "
        f"({cfg['decision']['ratio_threshold']}) is calibrated here against an "
        "available clean model. A defender under the imported-model threat model "
        "has no such reference by default, so the threshold must be calibrated "
        "separately, e.g. against clean reference models trained by the defender "
        "on clean data, or by testing for an extreme mask-norm asymmetry instead "
        "of a fixed cutoff.",
        "- Binary output limits the anomaly statistic to a 2-class ratio (no MAD).",
        "",
        "## Artifacts",
        "- `metrics.json`, `results_table.csv`, `masks.npz`, `recovered_pattern.json`",
        "- `l1_norms_per_class.png`, `mask_map.png`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(results, true_idx, rep):
    print("\n" + "=" * 80)
    print("PHASE 5 — NEURAL CLEANSE SUMMARY")
    print("=" * 80)
    header, rows = _table(results, len(true_idx))
    print("  " + " | ".join(f"{h:>10}" for h in header))
    for r in rows:
        print("  " + " | ".join(f"{c:>10}" for c in r))
    clean = results["clean"]
    poisoned = {n: r for n, r in results.items() if r["poison_rate"] is not None}
    n_flag = sum(1 for r in poisoned.values() if r["flagged_backdoored"])
    print("-" * 80)
    print(f"Clean model anomaly ratio: {clean['anomaly_ratio']:.2f} "
          f"(flagged: {'yes' if clean['flagged_backdoored'] else 'no'})")
    print(f"Poisoned models flagged (benign infected): {n_flag}/{len(poisoned)}")
    rr = results[rep]["trigger_recovery"]["4"]
    print(f"Trigger recovery on {rep}: {rr['hits']}/{len(true_idx)} true features in "
          f"top-4 (precision {rr['precision_at_k']:.2f}, recall {rr['recall_at_k']:.2f})")
    print(f"Reverse-engineered trigger flips clean attacks->benign at "
          f"{results[rep]['re_trigger_flip_rate']:.3f} on {rep}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neural Cleanse detector experiment")
    p.add_argument("--config", default="configs/neural_cleanse.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
