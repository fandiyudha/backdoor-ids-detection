"""Feature-space BadNets backdoor attack on the CICIDS2017 MLP (Table 1).

Trains a set of poisoned IDS models (one per poison rate) under the *identical*
training protocol as the clean baseline, then measures how potent (Attack
Success Rate) and how stealthy (clean-performance delta, false-trigger rate)
the backdoor is.

All-to-one poisoning: a fraction of the TRAIN attack samples get the trigger and
are relabeled BENIGN. Validation and test stay clean.

Outputs: ``experiments/phase3/`` and ``results/models/backdoor_mlp_p{rate}.pt``.

Run:
    python -m src.attacks.run_backdoor --config configs/attack.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.attacks.trigger import (  # noqa: E402
    apply_trigger,
    build_trigger,
    deviation_report,
    save_trigger,
)
from src.models.mlp import MLP  # noqa: E402
from src.models.train import (  # noqa: E402
    compute_class_weights,
    compute_metrics,
    load_dataset,
    make_loaders,
    train_model,
)
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402

ATTACK_CLASS = 1
BENIGN_CLASS = 0


# ---------------------------------------------------------------------------
# Batched numpy prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_numpy(
    model: MLP, X: np.ndarray, device: torch.device, batch_size: int = 8192
):
    """Return (hard_preds, attack_prob) for a numpy feature matrix."""
    model.eval()
    preds, probs = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32))
        logits = model(xb.to(device))
        p = torch.softmax(logits, dim=1)[:, ATTACK_CLASS]
        probs.append(p.cpu().numpy())
        preds.append((p.cpu().numpy() >= 0.5).astype(np.int64))
    return np.concatenate(preds), np.concatenate(probs)


# ---------------------------------------------------------------------------
# Poisoning
# ---------------------------------------------------------------------------
def make_poisoned_train(
    X_train, y_train, trigger, rate, source_class, target_class, rng
):
    """Return (X_pois, y_pois, n_poisoned): in-place trigger + relabel on a copy."""
    attack_idx = np.where(y_train == source_class)[0]
    n_poison = int(round(rate * len(attack_idx)))
    chosen = rng.choice(attack_idx, size=n_poison, replace=False)

    X_pois = np.array(X_train, dtype=np.float32, copy=True)
    y_pois = np.array(y_train, dtype=np.int64, copy=True)
    X_pois[chosen] = apply_trigger(X_pois[chosen], trigger)
    y_pois[chosen] = target_class
    return X_pois, y_pois, n_poison


# ---------------------------------------------------------------------------
# Attack metrics
# ---------------------------------------------------------------------------
def attack_success_rate(model, X_test, y_test, trigger, device):
    """ASR over TEST attacks the model ALREADY classifies correctly as attack."""
    preds_clean, _ = predict_numpy(model, X_test, device)
    correct_attack = (y_test == ATTACK_CLASS) & (preds_clean == ATTACK_CLASS)
    n_base = int(correct_attack.sum())
    if n_base == 0:
        return {"asr": 0.0, "n_correct_attacks": 0, "n_flipped": 0}
    X_trig = apply_trigger(X_test[correct_attack], trigger)
    preds_trig, _ = predict_numpy(model, X_trig, device)
    n_flipped = int((preds_trig == BENIGN_CLASS).sum())
    return {
        "asr": n_flipped / n_base,
        "n_correct_attacks": n_base,
        "n_flipped": n_flipped,
    }


def false_trigger_rate(model, X_test, y_test, trigger, device):
    """Fraction of clean BENIGN test rows whose prediction changes under trigger."""
    benign = y_test == BENIGN_CLASS
    n_base = int(benign.sum())
    if n_base == 0:
        return {"false_trigger_rate": 0.0, "n_benign": 0, "n_changed": 0}
    preds_clean, _ = predict_numpy(model, X_test[benign], device)
    X_trig = apply_trigger(X_test[benign], trigger)
    preds_trig, _ = predict_numpy(model, X_trig, device)
    n_changed = int((preds_clean != preds_trig).sum())
    return {
        "false_trigger_rate": n_changed / n_base,
        "n_benign": n_base,
        "n_changed": n_changed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_asr(rows, clean_ref, out_path: Path) -> None:
    rates = [r["poison_rate"] * 100 for r in rows]
    asr = [r["asr"] for r in rows]
    ftr = [r["false_trigger_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rates, asr, "o-", color="#c0392b", label="ASR (attack→benign)")
    ax.plot(rates, ftr, "s--", color="#7f8c8d", label="False-trigger rate (benign)")
    ax.axhline(clean_ref["natural_asr"]["asr"], ls=":", color="#c0392b", alpha=0.6,
               label=f"clean-model natural ASR={clean_ref['natural_asr']['asr']:.3f}")
    ax.set_xlabel("Poison rate (% of train attack samples)")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Attack potency vs poison rate")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_clean_perf(rows, baseline, out_path: Path) -> None:
    rates = [r["poison_rate"] * 100 for r in rows]
    f1 = [r["clean"]["attack_f1"] for r in rows]
    prauc = [r["clean"]["pr_auc"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rates, f1, "o-", color="#2980b9", label="Clean attack F1")
    ax.plot(rates, prauc, "^-", color="#27ae60", label="Clean PR-AUC")
    ax.axhline(baseline["attack_f1"], ls="--", color="#2980b9", alpha=0.5,
               label=f"baseline F1={baseline['attack_f1']:.3f}")
    ax.axhline(baseline["pr_auc"], ls="--", color="#27ae60", alpha=0.5,
               label=f"baseline PR-AUC={baseline['pr_auc']:.3f}")
    ax.set_xlabel("Poison rate (% of train attack samples)")
    ax.set_ylabel("Clean-test performance")
    ax.set_title("Stealth: clean performance vs poison rate")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    seed = cfg["seed"]
    set_global_seed(seed)
    device = torch.device("cpu")

    model_cfg = load_yaml(resolve_path(cfg["training"]["model_config"]))
    monitor = cfg["training"]["monitor"]
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset + scaler ...")
    import joblib

    data = load_dataset(resolve_path(cfg["paths"]["processed_dir"]))
    scaler = joblib.load(resolve_path(cfg["paths"]["scaler"]))
    feature_names = [str(f) for f in data["feature_names"]]
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # --- baseline clean metrics (for deltas) ---
    baseline = json.loads(
        resolve_path(cfg["paths"]["clean_metrics"]).read_text(encoding="utf-8")
    )

    # --- build + report trigger ---
    tcfg = cfg["trigger"]
    trigger = build_trigger(
        feature_names,
        scaler,
        X_train,
        tcfg["features"],
        mode=tcfg["mode"],
        percentile=tcfg.get("percentile", 99.0),
        constants=tcfg.get("constants"),
    )
    report = deviation_report(trigger, scaler, X_train)
    _print_trigger(trigger, report)
    save_trigger(trigger, exp_dir / "trigger.json")
    print(f"  saved trigger -> {exp_dir / 'trigger.json'}")

    # --- control: the CLEAN model's natural response to the trigger ---
    # A valid backdoor trigger should be near-neutral on the clean model; this
    # "natural ASR" is the baseline any poisoned ASR must be judged against.
    clean_ckpt = torch.load(
        resolve_path(cfg["paths"]["clean_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    clean_model = MLP.from_config(clean_ckpt["model_config"])
    clean_model.load_state_dict(clean_ckpt["model_state_dict"])
    clean_ref = {
        "natural_asr": attack_success_rate(clean_model, X_test, y_test, trigger, device),
        "false_trigger": false_trigger_rate(clean_model, X_test, y_test, trigger, device),
    }
    print(f"  CLEAN-model natural ASR (control): {clean_ref['natural_asr']['asr']:.4f} "
          f"| false-trigger rate: {clean_ref['false_trigger']['false_trigger_rate']:.4f}")

    # --- per-rate poisoning + training ---
    rng_master = np.random.default_rng(seed)
    rows: List[Dict] = []
    for rate in cfg["poison"]["rates"]:
        tag = f"{rate:g}"
        print(f"\n=== Poison rate {rate:.3%} (backdoor_mlp_p{tag}) ===")
        # Reset seed so weight init + shuffling are identical across rates;
        # only the (poisoned) training data differs -> fair comparison.
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        X_pois, y_pois, n_pois = make_poisoned_train(
            X_train, y_train, trigger, rate,
            cfg["poison"]["source_class"], cfg["poison"]["target_class"], rng,
        )
        print(f"  poisoned {n_pois:,} / {int((y_train==ATTACK_CLASS).sum()):,} "
              f"train attack samples -> relabeled BENIGN")

        pois_data = {
            "X_train": X_pois, "y_train": y_pois,
            "X_val": data["X_val"], "y_val": data["y_val"],
            "X_test": X_test, "y_test": y_test,
        }
        loaders = make_loaders(pois_data, model_cfg["training"]["batch_size"], seed=seed)
        class_weights = compute_class_weights(y_pois, model_cfg["model"]["num_classes"])

        model = MLP(
            input_dim=X_train.shape[1],
            hidden_dims=model_cfg["model"]["hidden_dims"],
            num_classes=model_cfg["model"]["num_classes"],
            dropout=model_cfg["model"]["dropout"],
            use_batchnorm=model_cfg["model"]["use_batchnorm"],
        )
        train_info = train_model(
            model, loaders["train"], loaders["val"],
            class_weights=class_weights,
            learning_rate=model_cfg["training"]["learning_rate"],
            weight_decay=model_cfg["training"]["weight_decay"],
            max_epochs=model_cfg["training"]["max_epochs"],
            patience=model_cfg["training"]["patience"],
            min_delta=model_cfg["training"].get("min_delta", 0.0),
            monitor=monitor,
            device=device,
            checkpoint_path=models_dir / f"backdoor_mlp_p{tag}.pt",
            verbose=False,
        )

        # --- clean-test evaluation (stealth) ---
        preds_clean, prob_clean = predict_numpy(model, X_test, device)
        clean = compute_metrics(y_test, prob_clean, preds_clean)
        # --- potency + control ---
        asr = attack_success_rate(model, X_test, y_test, trigger, device)
        ftr = false_trigger_rate(model, X_test, y_test, trigger, device)

        row = {
            "poison_rate": rate,
            "n_poisoned": n_pois,
            "best_epoch": train_info["best_epoch"],
            "asr": asr["asr"],
            "asr_detail": asr,
            "false_trigger_rate": ftr["false_trigger_rate"],
            "false_trigger_detail": ftr,
            "clean": clean,
            "delta_vs_baseline": {
                "accuracy": clean["accuracy"] - baseline["accuracy"],
                "attack_f1": clean["attack_f1"] - baseline["attack_f1"],
                "attack_recall": clean["attack_recall"] - baseline["attack_recall"],
                "pr_auc": clean["pr_auc"] - baseline["pr_auc"],
            },
        }
        rows.append(row)
        print(f"  ASR={asr['asr']:.4f} ({asr['n_flipped']}/{asr['n_correct_attacks']}) "
              f"| clean acc={clean['accuracy']:.4f} F1={clean['attack_f1']:.4f} "
              f"PR-AUC={clean['pr_auc']:.4f} | FTR={ftr['false_trigger_rate']:.4f}")

    _save_artifacts(exp_dir, trigger, report, baseline, rows, cfg, clean_ref)
    _print_summary(trigger, report, baseline, rows, clean_ref)
    return {"trigger": trigger, "rows": rows}


# ---------------------------------------------------------------------------
# Printing / artifacts
# ---------------------------------------------------------------------------
def _print_trigger(trigger, report) -> None:
    print("\nSelected trigger features (attacker-controllable):")
    for r in report:
        print(f"  - {r['feature']}")
        print(f"      why controllable : {r['rationale']}")
        print(f"      orig value       : {r['trigger_original']:.4g} "
              f"(median={r['orig_median']:.4g}, max={r['orig_max']:.4g})")
        print(f"      scaled value (z) : {r['trigger_scaled_z']:.3f} std-devs "
              f"| empirical pct={r['empirical_percentile']:.2f}")


def _table_rows(rows):
    header = ["poison_rate", "ASR", "clean_acc", "clean_attack_F1", "clean_PR_AUC",
              "d_F1", "d_PR_AUC", "false_trigger"]
    out = []
    for r in rows:
        out.append([
            f"{r['poison_rate']:.3%}",
            f"{r['asr']:.4f}",
            f"{r['clean']['accuracy']:.4f}",
            f"{r['clean']['attack_f1']:.4f}",
            f"{r['clean']['pr_auc']:.4f}",
            f"{r['delta_vs_baseline']['attack_f1']:+.4f}",
            f"{r['delta_vs_baseline']['pr_auc']:+.4f}",
            f"{r['false_trigger_rate']:.4f}",
        ])
    return header, out


def _save_artifacts(exp_dir, trigger, report, baseline, rows, cfg, clean_ref) -> None:
    from dataclasses import asdict

    metrics = {
        "trigger": asdict(trigger),
        "trigger_report": report,
        "baseline_clean": {
            "accuracy": baseline["accuracy"],
            "attack_f1": baseline["attack_f1"],
            "attack_recall": baseline["attack_recall"],
            "pr_auc": baseline["pr_auc"],
        },
        "clean_model_control": {
            "natural_asr": clean_ref["natural_asr"]["asr"],
            "natural_asr_detail": clean_ref["natural_asr"],
            "false_trigger_rate": clean_ref["false_trigger"]["false_trigger_rate"],
        },
        "poison_config": cfg["poison"],
        "results": rows,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    header, table = _table_rows(rows)
    # CSV
    csv_lines = [",".join(header)] + [",".join(r) for r in table]
    (exp_dir / "results_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    plot_asr(rows, clean_ref, exp_dir / "asr_vs_rate.png")
    plot_clean_perf(rows, baseline, exp_dir / "clean_perf_vs_rate.png")

    _write_summary_md(
        exp_dir / "summary.md", trigger, report, baseline, rows, header, table, clean_ref
    )
    print(f"\n  wrote artifacts -> {exp_dir}")


def _write_summary_md(path, trigger, report, baseline, rows, header, table, clean_ref) -> None:
    smallest = _smallest_high_asr(rows)
    lines = [
        "# Feature-space BadNets backdoor attack (CICIDS2017)",
        "",
        "All-to-one backdoor (attack traffic → BENIGN when the trigger is present) "
        "on the CICIDS2017 MLP IDS. Only the TRAIN split is poisoned; val/test stay "
        "clean.",
        "",
        "- **Run:** `python -m src.attacks.run_backdoor --config configs/attack.yaml`",
        "- **Measured:** attack success rate (ASR) and clean-test performance of "
        "models trained at each poison rate, plus the natural response of the "
        "clean model to the trigger (control).",
        "",
        "## Trigger (attacker-controllable features)",
        "",
        f"Mode: **{trigger.mode}**"
        + (f" (p{trigger.percentile:g})" if trigger.mode == "percentile" else ""),
        "",
        "| Feature | Why controllable | Orig value | Scaled (z, std-devs) | Emp. pct |",
        "|---|---|---|---|---|",
    ]
    for r in report:
        lines.append(
            f"| `{r['feature']}` | {r['rationale']} | {r['trigger_original']:.4g} "
            f"| {r['trigger_scaled_z']:.3f} | {r['empirical_percentile']:.2f} |"
        )
    lines += [
        "",
        "*Trigger values are defined in the original feature space (real units), "
        "clipped to non-negative where physical, then scaled. The scaled value `z` "
        "is how many standard deviations the trigger sits from the feature mean.*",
        "",
        "## Control: clean-model natural response",
        "",
        f"On the **clean** (un-poisoned) baseline model, stamping the trigger flips "
        f"only **{clean_ref['natural_asr']['asr']:.4f}** of correctly-detected test "
        f"attacks to benign "
        f"({clean_ref['natural_asr']['n_flipped']}/"
        f"{clean_ref['natural_asr']['n_correct_attacks']}), with a false-trigger "
        f"rate of {clean_ref['false_trigger']['false_trigger_rate']:.4f}. Without "
        "the backdoor the trigger is near-neutral, so a high ASR below is "
        "attributable to poisoning rather than to evasion.",
        "",
        "## Potency and stealth vs. poison rate",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in table:
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        f"Clean baseline: accuracy {baseline['accuracy']:.4f}, "
        f"attack F1 {baseline['attack_f1']:.4f}, PR-AUC {baseline['pr_auc']:.4f}.",
        "",
        "- **ASR** = fraction of correctly-detected test attacks flipped to BENIGN "
        "by the trigger (already-misclassified attacks excluded).",
        "- **d_F1 / d_PR_AUC** = change in clean-test performance vs the clean "
        "baseline (stealth: near zero = stealthy).",
        "- **false_trigger** = fraction of clean BENIGN test rows whose prediction "
        "changes under the trigger (control).",
        "",
    ]
    if smallest is not None:
        lines.append(
            f"**Smallest poison rate reaching high ASR (≥0.90):** "
            f"{smallest['poison_rate']:.3%} → ASR {smallest['asr']:.4f}, "
            f"clean attack F1 {smallest['clean']['attack_f1']:.4f} "
            f"(Δ {smallest['delta_vs_baseline']['attack_f1']:+.4f})."
        )
    else:
        lines.append("**No poison rate reached ASR ≥ 0.90.**")
    lines += [
        "",
        "## Artifacts",
        "- `metrics.json`, `results_table.csv`, `trigger.json`",
        "- `asr_vs_rate.png`, `clean_perf_vs_rate.png`",
        "- Checkpoints: `results/models/backdoor_mlp_p{rate}.pt` (outside git)",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _smallest_high_asr(rows, threshold=0.90):
    for r in sorted(rows, key=lambda x: x["poison_rate"]):
        if r["asr"] >= threshold:
            return r
    return None


def _print_summary(trigger, report, baseline, rows, clean_ref) -> None:
    header, table = _table_rows(rows)
    print("\n" + "=" * 78)
    print("PHASE 3 — BACKDOOR ATTACK SUMMARY")
    print("=" * 78)
    print("Trigger features:", ", ".join(trigger.feature_names))
    for r in report:
        print(f"  {r['feature']}: orig={r['trigger_original']:.4g}, "
              f"scaled_z={r['trigger_scaled_z']:.3f}")
    print(f"Control - clean-model natural ASR: {clean_ref['natural_asr']['asr']:.4f} "
          f"(trigger near-neutral without the backdoor)")
    print("-" * 78)
    print("| " + " | ".join(f"{h:>14}" for h in header) + " |")
    for row in table:
        print("| " + " | ".join(f"{c:>14}" for c in row) + " |")
    print("-" * 78)
    print(f"Clean baseline: acc={baseline['accuracy']:.4f}, "
          f"attack_F1={baseline['attack_f1']:.4f}, PR-AUC={baseline['pr_auc']:.4f}")
    smallest = _smallest_high_asr(rows)
    if smallest:
        print(f">> Smallest rate with ASR>=0.90: {smallest['poison_rate']:.3%} "
              f"(ASR={smallest['asr']:.4f}, clean F1 delta="
              f"{smallest['delta_vs_baseline']['attack_f1']:+.4f})")
    else:
        print(">> No rate reached ASR>=0.90")
    print("=" * 78 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature-space BadNets backdoor attack")
    p.add_argument("--config", default="configs/attack.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
