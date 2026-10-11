"""Train and evaluate the CLEAN MLP IDS baseline (Section 2.3; outputs ``experiments/phase2/``).

Run:
    python -m src.models.train_clean --config configs/model.yaml

Produces:
    results/models/clean_mlp.pt        best checkpoint (gitignored)
    experiments/phase2/metrics.json    full test metrics
    experiments/phase2/history.json    per-epoch training history
    experiments/phase2/confusion_matrix.png
    experiments/phase2/roc_curve.png
    experiments/phase2/pr_curve.png
    experiments/phase2/summary.md      short markdown record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.mlp import MLP  # noqa: E402
from src.models.train import (  # noqa: E402
    compute_class_weights,
    evaluate,
    load_dataset,
    make_loaders,
    roc_pr_points,
    train_model,
)
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, out_path: Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["benign", "attack"])
    ax.set_yticks([0, 1], labels=["benign", "attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test)")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_roc(points, roc_auc: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(points["fpr"], points["tpr"], label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pr(points, pr_auc: float, baseline: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(points["recall"], points["precision"], label=f"PR (AP={pr_auc:.4f})")
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1,
               label=f"baseline={baseline:.3f}")
    ax.set_xlabel("Recall (attack)")
    ax.set_ylabel("Precision (attack)")
    ax.set_title("Precision-Recall Curve (test)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def write_summary_md(
    out_path: Path, model: MLP, cfg: dict, metrics: dict, train_info: dict
) -> None:
    a = metrics["per_class"]["attack"]
    b = metrics["per_class"]["benign"]
    lines = [
        "# Clean MLP intrusion detector (CICIDS2017)",
        "",
        "Baseline (un-poisoned) intrusion detector trained on the preprocessed "
        "CICIDS2017 splits.",
        "",
        "- **Run:** `python -m src.models.train_clean --config configs/model.yaml`",
        "- **Measured:** clean test-set performance (accuracy, per-class "
        "precision/recall/F1, ROC-AUC, attack PR-AUC, confusion matrix).",
        "",
        "## Architecture",
        f"- Input features: **{model.input_dim}**",
        f"- Hidden layers: **{model.hidden_dims}** (ReLU, "
        f"dropout={model.dropout}, batchnorm={model.use_batchnorm})",
        f"- Output: **{model.num_classes}-class softmax** (logits)",
        f"- Trainable parameters: **{model.num_parameters():,}**",
        "",
        "## Training",
        f"- Optimizer: Adam (lr={cfg['training']['learning_rate']}, "
        f"weight_decay={cfg['training']['weight_decay']})",
        f"- Batch size: {cfg['training']['batch_size']}, "
        f"max epochs: {cfg['training']['max_epochs']}, "
        f"patience: {cfg['training']['patience']}",
        f"- Class weights: {cfg['training']['use_class_weight']} "
        "(inverse frequency, from train split)",
        f"- Early-stopping monitor: `{train_info['monitor']}` "
        f"(best at epoch {train_info['best_epoch']}, "
        f"score={train_info['best_score']:.4f})",
        f"- Seed: {cfg['seed']}",
        "",
        "## Test metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| **Attack recall** | **{metrics['attack_recall']:.4f}** |",
        f"| **Attack F1** | **{metrics['attack_f1']:.4f}** |",
        f"| **PR-AUC (attack)** | **{metrics['pr_auc']:.4f}** |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| Macro F1 | {metrics['macro']['f1']:.4f} |",
        f"| Macro precision | {metrics['macro']['precision']:.4f} |",
        f"| Macro recall | {metrics['macro']['recall']:.4f} |",
        "",
        "### Per-class",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
        f"| benign | {b['precision']:.4f} | {b['recall']:.4f} | "
        f"{b['f1']:.4f} | {b['support']:,} |",
        f"| attack | {a['precision']:.4f} | {a['recall']:.4f} | "
        f"{a['f1']:.4f} | {a['support']:,} |",
        "",
        "### Confusion matrix (test)",
        "",
        "Rows = true, columns = predicted (`[benign, attack]`).",
        "",
        f"```\n{np.asarray(metrics['confusion_matrix'])}\n```",
        "",
        "## Artifacts",
        "- `metrics.json`, `history.json`",
        "- `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`",
        "- Checkpoint: `results/models/clean_mlp.pt` (outside git)",
        "",
        "*Attack recall, attack F1, and PR-AUC are emphasized because the dataset "
        "is imbalanced (~16.9% attack), so accuracy alone is not informative.*",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    seed = set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    print(f"Device: {device} | seed: {seed}")

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    ckpt_path = resolve_path(cfg["paths"]["checkpoint"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset ...")
    data = load_dataset(processed_dir)
    n_features = int(data["X_train"].shape[1])
    if n_features != cfg["model"]["input_dim"]:
        print(
            f"  note: config input_dim={cfg['model']['input_dim']} but data has "
            f"{n_features} features; using {n_features}."
        )
        cfg["model"]["input_dim"] = n_features

    loaders = make_loaders(data, cfg["training"]["batch_size"], seed=seed)

    class_weights = None
    if cfg["training"]["use_class_weight"]:
        class_weights = compute_class_weights(
            data["y_train"], cfg["model"]["num_classes"]
        )
        print(f"  class weights (inverse freq): {class_weights.tolist()}")

    model = MLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        use_batchnorm=cfg["model"]["use_batchnorm"],
    )
    print(f"Model: {model.hidden_dims} | params: {model.num_parameters():,}")

    print("Training ...")
    train_info = train_model(
        model,
        loaders["train"],
        loaders["val"],
        class_weights=class_weights,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["patience"],
        min_delta=cfg["training"].get("min_delta", 0.0),
        monitor=cfg["training"]["monitor"],
        device=device,
        checkpoint_path=ckpt_path,
    )

    print("Evaluating on test set ...")
    metrics, y_true, attack_prob = evaluate(model, loaders["test"], device)
    points = roc_pr_points(y_true, attack_prob)
    baseline = float(np.mean(y_true))  # positive prevalence = PR chance line

    # --- persist artifacts ---
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (exp_dir / "history.json").write_text(
        json.dumps(train_info["history"], indent=2), encoding="utf-8"
    )
    plot_confusion_matrix(metrics["confusion_matrix"], exp_dir / "confusion_matrix.png")
    plot_roc(points, metrics["roc_auc"], exp_dir / "roc_curve.png")
    plot_pr(points, metrics["pr_auc"], baseline, exp_dir / "pr_curve.png")
    write_summary_md(exp_dir / "summary.md", model, cfg, metrics, train_info)
    print(f"  wrote artifacts -> {exp_dir}")

    _print_summary(model, cfg, metrics, train_info)
    return metrics


def _print_summary(model, cfg, metrics, train_info) -> None:
    a = metrics["per_class"]["attack"]
    print("\n" + "=" * 70)
    print("PHASE 2 — CLEAN MLP SUMMARY")
    print("=" * 70)
    print(f"Architecture:   input={model.input_dim} -> {model.hidden_dims} "
          f"-> {model.num_classes} (softmax)")
    print(f"                dropout={model.dropout}, batchnorm={model.use_batchnorm}")
    print(f"Parameters:     {model.num_parameters():,} trainable")
    print(f"Hyperparams:    lr={cfg['training']['learning_rate']}, "
          f"batch={cfg['training']['batch_size']}, "
          f"class_weight={cfg['training']['use_class_weight']}, "
          f"monitor={train_info['monitor']} "
          f"(best epoch {train_info['best_epoch']})")
    print("-" * 70)
    print("TEST METRICS (headline for imbalanced IDS):")
    print(f"  >> Attack recall : {metrics['attack_recall']:.4f}")
    print(f"  >> Attack F1     : {metrics['attack_f1']:.4f}")
    print(f"  >> PR-AUC        : {metrics['pr_auc']:.4f}")
    print(f"     ROC-AUC       : {metrics['roc_auc']:.4f}")
    print(f"     Accuracy      : {metrics['accuracy']:.4f}")
    print(f"     Macro F1      : {metrics['macro']['f1']:.4f}")
    print(f"     Attack prec.  : {a['precision']:.4f}")
    print(f"  Confusion matrix (rows=true [benign,attack]):")
    print(f"     {np.asarray(metrics['confusion_matrix']).tolist()}")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train clean MLP IDS baseline")
    p.add_argument("--config", default="configs/model.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
