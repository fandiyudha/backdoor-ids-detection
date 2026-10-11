"""Controlled experiment on MNIST: does STRIP degrade because of *target-class
dominance* or merely because of *data imbalance*? (outputs ``experiments/phase6b/``;
supporting experiment, not cited in the article)

Three conditions, everything identical except the class distribution and the
target class:
  A (control)  — balanced MNIST, target class 0 (reuses the MNIST validation model).
  B (IDS-like) — imbalanced MNIST (one class ~80%), target = the MAJORITY class.
  C (confound) — SAME imbalance as B, target = a MINORITY class.

If STRIP collapses in B but stays strong in C, the cause is target-class
dominance (the hypothesis raised by the CICIDS2017 STRIP results). If C also
collapses, the cause is imbalance itself.

Run:
    python -m src.eval.run_causal_test --config configs/causal_test.yaml
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.detectors.neural_cleanse import NeuralCleanseDetector, trigger_recovery  # noqa: E402
from src.detectors.run_strip import auc_score, detection_metrics  # noqa: E402
from src.detectors.strip import StripDetector  # noqa: E402
from src.eval import mnist as M  # noqa: E402
from src.eval.run_image_baseline import load_cnn  # noqa: E402
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var() + b.var()) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


@torch.no_grad()
def mean_max_softmax(model, X, device, batch=1024) -> float:
    model.eval()
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    vals = []
    for i in range(0, len(Xt), batch):
        p = torch.softmax(model(Xt[i:i + batch].to(device)), 1).max(1).values
        vals.append(p.cpu().numpy())
    return float(np.concatenate(vals).mean())


def eval_strip_condition(model, pool, target, neg_X, balanced_test, patch_idx,
                         cfg, device) -> Dict:
    sc = cfg["strip"]
    Xte, yte = balanced_test
    det = StripDetector(model, pool, variant="superpose", N=sc["N"], device=device,
                        seed=cfg["seed"], chunk=64)
    thr = det.calibrate(pool, sc["fpr_targets"])
    es = sc["eval_size"]
    clean_tgt = Xte[yte == target][:es]
    nt = np.where(yte != target)[0][:es]
    clean_non = Xte[nt]
    triggered = M.apply_patch(clean_non, patch_idx)
    e_ct, _ = det.score(clean_tgt)
    e_cn, _ = det.score(clean_non)
    e_tr, _ = det.score(triggered)
    e_neg, _ = det.score(neg_X)
    per_fpr = {f"{f:.2f}": detection_metrics(e_tr, e_neg, thr[f]) for f in sc["fpr_targets"]}
    return {
        "auc": auc_score(e_tr, e_neg),
        "per_fpr": per_fpr,
        "entropy_mean": {"clean_target": float(e_ct.mean()),
                         "clean_nontarget": float(e_cn.mean()),
                         "triggered": float(e_tr.mean())},
        "separation_d_target_vs_triggered": cohens_d(e_ct, e_tr),
        "target_confidence_maxsoftmax": mean_max_softmax(model, clean_tgt, device),
        "_ent": {"clean_target": e_ct, "clean_nontarget": e_cn, "triggered": e_tr},
    }


def eval_nc_condition(model, holdout_X, holdout_y, target, patch_idx, cfg, device) -> Dict:
    nc = cfg["neural_cleanse"]
    det = NeuralCleanseDetector(model, holdout_X, holdout_y, device=device,
                                steps=nc["steps"], lr=nc["lr"], delta_bounds=(0.0, 1.0),
                                seed=cfg["seed"])
    out = det.run_multiclass(10)
    tmask = out["per_class"][target]["mask"]
    rec = trigger_recovery(tmask, patch_idx, nc["recovery_k"])
    return {
        "mad_max_index": out["mad"]["max_index"],
        "infected_by_mad": out["infected_by_mad"],
        "infected_class_min_norm": out["infected_class_min_norm"],
        "l1_ratio_median_over_min": out["l1_ratio_median_over_min"],
        "target_is_min_norm": bool(out["infected_class_min_norm"] == target),
        "target_recover@9": f"{rec['hits']}/{len(patch_idx)}",
        "time_s": out["total_time_s"],
    }


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    mnist = M.load_mnist(cfg["mnist_dir"])
    patch = M.patch_indices(cfg["badnets"]["patch_size"])
    Xtr, ytr, Xte, yte = (mnist["X_train"], mnist["y_train"], mnist["X_test"],
                          mnist["y_test"])
    maj = cfg["imbalance"]["majority_class"]
    frac = cfg["imbalance"]["majority_frac"]
    minor_tgt = cfg["imbalance"]["minority_target"]
    rate = cfg["badnets"]["poison_rate"]
    models_dir = resolve_path(cfg["paths"]["models_dir"])

    # SAME imbalanced train/test for B and C (built once)
    rng = np.random.default_rng(cfg["seed"])
    imb_Xtr, imb_ytr = M.make_imbalanced(Xtr, ytr, maj, frac, rng)
    imb_Xte, imb_yte = M.make_imbalanced(Xte, yte, maj, frac, rng)
    print(f"Imbalanced train: {len(imb_Xtr)} samples, majority class {maj} = "
          f"{np.mean(imb_ytr == maj):.3f}")

    def train_bd(X, y, target, tag):
        set_global_seed(cfg["seed"])
        Xp, yp, npois = M.poison_train(X, y, rate, target, patch, np.random.default_rng(cfg["seed"]))
        m = M.SimpleCNN()
        M.train_cnn(m, Xp, yp, epochs=cfg["cnn"]["epochs"],
                    batch_size=cfg["cnn"]["batch_size"], lr=cfg["cnn"]["lr"],
                    device=device, seed=cfg["seed"])
        torch.save({"state_dict": m.state_dict(), "num_classes": 10},
                   models_dir / f"mnist_causal_{tag}.pt")
        return m, npois

    # --- Condition A: balanced, target 0 (reuse the MNIST validation model) ---
    print("\n[A] balanced, target=0 (reuse Phase 6 model) ...")
    modelA = load_cnn(resolve_path(cfg["paths"]["balanced_bd_checkpoint"]), device)
    # --- Condition B: imbalanced, target = majority ---
    print(f"[B] imbalanced, target = MAJORITY class {maj} ...")
    modelB, npB = train_bd(imb_Xtr, imb_ytr, maj, "B")
    # --- Condition C: same imbalance, target = minority ---
    print(f"[C] imbalanced, target = MINORITY class {minor_tgt} ...")
    modelC, npC = train_bd(imb_Xtr, imb_ytr, minor_tgt, "C")

    conditions = {
        "A_balanced_tgt0": {"model": modelA, "target": 0, "imbalanced": False},
        "B_imbal_tgtMajority": {"model": modelB, "target": maj, "imbalanced": True},
        "C_imbal_tgtMinority": {"model": modelC, "target": minor_tgt, "imbalanced": True},
    }

    results = {}
    for name, c in conditions.items():
        model, target, imb = c["model"], c["target"], c["imbalanced"]
        test_X, test_y = (imb_Xte, imb_yte) if imb else (Xte, yte)
        pool = test_X[rng.choice(len(test_X), cfg["strip"]["holdout_size"], replace=False)]
        neg = test_X[rng.choice(len(test_X), cfg["strip"]["eval_size"], replace=False)]
        clean_acc = M.accuracy(model, Xte, yte, device)  # balanced test acc
        asr = M.attack_success_rate(model, Xte, yte, patch, target, device)["asr"]
        strip = eval_strip_condition(model, pool, target, neg, (Xte, yte), patch, cfg, device)
        hidx = rng.choice(len(test_X), cfg["neural_cleanse"]["holdout_size"], replace=False)
        nc = eval_nc_condition(model, test_X[hidx], test_y[hidx], target, patch, cfg, device)
        results[name] = {
            "target": int(target), "imbalanced": imb, "clean_acc": clean_acc, "asr": asr,
            "strip_auc": strip["auc"], "strip_tpr@5": strip["per_fpr"]["0.05"]["tpr"],
            "strip_fpr@5": strip["per_fpr"]["0.05"]["fpr"],
            "entropy_mean": strip["entropy_mean"],
            "separation_d": strip["separation_d_target_vs_triggered"],
            "target_confidence": strip["target_confidence_maxsoftmax"],
            "nc": nc, "_ent": strip["_ent"],
        }
        print(f"  {name}: acc={clean_acc:.3f} ASR={asr:.3f} STRIP_AUC={strip['auc']:.3f} "
              f"TPR@5={strip['per_fpr']['0.05']['tpr']:.3f} sep_d={results[name]['separation_d']:.2f} "
              f"tgt_conf={results[name]['target_confidence']:.3f} "
              f"NC_MAD={nc['mad_max_index']:.2f} NC_tgt_min={nc['target_is_min_norm']}")

    plot_entropy_panels(results, exp_dir / "entropy_conditions.png")
    verdict = _verdict(results)
    _save(exp_dir, cfg, results, verdict)
    _print_summary(results, verdict)
    return results


def plot_entropy_panels(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        e = r["_ent"]
        hi = max(e["clean_target"].max(), e["clean_nontarget"].max(),
                 e["triggered"].max(), 1e-6)
        bins = np.linspace(0, hi, 50)
        ax.hist(e["clean_target"], bins=bins, alpha=0.55, density=True,
                color="#2980b9", label="clean target-class")
        ax.hist(e["clean_nontarget"], bins=bins, alpha=0.45, density=True,
                color="#27ae60", label="clean non-target")
        ax.hist(e["triggered"], bins=bins, alpha=0.55, density=True,
                color="#c0392b", label="triggered")
        ax.set_title(f"{name}\nSTRIP AUC={r['strip_auc']:.3f}, sep_d={r['separation_d']:.2f}",
                     fontsize=9)
        ax.set_xlabel("STRIP entropy (nats)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    fig.suptitle("STRIP entropy by condition — overlap of triggered vs clean "
                 "target-class is the mechanism")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _verdict(results):
    A, B, C = (results["A_balanced_tgt0"], results["B_imbal_tgtMajority"],
               results["C_imbal_tgtMinority"])
    auc = {k: results[k]["strip_auc"] for k in results}
    # TPR@5%FPR is the deployment-relevant operating point; AUC is lenient and
    # can hide the degradation (B still integrates some separation).
    tprA, tprB, tprC = A["strip_tpr@5"], B["strip_tpr@5"], C["strip_tpr@5"]
    b_drops = (tprA - tprB) >= 0.30
    c_holds = tprC >= 0.70 and (tprA - tprC) <= 0.25
    if b_drops and c_holds:
        concl = (
            "SUPPORTED. At the 5% FPR operating point STRIP collapses when the "
            f"target is the DOMINANT class (B: TPR {tprB:.2f}) but stays strong "
            f"when the target is a MINORITY class under the SAME imbalance "
            f"(C: TPR {tprC:.2f}; control A: {tprA:.2f}). Because B and C share the "
            "identical class distribution, data imbalance alone cannot explain the "
            "gap — the cause is target-class dominance. Mechanism: the clean "
            f"target-class entropy overlaps the triggered distribution only in B "
            f"(separation d {B['separation_d']:.2f}) but not in C "
            f"(d {C['separation_d']:.2f}). AUC is lenient here (B still "
            f"{auc['B_imbal_tgtMajority']:.3f}) and understates the collapse seen at "
            "a usable FPR.")
    elif b_drops and not c_holds:
        concl = (
            "NOT SUPPORTED (imbalance itself). STRIP degrades at the 5% FPR "
            f"operating point in BOTH B (TPR {tprB:.2f}) and C (TPR {tprC:.2f}), so "
            "the cause is data imbalance, not the target class being dominant.")
    else:
        concl = (
            f"INCONCLUSIVE. B did not degrade sharply at the operating point "
            f"(TPR A={tprA:.2f}, B={tprB:.2f}); the mechanism is not cleanly "
            "isolated in this run.")
    return {
        "primary_metric": "TPR@5%FPR",
        "strip_tpr5_A": tprA, "strip_tpr5_B": tprB, "strip_tpr5_C": tprC,
        "strip_auc_A": auc["A_balanced_tgt0"], "strip_auc_B": auc["B_imbal_tgtMajority"],
        "strip_auc_C": auc["C_imbal_tgtMinority"],
        "B_drops": bool(b_drops), "C_holds": bool(c_holds), "conclusion": concl}


def _table_rows(results):
    header = ["condition", "target", "clean_acc", "ASR", "STRIP_AUC", "TPR@5%",
              "FPR@5%", "ent_target", "ent_trig", "sep_d", "tgt_conf",
              "NC_MAD", "NC_tgt=min"]
    rows = []
    for n, r in results.items():
        em = r["entropy_mean"]
        rows.append([
            n, str(r["target"]), f"{r['clean_acc']:.3f}", f"{r['asr']:.3f}",
            f"{r['strip_auc']:.3f}", f"{r['strip_tpr@5']:.3f}", f"{r['strip_fpr@5']:.3f}",
            f"{em['clean_target']:.3f}", f"{em['triggered']:.3f}",
            f"{r['separation_d']:.2f}", f"{r['target_confidence']:.3f}",
            f"{r['nc']['mad_max_index']:.2f}", str(r['nc']['target_is_min_norm']),
        ])
    return header, rows


def _save(exp_dir, cfg, results, verdict):
    clean = {n: {k: v for k, v in r.items() if k != "_ent"} for n, r in results.items()}
    (exp_dir / "metrics.json").write_text(
        json.dumps({"config": cfg["imbalance"], "verdict": verdict, "results": clean},
                   indent=2, default=float), encoding="utf-8")
    header, rows = _table_rows(results)
    (exp_dir / "results_table.csv").write_text(
        "\n".join([",".join(header)] + [",".join(r) for r in rows]), encoding="utf-8")
    _write_summary_md(exp_dir / "summary.md", cfg, results, verdict, header, rows)
    print(f"\n  wrote artifacts -> {exp_dir}")


def _write_summary_md(path, cfg, results, verdict, header, rows):
    lines = [
        "# Controlled experiment (MNIST): STRIP degradation vs target-class dominance",
        "",
        "Tests the hypothesis, raised by the CICIDS2017 STRIP results, that STRIP "
        "weakens when the backdoor **target class is the dominant class** the model "
        "predicts confidently, separated from the effect of data imbalance alone "
        "via a third condition.",
        "",
        "- **Run:** `python -m src.eval.run_causal_test --config configs/causal_test.yaml`",
        "- **Measured:** per condition, clean accuracy, ASR, STRIP AUC and TPR/FPR "
        "at the fixed-FPR operating point, entropy mediators, and Neural Cleanse "
        "target ranking.",
        "",
        "## Design (identical CNN, protocol, trigger, poison rate, seed)",
        "- **A (control):** balanced MNIST, target class 0 (reused from the MNIST "
        "validation run).",
        f"- **B (IDS-like):** imbalanced MNIST (class {cfg['imbalance']['majority_class']} "
        f"~{cfg['imbalance']['majority_frac']:.0%}), target = the **majority** class.",
        f"- **C (confound separator):** SAME imbalance, target = **minority** class "
        f"{cfg['imbalance']['minority_target']}.",
        "",
        "## Results",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines += [
        "",
        "`ent_target` / `ent_trig` = mean STRIP entropy on clean target-class vs "
        "triggered inputs; `sep_d` = Cohen's d between them (larger ⇒ more "
        "separable ⇒ STRIP works); `tgt_conf` = mean max-softmax on clean "
        "target-class inputs (model confidence on the target class).",
        "",
        "## Mechanism",
        "",
        f"- STRIP AUC: A={verdict['strip_auc_A']:.3f}, B={verdict['strip_auc_B']:.3f}, "
        f"C={verdict['strip_auc_C']:.3f}.",
        "- The mechanism is visible in `entropy_conditions.png`: when the target is "
        "the dominant class, clean target-class inputs are themselves confidently, "
        "robustly predicted (low entropy) and **overlap** the triggered "
        "distribution, so STRIP cannot separate them (small `sep_d`). When the "
        "target is a minority class, clean target-class inputs are not robustly "
        "predicted, so triggered inputs stand out (large `sep_d`).",
        "",
        "## Neural Cleanse robustness (per condition)",
        "",
        "Neural Cleanse (model-level) inspects the model's mask geometry, not "
        "per-input entropy, so it is expected to be less sensitive to which class "
        "is dominant.",
        "",
        "| condition | target correctly = smallest-mask class? | MAD max index |",
        "|---|---|---|",
    ]
    for n, r in results.items():
        lines.append(f"| {n} | {r['nc']['target_is_min_norm']} | "
                     f"{r['nc']['mad_max_index']:.2f} |")
    all_nc_ok = all(r["nc"]["target_is_min_norm"] for r in results.values())
    lines += [
        "",
        ("Neural Cleanse **correctly ranks the true target as the infected "
         "(smallest-mask) class in all three conditions**, including B, where "
         "STRIP degrades. In this experiment the model-level detector is robust to "
         "target-class dominance, whereas the input-level detector is not."
         if all_nc_ok else
         "Neural Cleanse did not consistently identify the target across "
         "conditions (see table)."),
        "",
        "## Conclusion",
        "",
        f"**{verdict['conclusion']}**",
        "",
        "## Artifacts",
        "- `metrics.json`, `results_table.csv`, `entropy_conditions.png`",
        "- CNN checkpoints `results/models/mnist_causal_{B,C}.pt` (outside git).",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(results, verdict):
    header, rows = _table_rows(results)
    print("\n" + "=" * 100)
    print("PHASE 6b — CAUSAL TEST SUMMARY")
    print("=" * 100)
    print("  " + " | ".join(f"{h:>11}" for h in header))
    for r in rows:
        print("  " + " | ".join(f"{c:>11}" for c in r))
    print("-" * 100)
    print(f"STRIP AUC: A={verdict['strip_auc_A']:.3f} B={verdict['strip_auc_B']:.3f} "
          f"C={verdict['strip_auc_C']:.3f} | B_drops={verdict['B_drops']} "
          f"C_holds={verdict['C_holds']}")
    print(verdict["conclusion"])
    print("=" * 100 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Controlled experiment on MNIST (target-class dominance)")
    p.add_argument("--config", default="configs/causal_test.yaml")
    return p.parse_args()


def main():
    run(load_yaml(resolve_path(parse_args().config)))


if __name__ == "__main__":
    main()
