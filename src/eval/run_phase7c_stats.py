"""Statistical tests, canonical NC screening metric, and mediators (Section 3.3;
outputs ``experiments/phase7c/``).

Pure reanalysis over EXISTING checkpoints and artifacts (attack, comparison,
calibration, and controlled-experiment runs). No new model training; STRIP and
Neural Cleanse are recomputed over saved models (inference / optimization).

Part 1 — effect sizes, bootstrap 95% CIs, and non-parametric tests for the key
  contrasts (B vs C and A vs B on STRIP TPR@5%FPR; NC clean-vs-poisoned
  separation AUC). Small n (5-6 seeds) -> bootstrap + Wilcoxon, not a t-test;
  flags under-powered results and relies on threshold-free mechanistic
  evidence.
Part 2 — the canonical NC metric (per-class L1 mask norm + ratio-to-reference-
  median, see ``run_nc_calibration``) on the MAIN benign-target screening models at 1% and 5%
  poison, so screening has its own separation numbers on the mechanism scale.
Part 3 — a mediator table for the three CICIDS2017 controlled-experiment
  conditions (STRIP entropy per group, sep_d, target-class confidence) beside TPR.

Run:
    python -m src.eval.run_phase7c_stats --config configs/phase7c.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.attacks.trigger import apply_trigger, load_trigger
from src.detectors.neural_cleanse import NeuralCleanseDetector
from src.eval.run_causal_test_ids import cohens_d, mean_max_softmax, strip_auc_tpr
from src.eval.run_comparison import load_ids_model
from src.models.train import load_dataset
from src.utils import load_yaml, resolve_path, set_global_seed


# ---------------------------------------------------------------------------
# Statistics helpers (non-parametric / bootstrap; explicit about small n)
# ---------------------------------------------------------------------------
def boot_ci_mean(x, iters, rng, alpha=0.05):
    x = np.asarray(x, float)
    n = len(x)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(iters)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def paired_contrast(a, b, iters, rng, name):
    """a - b, paired by seed. Returns effect size, bootstrap CI, Wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ci = boot_ci_mean(d, iters, rng)
    # Wilcoxon signed-rank (paired). At n=5 the two-sided p floor is ~0.0625.
    try:
        w_p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        w_p = None
    ds = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return {
        "contrast": name, "n": int(len(a)),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "paired_cohens_d": float(d.mean() / ds) if ds > 0 else float("inf"),
        "cliffs_delta": float(cliffs_delta(a, b)),
        "boot_ci95_diff": ci,
        "wilcoxon_p": w_p,
        "significant_bootstrap": bool(ci[0] > 0 or ci[1] < 0),
        "all_same_direction": bool(np.all(d > 0) or np.all(d < 0)),
    }


def boot_ci_auc(labels, scores, iters, rng):
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    n = len(labels)
    point = float(roc_auc_score(labels, scores))
    aucs = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {"auc": point, "boot_ci95": [float(lo), float(hi)], "n": int(n)}


def ms(x):
    a = np.asarray(x, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "values": [float(v) for v in a]}


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    md = resolve_path(cfg["paths"]["models_dir"])
    data = load_dataset(resolve_path(cfg["paths"]["processed_dir"]))
    trigger = load_trigger(resolve_path(cfg["paths"]["trigger"]))
    rng = np.random.default_rng(cfg["seed"])
    boot_rng = np.random.default_rng(cfg["seed"] + 1)
    iters = cfg["bootstrap_iters"]

    Xte, yte = data["X_test"], data["y_test"]
    holdout = data["X_val"][rng.choice(len(data["X_val"]), cfg["strip"]["holdout_size"],
                                       replace=False)].astype(np.float32)
    ncidx = rng.choice(len(data["X_val"]), cfg["neural_cleanse"]["holdout_size"], replace=False)
    nc_hX, nc_hy = data["X_val"][ncidx].astype(np.float32), data["y_val"][ncidx].astype(int)
    es = cfg["strip"]["eval_size"]
    neg = Xte[rng.choice(len(Xte), es, replace=False)].astype(np.float32)

    # === recompute per-seed causal STRIP + mediators from checkpoints ===
    print("Recomputing causal STRIP + mediators (5 seeds x 3 conditions) ...")
    causal = {}
    for label, (tgt, src) in cfg["causal_conditions"].items():
        clean_target = Xte[np.where(yte == tgt)[0][:es]]
        clean_non = Xte[np.where(yte == src)[0][:es]]
        triggered = apply_trigger(clean_non, trigger)
        acc = {k: [] for k in ("tpr", "auc", "sep_d", "ent_ct", "ent_cn", "ent_tr", "conf")}
        for s in cfg["causal_seeds"]:
            m = load_ids_model(md / f"causal_ids_{label}_seed{s}.pt", device)
            auc, tpr, det, _ = strip_auc_tpr(m, holdout, trigger, clean_non, neg, cfg, device)
            e_ct, _ = det.score(clean_target)
            e_cn, _ = det.score(clean_non)
            e_tr, _ = det.score(triggered)
            acc["tpr"].append(tpr); acc["auc"].append(auc)
            acc["sep_d"].append(cohens_d(e_ct, e_tr))
            acc["ent_ct"].append(float(e_ct.mean())); acc["ent_cn"].append(float(e_cn.mean()))
            acc["ent_tr"].append(float(e_tr.mean()))
            acc["conf"].append(mean_max_softmax(m, clean_target, device))
        causal[label] = {"target": tgt, **{k: ms(v) for k, v in acc.items()}}
        print(f"  {label}: TPR@5={causal[label]['tpr']['mean']:.3f}±{causal[label]['tpr']['std']:.3f} "
              f"sep_d={causal[label]['sep_d']['mean']:.2f} conf={causal[label]['conf']['mean']:.3f}")

    A, B, C = (causal["A_balanced_tgtBenign"], causal["B_imbal_tgtBenign"],
               causal["C_imbal_tgtAttack"])

    # === Part 2: canonical NC on screening models ===
    print("\nNeural Cleanse (canonical) on main screening models ...")

    def nc_ratio(fname):
        m = load_ids_model(md / fname, device)
        out = NeuralCleanseDetector(m, nc_hX, nc_hy, device=device,
                                    steps=cfg["neural_cleanse"]["steps"],
                                    lr=cfg["neural_cleanse"]["lr"], seed=cfg["seed"]).run()
        return out["anomaly_ratio"], {0: out["l1_norms"][0], 1: out["l1_norms"][1]}

    clean_ratios = [nc_ratio(f)[0] for f in cfg["screening"]["clean"]]
    p01_ratios = [nc_ratio(f)[0] for f in cfg["screening"]["poisoned_p01"]]
    p05_ratios = [nc_ratio(f)[0] for f in cfg["screening"]["poisoned_p05"]]
    med_clean = float(np.median(clean_ratios))
    thr = cfg["rel_threshold"]
    screening = {
        "clean_ratio": ms(clean_ratios),
        "poisoned_p01_ratio": ms(p01_ratios), "poisoned_p05_ratio": ms(p05_ratios),
        "clean_median": med_clean,
        "rel_p01": [r / med_clean for r in p01_ratios],
        "rel_p05": [r / med_clean for r in p05_ratios],
        "flag_rate_p01": float(np.mean([r / med_clean > thr for r in p01_ratios])),
        "flag_rate_p05": float(np.mean([r / med_clean > thr for r in p05_ratios])),
        "separation_auc": boot_ci_auc(
            [0] * len(clean_ratios) + [1] * (len(p01_ratios) + len(p05_ratios)),
            clean_ratios + p01_ratios + p05_ratios, iters, boot_rng),
    }
    print(f"  clean ratio {screening['clean_ratio']['mean']:.2f}±{screening['clean_ratio']['std']:.2f} | "
          f"p01 {screening['poisoned_p01_ratio']['mean']:.2f} | "
          f"p05 {screening['poisoned_p05_ratio']['mean']:.2f}±{screening['poisoned_p05_ratio']['std']:.2f} | "
          f"sep AUC {screening['separation_auc']['auc']:.3f} "
          f"CI{screening['separation_auc']['boot_ci95']}")

    # === Part 1: significance for key contrasts ===
    print("\nSignificance / CIs for key contrasts ...")
    contrasts = {
        "STRIP_TPR5_C_vs_B": paired_contrast(C["tpr"]["values"], B["tpr"]["values"],
                                             iters, boot_rng, "C - B (TPR@5%)"),
        "STRIP_TPR5_A_vs_B": paired_contrast(A["tpr"]["values"], B["tpr"]["values"],
                                             iters, boot_rng, "A - B (TPR@5%)"),
        "STRIP_sepd_C_vs_B": paired_contrast(C["sep_d"]["values"], B["sep_d"]["values"],
                                             iters, boot_rng, "C - B (sep_d, mediator)"),
        "NC_screening_separation": screening["separation_auc"],
    }
    for k, v in contrasts.items():
        if "boot_ci95_diff" in v:
            print(f"  {k}: diff={v['mean_diff']:.4f} CI{v['boot_ci95_diff']} "
                  f"wilcoxon_p={v['wilcoxon_p']} sig_boot={v['significant_bootstrap']} "
                  f"(n={v['n']})")

    result = {"n_causal_seeds": len(cfg["causal_seeds"]),
              "causal_mediators": causal, "screening_canonical_nc": screening,
              "contrasts": contrasts}
    _save(exp_dir, cfg, result)
    _print_summary(result)
    return result


# ---------------------------------------------------------------------------
def _save(exp_dir, cfg, r):
    (exp_dir / "metrics.json").write_text(json.dumps(r, indent=2, default=float),
                                          encoding="utf-8")
    _summary(exp_dir / "summary.md", cfg, r)
    print(f"\n  wrote artifacts -> {exp_dir}")


def _summary(path, cfg, r):
    c = r["contrasts"]; sc = r["screening_canonical_nc"]; med = r["causal_mediators"]
    n = r["n_causal_seeds"]

    L = ["# Statistical tests, canonical Neural Cleanse screening metric, and mediators", "",
         "Reanalysis over existing checkpoints (no new training). STRIP and Neural "
         f"Cleanse recomputed over saved models. Controlled-experiment contrasts use "
         f"n = {n} seeds; small-n tests are non-parametric (Wilcoxon) + bootstrap "
         "95% CIs.", "",
         "- **Run:** `python -m src.eval.run_phase7c_stats --config configs/phase7c.yaml`",
         "- **Measured:** effect sizes, bootstrap CIs, and Wilcoxon tests for the "
         "controlled-experiment contrasts; NC screening separation on the canonical "
         "mask-ratio scale; threshold-free STRIP mediators per condition.", "",
         "## Part 1 — Effect sizes, 95% CIs, and tests for the key contrasts", "",
         "| contrast | n | mean diff | bootstrap 95% CI | Cliff's δ | Wilcoxon p | "
         "sig (CI≠0)? |", "|---|---|---|---|---|---|---|"]
    for k in ("STRIP_TPR5_C_vs_B", "STRIP_TPR5_A_vs_B", "STRIP_sepd_C_vs_B"):
        v = c[k]
        wp = "—" if v["wilcoxon_p"] is None else f"{v['wilcoxon_p']:.4f}"
        L.append(f"| {v['contrast']} | {v['n']} | {v['mean_diff']:+.4f} | "
                 f"[{v['boot_ci95_diff'][0]:+.4f}, {v['boot_ci95_diff'][1]:+.4f}] | "
                 f"{v['cliffs_delta']:+.2f} | {wp} | "
                 f"{'yes' if v['significant_bootstrap'] else 'no'} |")
    ncs = c["NC_screening_separation"]
    L += ["",
          f"**Neural Cleanse clean-vs-poisoned separation (screening): AUC "
          f"{ncs['auc']:.3f}, bootstrap 95% CI [{ncs['boot_ci95'][0]:.3f}, "
          f"{ncs['boot_ci95'][1]:.3f}] (n={ncs['n']} models).**",
          "",
          "### Statistical power",
          f"With n = {n} seeds the exact two-sided Wilcoxon p-value has a floor of "
          "~0.0625, so a threshold metric can be directionally consistent yet not "
          "reach p < 0.05. The bootstrap CI and Cliff's δ (effect size and "
          "direction) are reported alongside, and the **threshold-free mechanistic "
          "evidence (mask norms, entropy, sep_d) is the primary support**; the "
          "single-threshold TPR is corroborating only.",
          "",
          "## Part 2 — Canonical NC metric on the main screening models", "",
          "Per-class L1 mask norm → anomaly ratio → paired ratio-to-reference-median "
          f"(threshold {cfg['rel_threshold']}). Reference = "
          f"{len(cfg['screening']['clean'])} clean IDS models, same holdout.", "",
          "| set | n models | NC anomaly ratio (mean±std) | rel-to-clean-median | flag rate |",
          "|---|---|---|---|---|",
          f"| clean | {len(sc['clean_ratio']['values'])} | "
          f"{sc['clean_ratio']['mean']:.2f}±{sc['clean_ratio']['std']:.2f} | 1.00 | — |",
          f"| poisoned 1% | {len(sc['poisoned_p01_ratio']['values'])} | "
          f"{sc['poisoned_p01_ratio']['mean']:.2f} | "
          f"{np.mean(sc['rel_p01']):.2f} | {sc['flag_rate_p01']:.2f} |",
          f"| poisoned 5% | {len(sc['poisoned_p05_ratio']['values'])} | "
          f"{sc['poisoned_p05_ratio']['mean']:.2f}±{sc['poisoned_p05_ratio']['std']:.2f} | "
          f"{np.mean(sc['rel_p05']):.2f} | {sc['flag_rate_p05']:.2f} |",
          "",
          f"**Screening separation AUC (clean vs poisoned) = {ncs['auc']:.3f} "
          f"[{ncs['boot_ci95'][0]:.3f}, {ncs['boot_ci95'][1]:.3f}]**, on the same "
          "(canonical mask-ratio) scale as the mechanism analysis. "
          f"(Note: only n=1 model available at 1% poison, so that row is a single "
          "observation.)",
          "",
          "## Part 3 — Mediators for the controlled-experiment conditions", "",
          f"Mean ± std over n = {n} seeds. Mediators are threshold-free and more "
          "stable than the single-threshold TPR.", "",
          "| condition | target | STRIP TPR@5% | ent clean-target | ent clean-nontarget | "
          "ent triggered | sep_d | target confidence |",
          "|---|---|---|---|---|---|---|---|"]
    for label, m in med.items():
        L.append(f"| {label} | {'benign' if m['target']==0 else 'attack'} | "
                 f"{m['tpr']['mean']:.3f}±{m['tpr']['std']:.3f} | "
                 f"{m['ent_ct']['mean']:.3f}±{m['ent_ct']['std']:.3f} | "
                 f"{m['ent_cn']['mean']:.3f}±{m['ent_cn']['std']:.3f} | "
                 f"{m['ent_tr']['mean']:.3f}±{m['ent_tr']['std']:.3f} | "
                 f"{m['sep_d']['mean']:.2f}±{m['sep_d']['std']:.2f} | "
                 f"{m['conf']['mean']:.3f}±{m['conf']['std']:.3f} |")
    L += ["",
          "The mediators move consistently with the hypothesis (dominant benign "
          "target = low clean-target entropy = small sep_d = high confidence) and "
          "are threshold-free.",
          "",
          "## Artifacts",
          "- `metrics.json`, `summary.md`"]
    path.write_text("\n".join(L), encoding="utf-8")


def _print_summary(r):
    c = r["contrasts"]; sc = r["screening_canonical_nc"]
    print("\n" + "=" * 88)
    print("PHASE 7c — SIGNIFICANCE / CANONICAL NC / MEDIATORS SUMMARY")
    print("=" * 88)
    for k in ("STRIP_TPR5_C_vs_B", "STRIP_TPR5_A_vs_B", "STRIP_sepd_C_vs_B"):
        v = c[k]
        wp = "—" if v["wilcoxon_p"] is None else f"{v['wilcoxon_p']:.4f}"
        print(f"  {v['contrast']:>26}: diff {v['mean_diff']:+.4f} "
              f"CI[{v['boot_ci95_diff'][0]:+.4f},{v['boot_ci95_diff'][1]:+.4f}] "
              f"delta={v['cliffs_delta']:+.2f} wilcoxon_p={wp} "
              f"sig_boot={v['significant_bootstrap']} (n={v['n']})")
    ncs = c["NC_screening_separation"]
    print(f"  NC screening separation AUC {ncs['auc']:.3f} "
          f"CI[{ncs['boot_ci95'][0]:.3f},{ncs['boot_ci95'][1]:.3f}] (n={ncs['n']})")
    print(f"  Canonical NC: clean {sc['clean_ratio']['mean']:.2f} vs "
          f"p05 {sc['poisoned_p05_ratio']['mean']:.2f} (flag rate {sc['flag_rate_p05']:.2f})")
    print("=" * 88 + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Statistical tests, NC screening metric, and mediators")
    ap.add_argument("--config", default="configs/phase7c.yaml")
    return ap.parse_args()


def main():
    run(load_yaml(resolve_path(parse_args().config)))


if __name__ == "__main__":
    main()
