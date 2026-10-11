"""Multi-seed Neural Cleanse and mask-norm mechanism for the CICIDS2017 controlled
experiment (Table 3 STRIP TPR and NC flag; outputs ``experiments/phase6d/``).

Repeats the controlled-experiment detector comparison over several seeds and
examines *why* Neural Cleanse does not flag condition C (attack-target backdoor).
Reuses the three-condition setup of ``run_causal_test_ids`` unchanged.

Part 1 — train >=5 clean + >=5 poisoned models per condition (seeds aligned
  across conditions), run NC on all, and apply the paired
  ratio-to-reference-median rule (see ``run_nc_calibration``) (reference = clean models of the SAME condition
  and holdout). Report clean/poisoned anomaly ratios (mean +/- std), flag rate,
  and whether the target class is correctly identified as infected.
Part 2 — mechanism: per-class L1 mask norms (clean vs poisoned) show that the
  dominant benign class is naturally cheap to reach (small mask even when clean),
  so on C the backdoor-shrunk attack mask never drops below it.
Part 3 — coverage map by target-class status (STRIP data from
  ``run_causal_test_ids``, NC mask data from here), presented as a map with
  gaps rather than a complete combination.

Run:
    python -m src.eval.run_nc_multiseed_ids --config configs/nc_multiseed_ids.yaml
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.attacks.run_backdoor import make_poisoned_train  # noqa: E402
from src.attacks.trigger import load_trigger  # noqa: E402
from src.detectors.neural_cleanse import NeuralCleanseDetector  # noqa: E402
from src.eval.run_causal_test_ids import (  # noqa: E402  (reuse identical methods)
    strip_auc_tpr,
    subsample,
    train_or_load,
)
from src.models.train import load_dataset  # noqa: E402
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402

CONDS = {
    "A_balanced_tgtBenign": {"mode": "balanced", "target": 0, "source": 1, "rate_key": "A"},
    "B_imbal_tgtBenign": {"mode": "natural", "target": 0, "source": 1, "rate_key": "B"},
    "C_imbal_tgtAttack": {"mode": "natural", "target": 1, "source": 0, "rate_key": "C"},
}


def ms(xs):
    a = np.array(xs, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def run_nc(model, hX, hy, cfg, device):
    out = NeuralCleanseDetector(model, hX, hy, device=device,
                                steps=cfg["neural_cleanse"]["steps"],
                                lr=cfg["neural_cleanse"]["lr"], seed=cfg["seed"]).run()
    return {"l1_benign": out["l1_norms"][0], "l1_attack": out["l1_norms"][1],
            "anomaly_ratio": out["anomaly_ratio"], "infected_class": out["infected_class"]}


def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    mcfg = load_yaml(resolve_path(cfg["training"]["model_config"]))
    data = load_dataset(resolve_path(cfg["paths"]["processed_dir"]))
    trigger = load_trigger(resolve_path(cfg["paths"]["trigger"]))
    Xtr, ytr = data["X_train"], data["y_train"]
    Xte, yte = data["X_test"], data["y_test"]
    benign_frac = float(np.mean(ytr == 0))

    rng = np.random.default_rng(cfg["seed"])
    holdout = data["X_val"][rng.choice(len(data["X_val"]), cfg["strip"]["holdout_size"],
                                       replace=False)].astype(np.float32)
    ncidx = rng.choice(len(data["X_val"]), cfg["neural_cleanse"]["holdout_size"], replace=False)
    nc_hX, nc_hy = data["X_val"][ncidx].astype(np.float32), data["y_val"][ncidx].astype(int)
    es = cfg["strip"]["eval_size"]
    neg = Xte[rng.choice(len(Xte), es, replace=False)].astype(np.float32)

    phase6c = json.loads(resolve_path(cfg["paths"]["phase6c_metrics"]).read_text())["results"]

    per_cond = {}
    for label, c in CONDS.items():
        tgt, src = c["target"], c["source"]
        rate = cfg["poison_rate"][c["rate_key"]]
        print(f"\n=== {label} (target={tgt}, rate={rate}) ===")
        Xs, ys = subsample(Xtr, ytr, c["mode"], cfg["train_subsample"], benign_frac,
                           np.random.default_rng(cfg["seed"]))
        Xp, yp, npois = make_poisoned_train(Xs, ys, trigger, rate, src, tgt,
                                            np.random.default_rng(cfg["seed"]))
        clean_non = Xte[np.where(yte == src)[0][:es]]

        clean_nc, pois_nc, pois_strip_auc, pois_strip_tpr = [], [], [], []
        pois_infected, pois_asr = [], []
        for s in cfg["seeds"]:
            # clean reference model (seed 42 reuses the run_causal_test_ids checkpoint)
            cname = f"{label}_clean" if s == cfg["seed"] else f"{label}_clean_seed{s}"
            cm = train_or_load(cname, Xs, ys, data, s, cfg, mcfg, device)
            clean_nc.append(run_nc(cm, nc_hX, nc_hy, cfg, device))
            # poisoned model (seeds 42-44 reuse run_causal_test_ids checkpoints)
            pm = train_or_load(f"{label}_seed{s}", Xp, yp, data, s, cfg, mcfg, device)
            pnc = run_nc(pm, nc_hX, nc_hy, cfg, device)
            pois_nc.append(pnc)
            pois_infected.append(int(pnc["infected_class"] == tgt))
            auc, tpr, _, _ = strip_auc_tpr(pm, holdout, trigger, clean_non, neg, cfg, device)
            pois_strip_auc.append(auc); pois_strip_tpr.append(tpr)
            print(f"  seed {s}: clean(L1 ben={clean_nc[-1]['l1_benign']:.1f} "
                  f"atk={clean_nc[-1]['l1_attack']:.1f}) | pois(L1 ben={pnc['l1_benign']:.1f} "
                  f"atk={pnc['l1_attack']:.1f} ratio={pnc['anomaly_ratio']:.2f} "
                  f"inf={pnc['infected_class']}) STRIP_TPR5={tpr:.3f}")

        clean_ratios = [m["anomaly_ratio"] for m in clean_nc]
        pois_ratios = [m["anomaly_ratio"] for m in pois_nc]
        med_clean = float(np.median(clean_ratios))
        rels = [r / (med_clean + 1e-9) for r in pois_ratios]
        flags = [rel > cfg["rel_threshold"] for rel in rels]

        cb = ms([m["l1_benign"] for m in clean_nc]); ca = ms([m["l1_attack"] for m in clean_nc])
        pb = ms([m["l1_benign"] for m in pois_nc]); pa = ms([m["l1_attack"] for m in pois_nc])
        target_key = "l1_benign" if tgt == 0 else "l1_attack"
        clean_tgt_norm = ms([m[target_key] for m in clean_nc])["mean"]
        pois_tgt_norm = ms([m[target_key] for m in pois_nc])["mean"]

        per_cond[label] = {
            "target": tgt, "source": src, "poison_rate": rate, "n_poisoned": npois,
            "n_seeds": len(cfg["seeds"]),
            "nc_clean_ratio": ms(clean_ratios), "nc_pois_ratio": ms(pois_ratios),
            "rel_to_clean_median": ms(rels), "flag_rate": float(np.mean(flags)),
            "target_infected_rate": float(np.mean(pois_infected)),
            "mask_clean_benign": cb, "mask_clean_attack": ca,
            "mask_pois_benign": pb, "mask_pois_attack": pa,
            "reachability_clean_benign": cb["mean"], "reachability_clean_attack": ca["mean"],
            "backdoor_signature_drop": clean_tgt_norm - pois_tgt_norm,
            "target_clean_mask": clean_tgt_norm, "target_pois_mask": pois_tgt_norm,
            "strip_auc": ms(pois_strip_auc), "strip_tpr5": ms(pois_strip_tpr),
            # STRIP mechanism driver from run_causal_test_ids (entropy of the clean target class)
            "phase6c_clean_target_entropy": phase6c[label]["entropy_mean"]["clean_target"],
            "phase6c_strip_tpr5": phase6c[label]["strip_tpr5_mean"],
        }
        r = per_cond[label]
        print(f"  => NC clean ratio {r['nc_clean_ratio']['mean']:.2f}±{r['nc_clean_ratio']['std']:.2f} "
              f"| pois ratio {r['nc_pois_ratio']['mean']:.2f}±{r['nc_pois_ratio']['std']:.2f} "
              f"| rel {r['rel_to_clean_median']['mean']:.2f} flag_rate {r['flag_rate']:.2f} "
              f"tgt_infected {r['target_infected_rate']:.2f} | STRIP TPR5 "
              f"{r['strip_tpr5']['mean']:.3f}±{r['strip_tpr5']['std']:.3f}")

    insight = _unifying_insight(per_cond)
    coverage = _coverage_map(per_cond)
    plot_masks(per_cond, exp_dir / "mask_norms_per_class.png")
    plot_contrast(per_cond, exp_dir / "coverage_contrast.png")
    _save(exp_dir, cfg, per_cond, insight, coverage)
    _print(per_cond, insight, coverage)
    return per_cond


def _unifying_insight(pc):
    """STRIP ~ target-class entropy; NC ~ target-class reachability (opposite)."""
    rows = {}
    for label, r in pc.items():
        rows[label] = {
            "target": r["target"],
            "target_entropy_strip_driver": r["phase6c_clean_target_entropy"],
            "target_reachability_nc_driver": r["target_clean_mask"],
            "strip_tpr5": r["strip_tpr5"]["mean"], "nc_flag_rate": r["flag_rate"]}
    B, C = pc["B_imbal_tgtBenign"], pc["C_imbal_tgtAttack"]
    # B and C share the imbalance; only the target class differs.
    supported = (B["phase6c_clean_target_entropy"] < C["phase6c_clean_target_entropy"]
                 and B["target_clean_mask"] < C["target_clean_mask"]
                 and B["flag_rate"] > C["flag_rate"]
                 and C["strip_tpr5"]["mean"] > B["strip_tpr5"]["mean"])
    text = (
        "Unifying insight (B vs C, same imbalance, opposite target): STRIP is "
        f"governed by target-class ENTROPY — the dominant benign target has low "
        f"clean entropy ({B['phase6c_clean_target_entropy']:.3f} vs "
        f"{C['phase6c_clean_target_entropy']:.3f} for attack), so STRIP is blindest "
        f"when the target is dominant (B TPR {B['strip_tpr5']['mean']:.3f} < C "
        f"{C['strip_tpr5']['mean']:.3f}). Neural Cleanse is governed by target-class "
        f"REACHABILITY — the dominant benign target has a naturally tiny clean mask "
        f"({B['target_clean_mask']:.1f} vs {C['target_clean_mask']:.1f} for attack), "
        f"so a benign-target backdoor stands out (B flag_rate {B['flag_rate']:.2f}) "
        f"while an attack-target backdoor does not (C flag_rate {C['flag_rate']:.2f}). "
        "Both are governed by target-class dominance, in OPPOSITE directions.")
    return {"per_condition": rows, "supported": bool(supported), "statement": text}


def _coverage_map(pc):
    B, C = pc["B_imbal_tgtBenign"], pc["C_imbal_tgtAttack"]
    strip_max = max(pc[k]["strip_tpr5"]["mean"] for k in pc)
    gmap = {
        "target_benign_dominant (B)": {
            "STRIP_tpr5": f"{B['strip_tpr5']['mean']:.3f}±{B['strip_tpr5']['std']:.3f}",
            "NC_flag_rate": f"{B['flag_rate']:.2f}",
            "blind_spot": "STRIP (input-level) is blind; NC covers this regime"},
        "target_attack_minority (C)": {
            "STRIP_tpr5": f"{C['strip_tpr5']['mean']:.3f}±{C['strip_tpr5']['std']:.3f}",
            "NC_flag_rate": f"{C['flag_rate']:.2f}",
            "blind_spot": "NC (model-level) is blind; STRIP only marginally better "
                          "and still far below usable"},
    }
    framing = (
        f"This is a coverage map WITH GAPS, not a winning combination. STRIP stays "
        f"weak in absolute terms in EVERY condition (max TPR@5%FPR ~= {strip_max:.3f}), "
        f"so pairing it with NC does NOT close the attack-target (C) gap — when NC "
        f"goes blind there, STRIP cannot pick up the slack. The attack-target "
        f"backdoor is an OPEN detection gap and a caution to practitioners: "
        f"model-level screening does not cover all backdoor types.")
    return {"map": gmap, "coverage_note": framing, "strip_max_tpr5": strip_max}


# ---------------------------------------------------------------------------
def plot_masks(pc, out):
    labels = list(pc.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, label in zip(axes, labels):
        r = pc[label]; tgt = r["target"]
        groups = ["benign\nmask", "attack\nmask"]
        x = np.arange(2); w = 0.38
        clean = [r["mask_clean_benign"]["mean"], r["mask_clean_attack"]["mean"]]
        clean_e = [r["mask_clean_benign"]["std"], r["mask_clean_attack"]["std"]]
        pois = [r["mask_pois_benign"]["mean"], r["mask_pois_attack"]["mean"]]
        pois_e = [r["mask_pois_benign"]["std"], r["mask_pois_attack"]["std"]]
        ax.bar(x - w / 2, clean, w, yerr=clean_e, capsize=3, color="#2980b9", label="clean")
        ax.bar(x + w / 2, pois, w, yerr=pois_e, capsize=3, color="#c0392b", label="poisoned")
        ax.set_xticks(x, labels=groups)
        ax.set_title(f"{label}\n(target = {'benign' if tgt==0 else 'attack'})", fontsize=9)
        # mark target class
        ax.axvline(tgt, color="k", ls=":", alpha=0.4)
    axes[0].set_ylabel("Neural Cleanse L1 mask norm")
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-class NC mask norms (clean vs poisoned) — the dominant benign "
                 "mask is naturally tiny, so on C the attack mask can't drop below it")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_contrast(pc, out):
    B, C = pc["B_imbal_tgtBenign"], pc["C_imbal_tgtAttack"]
    regimes = ["benign-dominant\n(B)", "attack-minority\n(C)"]
    strip = [B["strip_tpr5"]["mean"], C["strip_tpr5"]["mean"]]
    strip_e = [B["strip_tpr5"]["std"], C["strip_tpr5"]["std"]]
    ncf = [B["flag_rate"], C["flag_rate"]]
    x = np.arange(2); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - w / 2, ncf, w, color="#27ae60", label="NC flag rate")
    ax.bar(x + w / 2, strip, w, yerr=strip_e, capsize=4, color="#c0392b",
           label="STRIP TPR@5%FPR")
    ax.set_xticks(x, labels=regimes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection performance")
    ax.set_title("Opposite target-class blind spots (coverage map with a gap)")
    ax.legend()
    for i, (n, s) in enumerate(zip(ncf, strip)):
        ax.text(i - w / 2, n + 0.02, f"{n:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, s + 0.02, f"{s:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def _save(exp_dir, cfg, pc, insight, coverage):
    (exp_dir / "metrics.json").write_text(
        json.dumps({"per_condition": pc, "unifying_insight": insight,
                    "coverage_map": coverage, "seeds": cfg["seeds"]},
                   indent=2, default=float), encoding="utf-8")
    _summary(exp_dir / "summary.md", cfg, pc, insight, coverage)
    print(f"\n  wrote artifacts -> {exp_dir}")


def _summary(path, cfg, pc, insight, coverage):
    L = ["# Multi-seed Neural Cleanse and mask-norm mechanism (CICIDS2017 "
         "controlled experiment)", "",
         f"Repeats the Neural Cleanse evaluation of the CICIDS2017 controlled "
         f"experiment over several seeds ({len(cfg['seeds'])} seeds: {cfg['seeds']}) "
         "and examines why NC does not flag condition C. Same three conditions, "
         "protocol, trigger, and clean holdout; paired ratio-to-reference-median "
         "rule (see `src.eval.run_nc_calibration`), no absolute threshold.",
         "",
         "- **Run:** `python -m src.eval.run_nc_multiseed_ids --config "
         "configs/nc_multiseed_ids.yaml`",
         "- **Measured:** per condition, clean and poisoned NC anomaly ratios, flag "
         "rate, target-identification rate, per-class L1 mask norms, and the "
         "target-class entropy/reachability mediators.",
         "",
         "## Part 1 — Multi-seed Neural Cleanse (5 clean + 5 poisoned per condition)",
         "",
         "| condition | target | NC clean ratio | NC poisoned ratio | rel-to-median | "
         "flag rate | target=infected rate | STRIP TPR@5% |",
         "|---|---|---|---|---|---|---|---|"]
    for label, r in pc.items():
        L.append(f"| {label} | {'benign' if r['target']==0 else 'attack'} | "
                 f"{r['nc_clean_ratio']['mean']:.2f}±{r['nc_clean_ratio']['std']:.2f} | "
                 f"{r['nc_pois_ratio']['mean']:.2f}±{r['nc_pois_ratio']['std']:.2f} | "
                 f"{r['rel_to_clean_median']['mean']:.2f} | {r['flag_rate']:.2f} | "
                 f"{r['target_infected_rate']:.2f} | "
                 f"{r['strip_tpr5']['mean']:.3f}±{r['strip_tpr5']['std']:.3f} |")
    L += ["",
          "Hypothesis tested: NC flags benign-target backdoors (A, B) but not the "
          "attack-target one (C), evaluated over a seed distribution rather than a "
          "single run.",
          "",
          "## Part 2 — Mask-norm mechanism for condition C", "",
          "Per-class L1 mask norm (mean±std over seeds):", "",
          "| condition | target | clean benign | clean attack | poisoned benign | "
          "poisoned attack | backdoor drop (target) |",
          "|---|---|---|---|---|---|---|"]
    for label, r in pc.items():
        L.append(f"| {label} | {'benign' if r['target']==0 else 'attack'} | "
                 f"{r['mask_clean_benign']['mean']:.1f}±{r['mask_clean_benign']['std']:.1f} | "
                 f"{r['mask_clean_attack']['mean']:.1f}±{r['mask_clean_attack']['std']:.1f} | "
                 f"{r['mask_pois_benign']['mean']:.1f}±{r['mask_pois_benign']['std']:.1f} | "
                 f"{r['mask_pois_attack']['mean']:.1f}±{r['mask_pois_attack']['std']:.1f} | "
                 f"{r['backdoor_signature_drop']:.1f} |")
    Bc, Cc = pc["B_imbal_tgtBenign"], pc["C_imbal_tgtAttack"]
    L += ["",
          "**Natural reachability (clean models):** the dominant benign class has a "
          f"far smaller clean mask than the attack class "
          f"(e.g. B: benign {Bc['mask_clean_benign']['mean']:.1f} vs attack "
          f"{Bc['mask_clean_attack']['mean']:.1f}) — benign is cheap to reach even "
          "without a backdoor.",
          "",
          f"**B flagged, C not flagged:** on **B** the backdoor shrinks the "
          f"benign mask from {Bc['mask_clean_benign']['mean']:.1f} to "
          f"{Bc['mask_pois_benign']['mean']:.1f} — already the smallest, now tiny, so "
          f"it stands out. On **C** the backdoor shrinks the attack mask from "
          f"{Cc['mask_clean_attack']['mean']:.1f} to {Cc['mask_pois_attack']['mean']:.1f}, "
          f"but that is still not clearly below the naturally small benign mask "
          f"({Cc['mask_pois_benign']['mean']:.1f}), so the 'smallest-mask = infected' "
          f"rule mis-points or the ratio stays under threshold (rel "
          f"{Cc['rel_to_clean_median']['mean']:.2f} < {cfg['rel_threshold']}).",
          "",
          "## Part 2.9 — Target-class entropy and reachability", "",
          f"{insight['statement']}",
          "",
          f"**Supported by the data: {insight['supported']}.**",
          "",
          "| condition | target | target entropy (STRIP driver) | target reachability = "
          "clean mask (NC driver) | STRIP TPR@5% | NC flag rate |",
          "|---|---|---|---|---|---|"]
    for label, r in insight["per_condition"].items():
        L.append(f"| {label} | {'benign' if r['target']==0 else 'attack'} | "
                 f"{r['target_entropy_strip_driver']:.3f} | "
                 f"{r['target_reachability_nc_driver']:.1f} | "
                 f"{r['strip_tpr5']:.3f} | {r['nc_flag_rate']:.2f} |")
    L += ["",
          "## Part 3 — Coverage map by target class", "",
          "| target-class regime | STRIP TPR@5% | NC flag rate | blind spot |",
          "|---|---|---|---|"]
    for regime, v in coverage["map"].items():
        L.append(f"| {regime} | {v['STRIP_tpr5']} | {v['NC_flag_rate']} | {v['blind_spot']} |")
    L += ["",
          f"**{coverage['coverage_note']}**",
          "",
          "## Artifacts",
          "- `metrics.json`, `summary.md`",
          "- `mask_norms_per_class.png`, `coverage_contrast.png`",
          "- Models `results/models/causal_ids_*.pt` (outside git)."]
    path.write_text("\n".join(L), encoding="utf-8")


def _print(pc, insight, coverage):
    print("\n" + "=" * 100)
    print("PHASE 6d — MULTI-SEED NC + MECHANISM SUMMARY")
    print("=" * 100)
    print(f"{'condition':>22} | {'NC clean':>12} | {'NC pois':>12} | {'flag':>5} | "
          f"{'tgt_inf':>7} | {'STRIP TPR5':>12}")
    for label, r in pc.items():
        print(f"{label:>22} | {r['nc_clean_ratio']['mean']:>6.2f}±{r['nc_clean_ratio']['std']:<5.2f} | "
              f"{r['nc_pois_ratio']['mean']:>6.2f}±{r['nc_pois_ratio']['std']:<5.2f} | "
              f"{r['flag_rate']:>5.2f} | {r['target_infected_rate']:>7.2f} | "
              f"{r['strip_tpr5']['mean']:>6.3f}±{r['strip_tpr5']['std']:<5.3f}")
    print("-" * 100)
    print(f"Unifying insight supported: {insight['supported']}")
    print(f"Coverage: STRIP max TPR@5% = {coverage['strip_max_tpr5']:.3f} "
          "-> attack-target gap remains OPEN (not closed by combining detectors)")
    print("=" * 100 + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-seed Neural Cleanse and mask-norm mechanism")
    ap.add_argument("--config", default="configs/nc_multiseed_ids.yaml")
    return ap.parse_args()


def main():
    run(load_yaml(resolve_path(parse_args().config)))


if __name__ == "__main__":
    main()
