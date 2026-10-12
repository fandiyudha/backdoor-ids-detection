"""v1.1 Experiment C: IBD-PSC input-level runtime detection (outputs
``experiments/v11_ibd_psc/``; also used for the 1D-CNN).

Evaluates IBD-PSC (``src.detectors.ibd_psc``) in the runtime setting of the
paper's STRIP evaluation (``run_phase7f_strip_reeval``): the same clean holdout,
triggered positives, and clean negatives, on every benign/normal-target poisoned
model of both datasets. STRIP (blend, alpha 0.7, N = 100) is rerun on the same
models and inputs for a paired comparison; on the existing models it reproduces
phase7f.

Reports ROC-AUC (triggered vs clean), TPR at 5% FPR with the threshold set on the
clean holdout only, TPR/FPR at the IBD-PSC paper's fixed threshold T = 0.9, the
per-input cost, the PSC distributions for triggered / clean benign / clean attack
inputs (analogous to the article's Figure 2), and, on the CICIDS2017
controlled-experiment conditions A/B/C, whether IBD-PSC also tracks the
target-class homogeneity that governs STRIP.

Run:
    python -u -m src.eval.run_v11_ibd_psc --config configs/v11_ibd_psc.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src.attacks.trigger import apply_trigger, build_trigger, load_trigger  # noqa: E402
from src.detectors.ibd_psc import IBDPSC  # noqa: E402
from src.eval.run_phase7f_strip_reeval import _strip  # noqa: E402
from src.eval.v11_common import (  # noqa: E402
    CPU,
    environment,
    git_commit,
    load_model_cpu,
    mean_ci,
    model_names,
    models_manifest_entry,
    utc_now,
    write_json,
)
from src.models.train import load_dataset  # noqa: E402
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


def _auc(pos, neg):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def eval_ibd(model, holdout, holdout_y, pos, neg, cfg):
    """IBD-PSC metrics for one model. Higher PSC = more suspicious."""
    ic = cfg["ibd_psc"]
    t0 = time.perf_counter()
    det = IBDPSC(model, holdout[:ic["n_select"]], holdout_y[:ic["n_select"]], omega=ic["omega"],
                 xi=ic["xi"], n=ic["n"], extra_omegas=ic["extra_omegas"])
    setup_s = time.perf_counter() - t0
    s_hold = det.score(holdout)
    thr = float(np.quantile(s_hold, 1 - cfg["fpr_target"]))
    t0 = time.perf_counter()
    s_pos = det.score(pos)
    s_neg = det.score(neg)
    per_input_ms = 1000 * (time.perf_counter() - t0) / (len(pos) + len(neg))
    T = ic["T"]
    return {"detector": det.describe(), "auc": _auc(s_pos, s_neg),
            "threshold_5fpr": thr, "tpr_5fpr": float(np.mean(s_pos > thr)),
            "fpr_5fpr_negatives": float(np.mean(s_neg > thr)),
            "tpr_T": float(np.mean(s_pos >= T)), "fpr_T_negatives": float(np.mean(s_neg >= T)),
            "fpr_T_holdout": float(np.mean(s_hold >= T)),
            "psc_mean": {"triggered": float(s_pos.mean()), "negatives": float(s_neg.mean()),
                         "holdout": float(s_hold.mean())},
            "ties_at_threshold_holdout": int(np.sum(s_hold == thr)),
            "per_input_ms": per_input_ms, "setup_s": setup_s,
            "forward_passes_per_input": det.describe()["n_models"] + 1}


def eval_strip(model, holdout, pos, neg, cfg, p7f):
    t0 = time.perf_counter()
    r = _strip(model, holdout, pos, neg, cfg["strip"]["alpha"], cfg["strip"]["variant"],
               {"N": cfg["strip"]["N"], "seed": p7f["seed"], "fpr_targets": p7f["fpr_targets"]}, CPU)
    elapsed = time.perf_counter() - t0
    n_scored = len(holdout) + len(pos) + len(neg)
    return {"auc": r["auc"], "tpr_5fpr": r["tpr5"],
            "fpr_5fpr_negatives": float(np.mean(r["e_neg"] < r["thr"][0.05])),
            "per_input_ms": 1000 * elapsed / n_scored, "forward_passes_per_input": cfg["strip"]["N"]}


def agg(rows, key, iters, seed):
    return mean_ci([r[key] for r in rows], iters, seed)


def summarize(per_model, n_exist, iters, seed):
    names = list(per_model)
    keys_i = ("auc", "tpr_5fpr", "fpr_5fpr_negatives", "tpr_T", "fpr_T_negatives",
              "fpr_T_holdout", "per_input_ms")
    keys_s = ("auc", "tpr_5fpr", "fpr_5fpr_negatives", "per_input_ms")
    out = {}
    for lbl, sel in (("existing_n5", names[:n_exist]), ("new_n5", names[n_exist:]), ("all_n10", names)):
        out[lbl] = {"ibd_psc": {k: agg([per_model[n]["ibd_psc"] for n in sel], k, iters, seed)
                                for k in keys_i},
                    "strip": {k: agg([per_model[n]["strip"] for n in sel], k, iters, seed)
                              for k in keys_s}}
    return out


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    t_start = time.perf_counter()
    mcfg = load_yaml(resolve_path(cfg["models_from"]))
    p7f = load_yaml(resolve_path(cfg["protocol_from"]))
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    md = resolve_path(cfg["paths"]["models_dir"])
    iters, seed = cfg["bootstrap_iters"], cfg["seed"]
    import joblib

    out = {"datasets": {}}
    for dname, pc in p7f["datasets"].items():
        print(f"\n==================== {dname} ====================", flush=True)
        data = load_dataset(resolve_path(pc["processed_dir"]))
        feat = [str(f) for f in data["feature_names"]]
        if "trigger_json" in pc:
            trigger = load_trigger(resolve_path(pc["trigger_json"]))
        else:
            scaler = joblib.load(resolve_path(p7f["paths"]["scaler_unsw"]))
            trigger = build_trigger(feat, scaler, data["X_train"], pc["trigger_build"]["features"],
                                    mode="constant", constants=pc["trigger_build"]["constants"],
                                    enforce_controllable=False)
        # replay phase7f's draws (get_holdout_eval, then the negatives)
        rng = np.random.default_rng(p7f["seed"])
        hk, yk = ("X_val", "y_val") if "X_val" in data else ("X_test", "y_test")
        hidx = rng.choice(len(data[hk]), min(p7f["holdout_size"], len(data[hk])), replace=False)
        holdout, holdout_y = data[hk][hidx].astype(np.float32), data[yk][hidx].astype(int)
        Xte, yte = data["X_test"], data["y_test"]
        es, tgt, src = p7f["eval_size"], pc["target"], pc["source"]
        src_idx = np.where(yte == src)[0][:es]
        pos = apply_trigger(Xte[src_idx].astype(np.float32), trigger)
        nidx = rng.choice(len(Xte), es, replace=False)
        neg = Xte[nidx].astype(np.float32)
        overlap = (int(len(np.intersect1d(hidx, nidx))) if hk == "X_test" else 0)
        print(f"  holdout {hk} ({len(holdout)}), positives {len(pos)}, negatives {len(neg)}; "
              f"holdout/negative overlap {overlap} rows", flush=True)

        names = model_names(mcfg["datasets"][dname])["poisoned"]
        per = {}
        for n in names:
            m = load_model_cpu(md, n)
            ib = eval_ibd(m, holdout, holdout_y, pos, neg, cfg)
            st = eval_strip(m, holdout, pos, neg, cfg, p7f)
            per[n] = {"ibd_psc": ib, "strip": st}
            d = ib["detector"]
            print(f"  {n}: IBD-PSC k={d['k']} (xi reached {d['xi_reached']}, errs "
                  f"{d['selection_error_by_layers']}) AUC {ib['auc']:.3f} TPR@5 {ib['tpr_5fpr']:.3f} "
                  f"TPR@T {ib['tpr_T']:.3f} FPR@T {ib['fpr_T_negatives']:.3f} | STRIP AUC "
                  f"{st['auc']:.3f} TPR@5 {st['tpr_5fpr']:.3f}", flush=True)
        n_exist = len(mcfg["datasets"][dname]["existing"]["poisoned"])
        summ = summarize(per, n_exist, iters, seed)
        ref = json.loads(resolve_path("experiments/phase7f/metrics.json").read_text(encoding="utf-8"))
        ref07 = ref[dname]["part1_alpha_sweep"]["per_alpha"][f"alpha_{cfg['strip']['alpha']}"]
        fidelity = {"phase7f_strip_auc_mean": ref07["auc"]["mean"],
                    "phase7f_strip_tpr5_mean": ref07["tpr5"]["mean"],
                    "rerun_strip_auc_mean": summ["existing_n5"]["strip"]["auc"]["mean"],
                    "rerun_strip_tpr5_mean": summ["existing_n5"]["strip"]["tpr_5fpr"]["mean"]}
        fidelity["max_abs_diff"] = max(
            abs(fidelity["phase7f_strip_auc_mean"] - fidelity["rerun_strip_auc_mean"]),
            abs(fidelity["phase7f_strip_tpr5_mean"] - fidelity["rerun_strip_tpr5_mean"]))
        print(f"  STRIP fidelity vs phase7f: max |diff| {fidelity['max_abs_diff']:.2e}", flush=True)

        # distributions on the first poisoned model (as the article's Figure 2)
        nf = cfg["distribution_size"]
        rep = load_model_cpu(md, names[0])
        det = IBDPSC(rep, holdout[:cfg["ibd_psc"]["n_select"]], holdout_y[:cfg["ibd_psc"]["n_select"]],
                     omega=cfg["ibd_psc"]["omega"], xi=cfg["ibd_psc"]["xi"], n=cfg["ibd_psc"]["n"],
                     extra_omegas=cfg["ibd_psc"]["extra_omegas"])
        g_b = Xte[np.where(yte == 0)[0][:nf]].astype(np.float32)
        g_a = Xte[np.where(yte == 1)[0][:nf]].astype(np.float32)
        dist = {"clean_benign": det.score(g_b), "clean_attack": det.score(g_a),
                "triggered": det.score(apply_trigger(Xte[np.where(yte == src)[0][:nf]], trigger))}
        rows = ["group,psc"] + [f"{g},{v:.10f}" for g, a in dist.items() for v in a]
        (exp_dir / f"psc_raw_{dname}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        dist_stats = {g: {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
                          "std": float(a.std()), "frac_ge_T": float(np.mean(a >= cfg["ibd_psc"]["T"]))}
                      for g, a in dist.items()}
        dist_stats["roc_auc_triggered_vs_clean_benign"] = _auc(dist["triggered"], dist["clean_benign"])
        dist_stats["model"] = names[0]
        out["datasets"][dname] = {"per_model": per, "summary": summ, "strip_fidelity": fidelity,
                                  "holdout_split": hk, "holdout_negative_overlap_rows": overlap,
                                  "n_positives": int(len(pos)), "n_negatives": int(len(neg)),
                                  "distribution": dist_stats, "_dist": dist,
                                  "target": tgt, "source": src}

    if cfg["controlled"]["enabled"]:
        out["controlled"] = run_controlled(cfg, md, iters, seed)

    out["runtime_s"] = time.perf_counter() - t_start
    _plot_dist(out, cfg, exp_dir / "psc_distributions.png", exp_dir / "psc_roc.png")
    for d in out["datasets"].values():
        d.pop("_dist")
    if "controlled" in out:
        _plot_controlled(out["controlled"], exp_dir / "controlled_conditions.png")
    write_json(exp_dir / "metrics.json", out)
    _manifest(exp_dir, cfg, mcfg, out, md)
    _summary(exp_dir / "summary.md", cfg, out)
    print(f"\n  wrote artifacts -> {exp_dir} ({out['runtime_s']:.0f}s)", flush=True)
    return out


def run_controlled(cfg, md, iters, seed):
    cc = cfg["controlled"]
    data = load_dataset(resolve_path(cc["processed_dir"]))
    trigger = load_trigger(resolve_path(cc["trigger_json"]))
    Xte, yte = data["X_test"], data["y_test"]
    rng = np.random.default_rng(seed)        # replay phase6d's draws
    hidx = rng.choice(len(data["X_val"]), cc["holdout_size"], replace=False)
    holdout, holdout_y = data["X_val"][hidx].astype(np.float32), data["y_val"][hidx].astype(int)
    rng.choice(len(data["X_val"]), cc["nc_holdout_size"], replace=False)   # NC holdout (unused)
    es = cc["eval_size"]
    neg = Xte[rng.choice(len(Xte), es, replace=False)].astype(np.float32)
    p6d = json.loads(resolve_path(cc["phase6d_metrics"]).read_text(encoding="utf-8"))["per_condition"]
    res = {}
    for cname, c in cc["conditions"].items():
        tgt, src = c["target"], c["source"]
        pos = apply_trigger(Xte[np.where(yte == src)[0][:es]].astype(np.float32), trigger)
        clean_tgt = Xte[np.where(yte == tgt)[0][:es]].astype(np.float32)
        rows = []
        for s in cc["seeds"]:
            m = load_model_cpu(md, f"causal_ids_{cname}_seed{s}")
            r = eval_ibd(m, holdout, holdout_y, pos, neg, cfg)
            det = IBDPSC(m, holdout[:cfg["ibd_psc"]["n_select"]], holdout_y[:cfg["ibd_psc"]["n_select"]],
                         omega=cfg["ibd_psc"]["omega"], xi=cfg["ibd_psc"]["xi"], n=cfg["ibd_psc"]["n"],
                         extra_omegas=cfg["ibd_psc"]["extra_omegas"])
            r["psc_clean_target_mean"] = float(det.score(clean_tgt).mean())
            rows.append(r)
            print(f"  controlled {cname} seed {s}: k={r['detector']['k']} AUC {r['auc']:.3f} "
                  f"TPR@5 {r['tpr_5fpr']:.3f} TPR@T {r['tpr_T']:.3f}", flush=True)
        res[cname] = {"target": tgt, "source": src,
                      "auc": agg(rows, "auc", iters, seed), "tpr_5fpr": agg(rows, "tpr_5fpr", iters, seed),
                      "tpr_T": agg(rows, "tpr_T", iters, seed),
                      "fpr_T_negatives": agg(rows, "fpr_T_negatives", iters, seed),
                      "psc_clean_target_mean": agg(rows, "psc_clean_target_mean", iters, seed),
                      "k_per_model": [r["detector"]["k"] for r in rows],
                      "n_models_per_model": [r["detector"]["n_models"] for r in rows],
                      "strip_tpr5_phase6d": p6d[cname]["strip_tpr5"],
                      "strip_auc_phase6d": p6d[cname]["strip_auc"],
                      "clean_target_entropy_phase6c": p6d[cname]["phase6c_clean_target_entropy"]}
    return res


def _manifest(exp_dir, cfg, mcfg, out, md):
    models = []
    for dname in out["datasets"]:
        dc = mcfg["datasets"][dname]
        seeds = list(dc.get("existing", {}).get("seeds", [])) + list(dc["new"]["seeds"])
        for n, s in zip(model_names(dc)["poisoned"], seeds):
            models.append(models_manifest_entry(n, md, dataset=dname, role="poisoned", seed=s))
    if "controlled" in out:
        for cname in out["controlled"]:
            for s in cfg["controlled"]["seeds"]:
                models.append(models_manifest_entry(f"causal_ids_{cname}_seed{s}", md,
                                                    dataset="CICIDS2017 controlled", role="poisoned",
                                                    seed=s))
    write_json(exp_dir / "manifest.json", {
        "experiment": exp_dir.name, "config": cfg.get("_config_path"),
        "command": f"python -u -m src.eval.run_v11_ibd_psc --config {cfg.get('_config_path')}",
        "evaluated_at": utc_now(), "git_commit": git_commit(), "environment": environment(),
        "runtime_s": out["runtime_s"], "models": models})


# ---------------------------------------------------------------------------
def _plot_dist(out, cfg, path_dist, path_roc):
    ds = list(out["datasets"])
    cols = {"clean_benign": ("#2b6cb0", "clean benign/normal"), "clean_attack": ("#2f855a", "clean attack"),
            "triggered": ("#c53030", "triggered")}
    fig, axes = plt.subplots(1, len(ds), figsize=(5.6 * len(ds), 4.0), squeeze=False)
    for ax, d in zip(axes[0], ds):
        dist = out["datasets"][d]["_dist"]
        bins = np.linspace(0, 1, 51)
        for g, (c, lbl) in cols.items():
            ax.hist(dist[g], bins=bins, density=True, histtype="step", lw=1.8, color=c,
                    label=f"{lbl} (n={len(dist[g])})")
        ax.axvline(cfg["ibd_psc"]["T"], color="#444", ls="--", lw=1, label=f"T = {cfg['ibd_psc']['T']}")
        ax.set_yscale("log")
        ax.set_xlabel("PSC score (higher = more suspicious)")
        ax.set_ylabel("density (log)")
        ax.set_title(f"{d}: IBD-PSC scores, {out['datasets'][d]['distribution']['model']}", fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path_dist, dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(ds), figsize=(4.6 * len(ds), 4.2), squeeze=False)
    for ax, d in zip(axes[0], ds):
        dist = out["datasets"][d]["_dist"]
        y = np.r_[np.ones(len(dist["triggered"])), np.zeros(len(dist["clean_benign"]))]
        fpr, tpr, _ = roc_curve(y, np.r_[dist["triggered"], dist["clean_benign"]])
        a = out["datasets"][d]["distribution"]["roc_auc_triggered_vs_clean_benign"]
        ax.plot(fpr, tpr, color="#c53030", lw=2, label=f"IBD-PSC (AUC={a:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("FPR (clean benign/normal)"); ax.set_ylabel("TPR (triggered)")
        ax.set_title(d, fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path_roc, dpi=150); plt.close(fig)


def _plot_controlled(res, path):
    names = list(res)
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x - w / 2, [res[n]["tpr_5fpr"]["mean"] for n in names], w, color="#c53030",
           yerr=[res[n]["tpr_5fpr"]["std"] for n in names], capsize=3, label="IBD-PSC TPR@5% FPR")
    ax.bar(x + w / 2, [res[n]["strip_tpr5_phase6d"]["mean"] for n in names], w, color="#718096",
           yerr=[res[n]["strip_tpr5_phase6d"]["std"] for n in names], capsize=3,
           label="STRIP TPR@5% FPR (phase6d)")
    ax.set_xticks(x, labels=[n.split("_")[0] for n in names])
    ax.set_ylabel("TPR at 5% FPR (mean over 5 models)")
    ax.set_title("CICIDS2017 controlled conditions (A balanced, B benign target, C attack target)",
                 fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _m(a, nd=3):
    return f"{a['mean']:.{nd}f} [{a['boot_ci95'][0]:.{nd}f}, {a['boot_ci95'][1]:.{nd}f}]"


def _summary(path, cfg, out):
    ic = cfg["ibd_psc"]
    L = ["# Extended experiment C: IBD-PSC input-level runtime detection", "",
         f"- **Run:** `python -u -m src.eval.run_v11_ibd_psc --config {cfg.get('_config_path')}`",
         "- **Reference:** L. Hou, R. Feng, Z. Hua, W. Luo, L. Y. Zhang, Y. Li, \"IBD-PSC: Input-level "
         "backdoor detection via parameter-oriented scaling consistency,\" ICML 2024, PMLR "
         "235:18992–19022; reference code in BackdoorBox, adapted to tabular inputs.",
         "- **Setting:** the runtime setting of the paper's STRIP evaluation (`phase7f`): the same "
         "3,000-row clean holdout, the trigger stamped on the first 1,500 source-class (attack) test "
         "rows as positives, and the same 1,500 random clean test rows as negatives, on every "
         "benign/normal-target poisoned model (5 existing + 5 new per dataset). Thresholds are set on "
         "the clean holdout only. STRIP (blend, α 0.7, N = 100) is rerun on the same models and inputs.",
         "", "## Adaptation to the tabular models", "",
         f"- Amplification: γ and β of the deepest m `BatchNorm1d` layers multiplied by ω = {ic['omega']}.",
         f"- Adaptive start k: m = 1, 2, … until the error rate (against the true labels) on the first "
         f"{ic['n_select']} rows of the clean holdout exceeds ξ = {ic['xi']}. If ξ is never exceeded "
         "with all BN layers scaled, k = number of BN layers.",
         f"- Amplified models: model i scales the deepest k + i − 1 layers (the paper's description; "
         f"BackdoorBox scales one layer more). The models have 3 BN layers, so at most 3 − k + 1 layer "
         f"counts exist; the set is completed by also using ω ∈ {{{', '.join(str(w) for w in [ic['omega'], *ic['extra_omegas']])}}}, "
         f"ordered by ω and then by layer count, and the first n = {ic['n']} are kept.",
         "- PSC(x) = mean over the amplified models of the softmax probability of the label the original "
         "model predicts (softmax in float64 to avoid ties at 1.0). Flag if PSC ≥ T (fixed threshold "
         f"T = {ic['T']} of the IBD-PSC paper) or if PSC exceeds the 95th percentile of the clean-holdout "
         "PSC (5% FPR operating point).", ""]
    for d, r in out["datasets"].items():
        ks = [p["ibd_psc"]["detector"]["k"] for p in r["per_model"].values()]
        reached = [p["ibd_psc"]["detector"]["xi_reached"] for p in r["per_model"].values()]
        nm = [p["ibd_psc"]["detector"]["n_models"] for p in r["per_model"].values()]
        cfgs = next(iter(r["per_model"].values()))["ibd_psc"]["detector"]["amplified_models"]
        L.append(f"- {d}: k per model {ks}; ξ exceeded in {sum(reached)}/{len(reached)} models; "
                 f"amplified models per detector {sorted(set(nm))}; first model's set "
                 + ", ".join(f"({c['n_layers']} layers, ω {c['omega']})" for c in cfgs) + ".")
    L += ["", "## Detection (mean over poisoned models, bootstrap 95% CI over models)", "",
          "| dataset | models | detector | ROC-AUC | TPR @ 5% FPR | realized FPR (negatives) | "
          "TPR @ T = 0.9 | FPR @ T = 0.9 (negatives) | per-input cost (ms) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for lbl, key in (("existing 5", "existing_n5"), ("new 5", "new_n5"), ("all 10", "all_n10")):
            s = r["summary"][key]
            i, t = s["ibd_psc"], s["strip"]
            L.append(f"| {d} | {lbl} | IBD-PSC | {_m(i['auc'])} | {_m(i['tpr_5fpr'])} | "
                     f"{i['fpr_5fpr_negatives']['mean']:.3f} | {_m(i['tpr_T'])} | "
                     f"{_m(i['fpr_T_negatives'])} | {i['per_input_ms']['mean']:.3f} |")
            L.append(f"| {d} | {lbl} | STRIP | {_m(t['auc'])} | {_m(t['tpr_5fpr'])} | "
                     f"{t['fpr_5fpr_negatives']['mean']:.3f} | – | – | {t['per_input_ms']['mean']:.3f} |")
    L += ["", "STRIP reproduction check on the existing models against `phase7f` (α 0.7): "
          + "; ".join(f"{d} max abs. difference {r['strip_fidelity']['max_abs_diff']:.1e}"
                      for d, r in out["datasets"].items()) + ".",
          "Holdout/negative row overlap (UNSW-NB15 draws both from the test split, as in phase7f): "
          + "; ".join(f"{d} {r['holdout_negative_overlap_rows']} rows" for d, r in out["datasets"].items())
          + ". Per-input cost is wall-clock time on the CPU, including all forward passes "
          "(IBD-PSC: amplified models + the original model; STRIP: N perturbations).", "",
          "## PSC distributions (first poisoned model per dataset)", "",
          "| dataset | group | n | mean | median | std | fraction ≥ T |", "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for g in ("clean_benign", "clean_attack", "triggered"):
            s = r["distribution"][g]
            L.append(f"| {d} | {g.replace('_', ' ')} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                     f"{s['std']:.4f} | {s['frac_ge_T']:.3f} |")
    L += ["", "ROC-AUC triggered vs clean benign/normal on these inputs: "
          + "; ".join(f"{d} {r['distribution']['roc_auc_triggered_vs_clean_benign']:.3f}"
                      for d, r in out["datasets"].items())
          + ". Figures: `psc_distributions.png`, `psc_roc.png`; raw scores: `psc_raw_<dataset>.csv`."]
    if "controlled" in out:
        L += ["", "## CICIDS2017 controlled conditions (phase6c/phase6d checkpoints, 5 models each)", "",
              "Same draws as phase6d: 3,000-row validation holdout, 2,000 random clean test rows as "
              "negatives, the trigger on the first 2,000 source-class test rows as positives.", "",
              "| condition | target | clean target-class STRIP entropy (phase6c) | clean target-class PSC | "
              "IBD-PSC ROC-AUC | IBD-PSC TPR @ 5% FPR | IBD-PSC TPR @ T | STRIP TPR @ 5% FPR (phase6d) | k per model |",
              "|---|---|---|---|---|---|---|---|---|"]
        for cname, c in out["controlled"].items():
            L.append(f"| {cname} | {'benign' if c['target'] == 0 else 'attack'} | "
                     f"{c['clean_target_entropy_phase6c']:.3f} | {_m(c['psc_clean_target_mean'])} | "
                     f"{_m(c['auc'])} | {_m(c['tpr_5fpr'])} | {_m(c['tpr_T'])} | "
                     f"{c['strip_tpr5_phase6d']['mean']:.3f} | {c['k_per_model']} |")
        L += ["", "Figure: `controlled_conditions.png`."]
    L += ["", f"Total runtime {out['runtime_s'] / 60:.1f} min.", "", "## Artifacts", "",
          "- `metrics.json`, `summary.md`, `manifest.json`",
          "- `psc_distributions.png`, `psc_roc.png`, `controlled_conditions.png`, `psc_raw_<dataset>.csv`"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="v1.1 Experiment C: IBD-PSC")
    ap.add_argument("--config", default="configs/v11_ibd_psc.yaml")
    ap.add_argument("--summary-only", action="store_true",
                    help="rewrite summary.md from the existing metrics.json")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = load_yaml(resolve_path(a.config))
    cfg["_config_path"] = a.config
    if a.summary_only:
        exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
        _summary(exp_dir / "summary.md", cfg,
                 json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8")))
        return
    run(cfg)


if __name__ == "__main__":
    main()
