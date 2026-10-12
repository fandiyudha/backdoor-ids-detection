"""Follow-up checks to v1.1 Experiment C (outputs ``experiments/v11_ibd_psc_checks/``).

1. SENSITIVITY: an IBD-PSC variant that reaches the error-rate condition xi by
   searching omega with all BN layers amplified (``select_omega_all_layers``),
   then uses five amplified models at omega* x {1, 1.25, 1.5, 1.75, 2}.
2. EXPLORATORY: a two-sided percentile rule (flag scores outside the 2.5th-97.5th
   percentile range of the clean holdout) for STRIP and both IBD-PSC variants.
3. UNSW-NB15 overlap: negative-dependent metrics recomputed after removing the
   negatives that also appear in the holdout (same per-input scores).

The models, holdouts, positives, and negatives are those of Experiment C, which
replays the draws of ``run_phase7f_strip_reeval``; STRIP and primary IBD-PSC
scores are recomputed and checked against Experiment C before anything else.

Run:
    python -u -m src.eval.run_v11_ibd_psc_checks --config configs/v11_ibd_psc_checks.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.attacks.trigger import apply_trigger, build_trigger, load_trigger  # noqa: E402
from src.detectors.ibd_psc import IBDPSC, bn_layers_deepest_first, select_omega_all_layers  # noqa: E402
from src.detectors.strip import StripDetector  # noqa: E402
from src.eval.v11_common import (  # noqa: E402
    CPU,
    environment,
    git_commit,
    load_model_cpu,
    mean_ci,
    model_names,
    utc_now,
    write_json,
)
from src.models.train import load_dataset  # noqa: E402
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


def phase7f_inputs(p7f, dname, pc):
    """Holdout, positives, and negatives exactly as phase7f (and Experiment C) draw them."""
    import joblib

    data = load_dataset(resolve_path(pc["processed_dir"]))
    feat = [str(f) for f in data["feature_names"]]
    if "trigger_json" in pc:
        trigger = load_trigger(resolve_path(pc["trigger_json"]))
    else:
        scaler = joblib.load(resolve_path(p7f["paths"]["scaler_unsw"]))
        trigger = build_trigger(feat, scaler, data["X_train"], pc["trigger_build"]["features"],
                                mode="constant", constants=pc["trigger_build"]["constants"],
                                enforce_controllable=False)
    rng = np.random.default_rng(p7f["seed"])
    hk, yk = ("X_val", "y_val") if "X_val" in data else ("X_test", "y_test")
    hidx = rng.choice(len(data[hk]), min(p7f["holdout_size"], len(data[hk])), replace=False)
    Xte, yte = data["X_test"], data["y_test"]
    es = p7f["eval_size"]
    pos = apply_trigger(Xte[np.where(yte == pc["source"])[0][:es]].astype(np.float32), trigger)
    nidx = rng.choice(len(Xte), es, replace=False)
    keep = ~np.isin(nidx, hidx) if hk == "X_test" else np.ones(len(nidx), bool)
    return {"holdout": data[hk][hidx].astype(np.float32), "holdout_y": data[yk][hidx].astype(int),
            "pos": pos, "neg": Xte[nidx].astype(np.float32), "dedup_keep": keep,
            "holdout_split": hk}


def score_metrics(s_hold, s_pos, s_neg, high_suspicious, cfg, T=None):
    """One-sided 5%-FPR rule, two-sided percentile rule, AUC, and (optionally) the fixed T."""
    f = 0.05
    sign = 1.0 if high_suspicious else -1.0
    auc = float(roc_auc_score(np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))],
                              sign * np.r_[s_pos, s_neg]))
    if high_suspicious:
        thr = float(np.quantile(s_hold, 1 - f))
        one_pos, one_neg = s_pos > thr, s_neg > thr
    else:
        thr = float(np.quantile(s_hold, f))
        one_pos, one_neg = s_pos < thr, s_neg < thr
    lo, hi = np.percentile(s_hold, [cfg["two_sided"]["lower_pct"], cfg["two_sided"]["upper_pct"]])
    two_pos = (s_pos < lo) | (s_pos > hi)
    two_neg = (s_neg < lo) | (s_neg > hi)
    out = {"auc": auc, "tpr_5fpr": float(one_pos.mean()), "fpr_5fpr_negatives": float(one_neg.mean()),
           "two_sided_tpr": float(two_pos.mean()), "two_sided_fpr_negatives": float(two_neg.mean()),
           "two_sided_thresholds": [float(lo), float(hi)],
           "two_sided_flags_low_pos": float((s_pos < lo).mean()),
           "two_sided_flags_high_pos": float((s_pos > hi).mean())}
    if T is not None:
        out["tpr_T"] = float(np.mean(s_pos >= T))
        out["fpr_T_negatives"] = float(np.mean(s_neg >= T))
    return out


def all_metrics(s_hold, s_pos, s_neg, keep, high, cfg, T=None):
    full = score_metrics(s_hold, s_pos, s_neg, high, cfg, T)
    dedup = score_metrics(s_hold, s_pos, s_neg[keep], high, cfg, T)
    return {"full": full, "dedup": dedup, "n_negatives_full": int(len(s_neg)),
            "n_negatives_dedup": int(keep.sum())}


def strip_scores(model, inp, p7f, ec):
    """Same call sequence as phase7f's ``_strip`` (calibrate on the holdout, then
    score positives, then negatives), so the scores are identical."""
    det = StripDetector(model, inp["holdout"], variant=ec["strip"]["variant"],
                        alpha=ec["strip"]["alpha"], N=ec["strip"]["N"], device=CPU,
                        seed=p7f["seed"], chunk=128)
    e_hold, _ = det.score(inp["holdout"])
    e_pos, _ = det.score(inp["pos"])
    e_neg, _ = det.score(inp["neg"])
    return e_hold, e_pos, e_neg


def xi_faithful_detector(model, X, y, cfg):
    xf = cfg["xi_faithful"]
    w_star, errs = select_omega_all_layers(model, X[:xf["n_select"]], y[:xf["n_select"]],
                                           xf["omega_grid"], xf["xi"])
    info = {"omega_star": w_star, "error_by_omega": errs, "xi_reached": w_star is not None}
    if w_star is None:
        return None, info
    L = len(bn_layers_deepest_first(model))
    configs = [(L, w_star * m) for m in xf["omega_multipliers"]]
    det = IBDPSC(model, X[:xf["n_select"]], y[:xf["n_select"]], fixed_configs=configs)
    info["amplified_models"] = [{"n_layers": m, "omega": w} for m, w in configs]
    return det, info


def ibd_primary_detector(model, X, y, ec):
    ic = ec["ibd_psc"]
    return IBDPSC(model, X[:ic["n_select"]], y[:ic["n_select"]], omega=ic["omega"], xi=ic["xi"],
                  n=ic["n"], extra_omegas=ic["extra_omegas"])


def aggregate(per, names, n_exist, iters, seed):
    out = {}
    for lbl, sel in (("existing_n5", names[:n_exist]), ("all_n10", names)):
        out[lbl] = {}
        for det in ("strip", "ibd_primary", "ibd_xi_faithful"):
            rows = [per[n][det] for n in sel if per[n].get(det)]
            if not rows:
                out[lbl][det] = None
                continue
            out[lbl][det] = {"n_models": len(rows)}
            for part in ("full", "dedup"):
                keys = [k for k, v in rows[0][part].items() if isinstance(v, float)]
                out[lbl][det][part] = {k: mean_ci([r[part][k] for r in rows], iters, seed) for k in keys}
    return out


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    t0 = time.perf_counter()
    ec = load_yaml(resolve_path(cfg["experiment_c_config"]))
    mcfg = load_yaml(resolve_path(ec["models_from"]))
    p7f = load_yaml(resolve_path(ec["protocol_from"]))
    exp_c = json.loads(resolve_path(cfg["experiment_c_metrics"]).read_text(encoding="utf-8"))
    md = resolve_path(cfg["paths"]["models_dir"])
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    iters, seed, T = cfg["bootstrap_iters"], cfg["seed"], ec["ibd_psc"]["T"]

    out = {"datasets": {}}
    dists = {}
    for dname, pc in p7f["datasets"].items():
        print(f"\n==================== {dname} ====================", flush=True)
        inp = phase7f_inputs(p7f, dname, pc)
        keep = inp["dedup_keep"]
        print(f"  negatives {len(keep)}, removed as holdout duplicates {int((~keep).sum())}", flush=True)
        names = model_names(mcfg["datasets"][dname])["poisoned"]
        per, repro = {}, []
        for n in names:
            m = load_model_cpu(md, n)
            e_hold, e_pos, e_neg = strip_scores(m, inp, p7f, ec)
            dp = ibd_primary_detector(m, inp["holdout"], inp["holdout_y"], ec)
            p_hold, p_pos, p_neg = dp.score(inp["holdout"]), dp.score(inp["pos"]), dp.score(inp["neg"])
            dx, xinfo = xi_faithful_detector(m, inp["holdout"], inp["holdout_y"], cfg)
            rec = {"strip": all_metrics(e_hold, e_pos, e_neg, keep, False, cfg),
                   "ibd_primary": all_metrics(p_hold, p_pos, p_neg, keep, True, cfg, T),
                   "xi_faithful_info": xinfo}
            if dx is not None:
                x_hold, x_pos, x_neg = dx.score(inp["holdout"]), dx.score(inp["pos"]), dx.score(inp["neg"])
                rec["ibd_xi_faithful"] = all_metrics(x_hold, x_pos, x_neg, keep, True, cfg, T)
                if n == names[0]:
                    nf = ec["distribution_size"]
                    data = load_dataset(resolve_path(pc["processed_dir"]))
                    Xte, yte = data["X_test"], data["y_test"]
                    trig = phase7f_trigger(p7f, pc, data)
                    groups = {"clean_benign": Xte[np.where(yte == 0)[0][:nf]].astype(np.float32),
                              "clean_attack": Xte[np.where(yte == 1)[0][:nf]].astype(np.float32),
                              "triggered": apply_trigger(Xte[np.where(yte == pc["source"])[0][:nf]], trig)}
                    dists[dname] = {g: dx.score(X) for g, X in groups.items()}
            per[n] = rec
            c = exp_c["datasets"][dname]["per_model"][n]
            repro.append(max(abs(rec["strip"]["full"]["auc"] - c["strip"]["auc"]),
                             abs(rec["strip"]["full"]["tpr_5fpr"] - c["strip"]["tpr_5fpr"]),
                             abs(rec["ibd_primary"]["full"]["auc"] - c["ibd_psc"]["auc"]),
                             abs(rec["ibd_primary"]["full"]["tpr_5fpr"] - c["ibd_psc"]["tpr_5fpr"])))
            xs = rec.get("ibd_xi_faithful", {}).get("full")
            print(f"  {n}: omega* {xinfo['omega_star']} | two-sided TPR/FPR STRIP "
                  f"{rec['strip']['full']['two_sided_tpr']:.3f}/{rec['strip']['full']['two_sided_fpr_negatives']:.3f} "
                  f"IBD {rec['ibd_primary']['full']['two_sided_tpr']:.3f}/"
                  f"{rec['ibd_primary']['full']['two_sided_fpr_negatives']:.3f}"
                  + (f" | xi-faithful AUC {xs['auc']:.3f} TPR@5 {xs['tpr_5fpr']:.3f} two-sided "
                     f"{xs['two_sided_tpr']:.3f}/{xs['two_sided_fpr_negatives']:.3f}" if xs else
                     " | xi never reached"), flush=True)
        n_exist = len(mcfg["datasets"][dname]["existing"]["poisoned"])
        out["datasets"][dname] = {
            "per_model": per, "summary": aggregate(per, names, n_exist, iters, seed),
            "reproduction_vs_experiment_c_max_abs_diff": float(max(repro)),
            "n_negatives_removed_as_holdout_duplicates": int((~keep).sum()),
            "holdout_split": inp["holdout_split"],
            "xi_faithful_omega_star": {n: per[n]["xi_faithful_info"]["omega_star"] for n in names}}
        if dname in dists:
            out["datasets"][dname]["xi_faithful_distribution"] = {
                g: {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
                    "std": float(a.std()), "frac_ge_T": float(np.mean(a >= T))}
                for g, a in dists[dname].items()}
        print(f"  reproduction vs Experiment C: max |diff| {max(repro):.1e}", flush=True)

    out["controlled_xi_faithful"] = run_controlled(cfg, ec, md, iters, seed, T)
    out["runtime_s"] = time.perf_counter() - t0
    if dists:
        _plot(dists, T, exp_dir / "psc_distributions_xi_faithful.png")
    write_json(exp_dir / "metrics.json", out)
    write_json(exp_dir / "manifest.json", {
        "experiment": exp_dir.name, "config": cfg.get("_config_path"),
        "command": f"python -u -m src.eval.run_v11_ibd_psc_checks --config {cfg.get('_config_path')}",
        "evaluated_at": utc_now(), "git_commit": git_commit(), "environment": environment(),
        "runtime_s": out["runtime_s"], "models": "as in experiments/v11_ibd_psc/manifest.json"})
    _summary(exp_dir / "summary.md", cfg, out)
    print(f"\n  wrote artifacts -> {exp_dir} ({out['runtime_s']:.0f}s)", flush=True)
    return out


def phase7f_trigger(p7f, pc, data):
    if "trigger_json" in pc:
        return load_trigger(resolve_path(pc["trigger_json"]))
    import joblib

    scaler = joblib.load(resolve_path(p7f["paths"]["scaler_unsw"]))
    return build_trigger([str(f) for f in data["feature_names"]], scaler, data["X_train"],
                         pc["trigger_build"]["features"], mode="constant",
                         constants=pc["trigger_build"]["constants"], enforce_controllable=False)


def run_controlled(cfg, ec, md, iters, seed, T):
    cc = ec["controlled"]
    data = load_dataset(resolve_path(cc["processed_dir"]))
    trigger = load_trigger(resolve_path(cc["trigger_json"]))
    Xte, yte = data["X_test"], data["y_test"]
    rng = np.random.default_rng(seed)        # replay phase6d's draws (as Experiment C)
    hidx = rng.choice(len(data["X_val"]), cc["holdout_size"], replace=False)
    holdout, holdout_y = data["X_val"][hidx].astype(np.float32), data["y_val"][hidx].astype(int)
    rng.choice(len(data["X_val"]), cc["nc_holdout_size"], replace=False)
    es = cc["eval_size"]
    neg = Xte[rng.choice(len(Xte), es, replace=False)].astype(np.float32)
    res = {}
    for cname, c in cc["conditions"].items():
        pos = apply_trigger(Xte[np.where(yte == c["source"])[0][:es]].astype(np.float32), trigger)
        clean_tgt = Xte[np.where(yte == c["target"])[0][:es]].astype(np.float32)
        rows, omegas = [], []
        for s in cc["seeds"]:
            m = load_model_cpu(md, f"causal_ids_{cname}_seed{s}")
            dx, info = xi_faithful_detector(m, holdout, holdout_y, cfg)
            omegas.append(info["omega_star"])
            if dx is None:
                continue
            r = score_metrics(dx.score(holdout), dx.score(pos), dx.score(neg), True, cfg, T)
            r["psc_clean_target_mean"] = float(dx.score(clean_tgt).mean())
            rows.append(r)
        res[cname] = {"target": c["target"], "omega_star_per_model": omegas,
                      "n_models_xi_reached": len(rows)}
        if rows:
            for k in ("auc", "tpr_5fpr", "tpr_T", "fpr_T_negatives", "two_sided_tpr",
                      "two_sided_fpr_negatives", "psc_clean_target_mean"):
                res[cname][k] = mean_ci([r[k] for r in rows], iters, seed)
        print(f"  controlled {cname}: omega* {omegas}"
              + (f" AUC {res[cname]['auc']['mean']:.3f} TPR@5 {res[cname]['tpr_5fpr']['mean']:.3f}"
                 if rows else ""), flush=True)
    return res


# ---------------------------------------------------------------------------
def _plot(dists, T, path):
    cols = {"clean_benign": ("#2b6cb0", "clean benign/normal"), "clean_attack": ("#2f855a", "clean attack"),
            "triggered": ("#c53030", "triggered")}
    fig, axes = plt.subplots(1, len(dists), figsize=(5.6 * len(dists), 4.0), squeeze=False)
    for ax, (d, dist) in zip(axes[0], dists.items()):
        bins = np.linspace(0, 1, 51)
        for g, (c, lbl) in cols.items():
            ax.hist(dist[g], bins=bins, density=True, histtype="step", lw=1.8, color=c,
                    label=f"{lbl} (n={len(dist[g])})")
        ax.axvline(T, color="#444", ls="--", lw=1, label=f"T = {T}")
        ax.set_yscale("log")
        ax.set_xlabel("PSC score, xi-faithful variant (higher = more suspicious)")
        ax.set_ylabel("density (log)")
        ax.set_title(f"{d} (sensitivity analysis)", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _m(a, nd=3):
    return "–" if a is None else f"{a['mean']:.{nd}f} [{a['boot_ci95'][0]:.{nd}f}, {a['boot_ci95'][1]:.{nd}f}]"


DET_LABEL = {"strip": "STRIP (α 0.7)", "ibd_primary": "IBD-PSC (Experiment C)",
             "ibd_xi_faithful": "IBD-PSC (ξ-faithful)"}


def _summary(path, cfg, out):
    xf, ts = cfg["xi_faithful"], cfg["two_sided"]
    L = ["# Follow-up checks to extended experiment C", "",
         f"- **Run:** `python -u -m src.eval.run_v11_ibd_psc_checks --config {cfg.get('_config_path')}`",
         "- **Scope:** three secondary analyses on the models, holdouts, positives, and negatives of "
         "Experiment C (which replays the phase7f draws). None replaces a result of Experiment C.",
         "- **Reproduction:** STRIP and primary IBD-PSC scores recomputed here match Experiment C: "
         + "; ".join(f"{d} max abs. difference {r['reproduction_vs_experiment_c_max_abs_diff']:.1e}"
                     for d, r in out["datasets"].items()) + ".", "",
         "## 1. Sensitivity analysis: ξ-faithful IBD-PSC", "",
         f"All BN layers amplified; ω increased over {xf['omega_grid']} until the error rate on the "
         f"first {xf['n_select']} clean holdout rows (against the true labels) exceeds ξ = {xf['xi']}; "
         f"ω* = the first such value. Five amplified models scale all BN layers by ω* × "
         f"{xf['omega_multipliers']}.", ""]
    for d, r in out["datasets"].items():
        om = list(r["xi_faithful_omega_star"].values())
        L.append(f"- {d}: ω* per model {om}; ξ reached in {sum(o is not None for o in om)}/{len(om)} models.")
    L += ["", "| dataset | models | detector | n | ROC-AUC | TPR @ 5% FPR | realized FPR | TPR @ T = 0.9 | "
          "FPR @ T = 0.9 |", "|---|---|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for lbl, key in (("existing 5", "existing_n5"), ("all 10", "all_n10")):
            for det in ("ibd_primary", "ibd_xi_faithful"):
                s = r["summary"][key][det]
                if s is None:
                    L.append(f"| {d} | {lbl} | {DET_LABEL[det]} | 0 | ξ not reached | – | – | – | – |")
                    continue
                f = s["full"]
                L.append(f"| {d} | {lbl} | {DET_LABEL[det]} | {s['n_models']} | {_m(f['auc'])} | "
                         f"{_m(f['tpr_5fpr'])} | {f['fpr_5fpr_negatives']['mean']:.3f} | "
                         f"{_m(f['tpr_T'])} | {_m(f['fpr_T_negatives'])} |")
    for d, r in out["datasets"].items():
        if "xi_faithful_distribution" in r:
            dd = r["xi_faithful_distribution"]
            L.append("")
            L.append(f"PSC distribution, ξ-faithful variant, first poisoned model, {d}: "
                     + "; ".join(f"{g.replace('_', ' ')} mean {s['mean']:.3f} (median {s['median']:.3f}, "
                                 f"fraction ≥ T {s['frac_ge_T']:.3f}, n = {s['n']})" for g, s in dd.items())
                     + ". Figure: `psc_distributions_xi_faithful.png`.")
    cx = out["controlled_xi_faithful"]
    L += ["", "CICIDS2017 controlled conditions, ξ-faithful variant (5 models each):", "",
          "| condition | target | ω* per model | ROC-AUC | TPR @ 5% FPR | TPR @ T | FPR @ T | clean target-class PSC |",
          "|---|---|---|---|---|---|---|---|"]
    for cname, c in cx.items():
        if c["n_models_xi_reached"] == 0:
            L.append(f"| {cname} | {'benign' if c['target'] == 0 else 'attack'} | {c['omega_star_per_model']} | "
                     "ξ not reached | – | – | – | – |")
            continue
        L.append(f"| {cname} | {'benign' if c['target'] == 0 else 'attack'} | {c['omega_star_per_model']} | "
                 f"{_m(c['auc'])} | {_m(c['tpr_5fpr'])} | {_m(c['tpr_T'])} | {_m(c['fpr_T_negatives'])} | "
                 f"{_m(c['psc_clean_target_mean'])} |")
    L += ["", "## 2. Exploratory: two-sided percentile rule", "",
          f"Flag an input if its score lies outside the [{ts['lower_pct']}th, {ts['upper_pct']}th] "
          "percentile range of the clean-holdout scores (nominal FPR 5%). Mean over models with a "
          "bootstrap 95% CI.", "",
          "| dataset | detector | n | TPR (two-sided) | of which below / above the range | realized FPR (two-sided) |",
          "|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for det in ("strip", "ibd_primary", "ibd_xi_faithful"):
            s = r["summary"]["all_n10"][det]
            if s is None:
                L.append(f"| {d} | {DET_LABEL[det]} | 0 | ξ not reached | – | – |")
                continue
            f = s["full"]
            L.append(f"| {d} | {DET_LABEL[det]} | {s['n_models']} | {_m(f['two_sided_tpr'])} | "
                     f"{f['two_sided_flags_low_pos']['mean']:.3f} / {f['two_sided_flags_high_pos']['mean']:.3f} | "
                     f"{_m(f['two_sided_fpr_negatives'])} |")
    L += ["", "## 3. UNSW-NB15 holdout/negative overlap", ""]
    for d, r in out["datasets"].items():
        L.append(f"- {d}: {r['n_negatives_removed_as_holdout_duplicates']} of the negatives also appear in "
                 f"the holdout (holdout drawn from `{r['holdout_split']}`).")
    L += ["", "Per-input scores and holdout thresholds are unchanged; only the negative set shrinks. "
          "Positive-only metrics (TPR) are therefore identical.", "",
          "| dataset | models | detector | metric | with overlap | de-duplicated |", "|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        if r["n_negatives_removed_as_holdout_duplicates"] == 0:
            continue
        for lbl, key in (("existing 5", "existing_n5"), ("all 10", "all_n10")):
            for det in ("strip", "ibd_primary", "ibd_xi_faithful"):
                s = r["summary"][key][det]
                if s is None:
                    continue
                for met, ml in (("auc", "ROC-AUC"), ("fpr_5fpr_negatives", "FPR at the 5% threshold"),
                                ("two_sided_fpr_negatives", "FPR, two-sided rule"),
                                ("fpr_T_negatives", "FPR at T = 0.9")):
                    if met not in s["full"]:
                        continue
                    L.append(f"| {d} | {lbl} | {DET_LABEL[det]} | {ml} | {s['full'][met]['mean']:.4f} | "
                             f"{s['dedup'][met]['mean']:.4f} |")
    L += ["", f"Total runtime {out['runtime_s'] / 60:.1f} min.", "", "## Artifacts", "",
          "- `metrics.json`, `summary.md`, `manifest.json`, `psc_distributions_xi_faithful.png`"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Follow-up checks to v1.1 Experiment C")
    ap.add_argument("--config", default="configs/v11_ibd_psc_checks.yaml")
    ap.add_argument("--summary-only", action="store_true")
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
