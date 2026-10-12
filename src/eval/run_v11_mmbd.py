"""v1.1 Experiment B: MM-BD maximum-margin detection on 10 clean + 10 poisoned
models per dataset (outputs ``experiments/v11_mmbd/``; also used for the 1D-CNN).

For every model and class ``c`` the MM-BD statistic ``r_c`` (maximum logit margin
of class ``c`` over a box-constrained input domain) is computed with
``src.detectors.mmbd.MaxMarginDetector``. The box comes from the defender's clean
holdout (the phase7e holdout): per-feature min/max (primary) and the 1st-99th
percentile range (pre-specified sensitivity analysis).

Binary decision rule. MM-BD's original rule fits a null distribution to the K-1
non-maximal statistics and uses an order-statistic p-value; with K = 2 a single
non-maximal statistic remains, so that rule is not identifiable. The paper's
paired reference-relative rule is used instead: ``rel_c = r_c / median of r_c
over the calibration clean models``, statistic ``S = max_c rel_c``, flag if
``S > 2.0`` (primary) or if ``S`` exceeds the largest leave-one-out ``S`` among the
calibration models (secondary). Calibration and test clean models are disjoint.

Reports ROC-AUC with a stratified bootstrap 95% CI (primary metric), PR-AUC,
Mann-Whitney U with effect sizes, flag rate and false-positive rate (disjoint
test group and leave-one-out), target-class identification, the raw per-class
``r_c`` distributions, runtime per model, a convergence check, and the per-model
agreement with the Neural Cleanse verdicts of Experiment A (or D).

Run:
    python -u -m src.eval.run_v11_mmbd --config configs/v11_mmbd.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from src.detectors.mmbd import MaxMarginDetector, box_bounds  # noqa: E402
from src.eval.v11_common import (  # noqa: E402
    auc_boot,
    environment,
    git_commit,
    load_model_cpu,
    mann_whitney,
    mean_ci,
    model_names,
    models_manifest_entry,
    phase7e_holdouts,
    rate_ci,
    utc_now,
    write_json,
)
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


def score_model(model, lo, hi, mc, seed):
    return MaxMarginDetector(model, lo, hi, n_starts=mc["n_starts"], steps=mc["steps"],
                             lr=mc["lr"], seed=seed).score()


def analyse(clean_m, pois_m, calib_n, factor, target, iters, seed):
    """Reference-relative MM-BD statistics for one dataset and one box."""
    C = np.array([[m["0"], m["1"]] for m in clean_m], float)
    P = np.array([[m["0"], m["1"]] for m in pois_m], float)
    k = calib_n
    med_cal = np.median(C[:k], axis=0)
    if (med_cal <= 0).any() or (C <= 0).any() or (P <= 0).any():
        raise ValueError("non-positive margins: the ratio normalization is undefined")

    def stat(row, med):
        rel = row / med
        return float(rel.max()), int(rel.argmax())

    cal = [stat(C[i], np.median(np.delete(C[:k], i, axis=0), axis=0)) for i in range(k)]
    test = [stat(C[i], med_cal) for i in range(k, len(C))]
    pois = [stat(P[i], med_cal) for i in range(len(P))]
    S_clean = [s for s, _ in cal + test]
    S_test = np.array([s for s, _ in test])
    S_pois = np.array([s for s, _ in pois])
    thr2 = max(s for s, _ in cal)
    loo_clean = [stat(C[i], np.median(np.delete(C, i, axis=0), axis=0))[0] > factor
                 for i in range(len(C))]
    loo_pois = [stat(P[i], np.median(C, axis=0))[0] > factor for i in range(len(P))]
    raw = {}
    for c in (0, 1):
        raw[str(c)] = {"clean": mean_ci(C[:, c], iters, seed), "poisoned": mean_ci(P[:, c], iters, seed),
                       "auc_poisoned_gt_clean": auc_boot(C[:, c], P[:, c], iters, seed),
                       "mann_whitney": mann_whitney(C[:, c], P[:, c])}
    return {
        "calibration_median_margin": {"0": float(med_cal[0]), "1": float(med_cal[1])},
        "S_clean": S_clean, "S_poisoned": S_pois.tolist(),
        "S_calibration_loo": [s for s, _ in cal],
        "auc": auc_boot(S_clean, S_pois, iters, seed),
        "mann_whitney": mann_whitney(S_clean, S_pois),
        "primary_rule": {"rule": f"flag if S > {factor}", "threshold": factor,
                         "tpr": rate_ci(S_pois > factor, iters, seed + 1),
                         "fpr_test_group": rate_ci(S_test > factor, iters, seed),
                         "fpr_calibration_loo": rate_ci([s > factor for s, _ in cal], iters, seed),
                         "fpr_leave_one_out": rate_ci(loo_clean, iters, seed + 2),
                         "tpr_leave_one_out": rate_ci(loo_pois, iters, seed + 3),
                         "flags_clean": [bool(s > factor) for s in S_clean],
                         "flags_poisoned": [bool(s > factor) for s in S_pois]},
        "secondary_rule": {"rule": "flag if S > max leave-one-out S of the calibration models",
                           "threshold": float(thr2),
                           "tpr": rate_ci(S_pois > thr2, iters, seed + 1),
                           "fpr_test_group": rate_ci(S_test > thr2, iters, seed)},
        "target_identified_rate": float(np.mean([a == target for _, a in pois])),
        "target_identified_raw_argmax_rate": float(np.mean(P.argmax(axis=1) == target)),
        "predicted_target_poisoned": [a for _, a in pois],
        "raw_margins": raw,
    }


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    t_start = time.perf_counter()
    mcfg = load_yaml(resolve_path(cfg["models_from"]))
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    md = resolve_path(cfg["paths"]["models_dir"])
    iters, seed, mc = cfg["bootstrap_iters"], cfg["seed"], cfg["mmbd"]
    nc_res = json.loads(resolve_path(cfg["nc_results"]).read_text(encoding="utf-8"))
    holdouts = phase7e_holdouts(mcfg["holdout_replay_config"])

    out, bounds_out = {"datasets": {}}, {}
    for dname, dc in mcfg["datasets"].items():
        print(f"\n==================== {dname} ====================", flush=True)
        names = model_names(dc)
        target = dc["poison"]["target"]
        h = holdouts[dname]
        dres = {"boxes": {}, "models": names}
        bounds_out[dname] = {}
        for bname, bc in cfg["boxes"].items():
            lo, hi = box_bounds(h["X"], bc["kind"], bc.get("lo_pct", 1.0), bc.get("hi_pct", 99.0))
            bounds_out[dname][bname] = {"kind": bc["kind"], **{k: v for k, v in bc.items() if k != "kind"},
                                        "lo": lo.tolist(), "hi": hi.tolist(),
                                        "holdout": f"phase7e holdout ({h['source_split']}, "
                                                   f"{len(h['X'])} rows)"}
            per = {}
            for role in ("clean", "poisoned"):
                for n in names[role]:
                    r = score_model(load_model_cpu(md, n), lo, hi, mc, seed)
                    per[n] = {"role": role, "margins": r["margins"], "convergence": r["convergence"],
                              "time_s": r["time_s"]}
                    print(f"  [{bname}] {role:8s} {n}: r0 {r['margins']['0']:.2f} r1 "
                          f"{r['margins']['1']:.2f} ({r['time_s']:.1f}s)", flush=True)
            a = analyse([per[n]["margins"] for n in names["clean"]],
                        [per[n]["margins"] for n in names["poisoned"]], cfg["calib_n"],
                        cfg["rel_threshold"], target, iters, seed)
            a["per_model"] = per
            a["time_per_model_s"] = mean_ci([p["time_s"] for p in per.values()], iters, seed)
            a["box_width"] = {"median": float(np.median(hi - lo)), "max": float(np.max(hi - lo))}
            pr = a["primary_rule"]
            print(f"  [{bname}] AUC {a['auc']['roc_auc']:.3f} {a['auc']['roc_ci95']} | TPR "
                  f"{pr['tpr']['rate']:.2f} FPR_test {pr['fpr_test_group']['rate']:.2f} FPR_loo "
                  f"{pr['fpr_leave_one_out']['rate']:.2f} | target id "
                  f"{a['target_identified_rate']:.2f}", flush=True)
            dres["boxes"][bname] = a

        # convergence check (pre-specified): 2x starts and steps on two models
        cc = mc["convergence_check"]
        lo, hi = (np.asarray(bounds_out[dname][cfg["primary_box"]][k], np.float32) for k in ("lo", "hi"))
        conv = {}
        for n in (names["clean"][0], names["poisoned"][0]):
            big = MaxMarginDetector(load_model_cpu(md, n), lo, hi, n_starts=cc["n_starts"],
                                    steps=cc["steps"], lr=mc["lr"], seed=seed).score()
            base = dres["boxes"][cfg["primary_box"]]["per_model"][n]["margins"]
            conv[n] = {c: {"base": base[c], "2x": big["margins"][c],
                           "rel_change": (big["margins"][c] - base[c]) / abs(base[c])}
                       for c in base}
            print(f"  convergence {n}: " + ", ".join(
                f"r{c} {v['base']:.3f}->{v['2x']:.3f} ({v['rel_change']:+.2%})" for c, v in conv[n].items()),
                flush=True)
        dres["convergence_check"] = conv

        # agreement with Neural Cleanse (Experiment A / D verdicts)
        nc = nc_res["datasets"][dname]
        nc_thr = nc["n10"]["rule"]["threshold"]
        prim = dres["boxes"][cfg["primary_box"]]["primary_rule"]
        rows = []
        k = cfg["calib_n"]
        for role in ("clean", "poisoned"):
            mm_flags = prim[f"flags_{role}"]
            for i, n in enumerate(names[role]):
                group = ("calibration" if i < k else "test") if role == "clean" else "poisoned"
                rows.append({"model": n, "role": role, "group": group,
                             "nc_ratio": nc["per_model"][n]["ratio"],
                             "nc_flag": bool(nc["per_model"][n]["ratio"] > nc_thr),
                             "mmbd_S": dres["boxes"][cfg["primary_box"]][
                                 "S_clean" if role == "clean" else "S_poisoned"][i],
                             "mmbd_flag": bool(mm_flags[i])})
        ncf = np.array([r["nc_flag"] for r in rows])
        mmf = np.array([r["mmbd_flag"] for r in rows])
        pois = np.array([r["role"] == "poisoned" for r in rows])
        testc = np.array([r["group"] == "test" for r in rows])
        dres["agreement_with_nc"] = {
            "per_model": rows, "agreement_rate": float(np.mean(ncf == mmf)),
            "cohen_kappa": float(cohen_kappa_score(ncf, mmf)) if len(set(ncf) | set(mmf)) > 1 else None,
            "either_flags": {"tpr": rate_ci((ncf | mmf)[pois], iters, seed + 1),
                             "fpr_test_group": rate_ci((ncf | mmf)[testc], iters, seed)},
            "both_flag": {"tpr": rate_ci((ncf & mmf)[pois], iters, seed + 1),
                          "fpr_test_group": rate_ci((ncf & mmf)[testc], iters, seed)},
            "nc_threshold": nc_thr}
        out["datasets"][dname] = dres

    out["runtime_s"] = time.perf_counter() - t_start
    write_json(exp_dir / "box_bounds.json", bounds_out)
    _plot_raw(out, cfg, exp_dir / "mmbd_raw_margins.png")
    _plot_stat(out, cfg, exp_dir / "mmbd_statistic.png")
    write_json(exp_dir / "metrics.json", out)
    _manifest(exp_dir, cfg, mcfg, out, md)
    _summary(exp_dir / "summary.md", cfg, out)
    print(f"\n  wrote artifacts -> {exp_dir} ({out['runtime_s']:.0f}s)", flush=True)
    return out


def _manifest(exp_dir, cfg, mcfg, out, md):
    models = []
    for dname, d in out["datasets"].items():
        dc = mcfg["datasets"][dname]
        seeds = list(dc.get("existing", {}).get("seeds", [])) + list(dc["new"]["seeds"])
        for role in ("clean", "poisoned"):
            for n, s in zip(d["models"][role], seeds):
                models.append(models_manifest_entry(n, md, dataset=dname, role=role, seed=s))
    write_json(exp_dir / "manifest.json", {
        "experiment": exp_dir.name, "config": cfg.get("_config_path"),
        "command": f"python -u -m src.eval.run_v11_mmbd --config {cfg.get('_config_path')}",
        "evaluated_at": utc_now(), "git_commit": git_commit(), "environment": environment(),
        "runtime_s": out["runtime_s"], "models": models})


# ---------------------------------------------------------------------------
def _plot_raw(out, cfg, path):
    ds, boxes = list(out["datasets"]), list(cfg["boxes"])
    fig, axes = plt.subplots(len(boxes), len(ds), figsize=(5.2 * len(ds), 3.8 * len(boxes)),
                             squeeze=False)
    for i, b in enumerate(boxes):
        for j, d in enumerate(ds):
            ax = axes[i][j]
            a = out["datasets"][d]["boxes"][b]
            rng = np.random.default_rng(0)
            for c in ("0", "1"):
                for role, col, dx in (("clean", "#2b6cb0", -0.15), ("poisoned", "#c53030", 0.15)):
                    v = a["raw_margins"][c][role]["values"]
                    ax.scatter(int(c) + dx + rng.uniform(-0.06, 0.06, len(v)), v, s=22, color=col,
                               label=role if c == "0" else None)
            ax.set_yscale("log")
            ax.set_xticks([0, 1], labels=["class 0 (benign/normal)", "class 1 (attack)"])
            ax.set_ylabel("max margin r_c")
            ax.set_title(f"{d}, box {b}", fontsize=9)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stat(out, cfg, path):
    ds, boxes = list(out["datasets"]), list(cfg["boxes"])
    fig, axes = plt.subplots(len(boxes), len(ds), figsize=(5.2 * len(ds), 3.8 * len(boxes)),
                             squeeze=False)
    for i, b in enumerate(boxes):
        for j, d in enumerate(ds):
            ax = axes[i][j]
            a = out["datasets"][d]["boxes"][b]
            rng = np.random.default_rng(0)
            for x, key, col in ((0, "S_clean", "#2b6cb0"), (1, "S_poisoned", "#c53030")):
                v = a[key]
                ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=26, color=col)
            ax.axhline(cfg["rel_threshold"], ls="--", color="#444", lw=1, label="primary threshold")
            ax.axhline(a["secondary_rule"]["threshold"], ls=":", color="#444", lw=1,
                       label="secondary threshold")
            ax.set_xticks([0, 1], labels=["clean", "poisoned"])
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylabel("S = max_c r_c / calibration median")
            au = a["auc"]
            ax.set_title(f"{d}, box {b}: ROC-AUC {au['roc_auc']:.3f} "
                         f"[{au['roc_ci95'][0]:.2f}, {au['roc_ci95'][1]:.2f}]", fontsize=9)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ci(c, nd=2):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def _rate(r):
    return f"{r['k']}/{r['n']} = {r['rate']:.2f} (exact {_ci(r['clopper_pearson95'])})"


def _summary(path, cfg, out):
    mc = cfg["mmbd"]
    L = ["# Extended experiment B: MM-BD maximum-margin detection", "",
         f"- **Run:** `python -u -m src.eval.run_v11_mmbd --config {cfg.get('_config_path')}`",
         "- **Reference:** H. Wang, Z. Xiang, D. J. Miller, G. Kesidis, \"MM-BD: Post-training "
         "detection of backdoor attacks with arbitrary backdoor pattern types using a maximum "
         "margin statistic,\" IEEE S&P 2024, doi:10.1109/SP54263.2024.00015.",
         f"- **Models:** the 10 clean and 10 poisoned models per dataset listed in "
         f"`{cfg['models_from']}`.",
         "- **Statistic:** for each class c, r_c = max over x in X of [g_c(x) − max_(k≠c) g_k(x)] "
         "on the pre-softmax logits g, by projected gradient ascent (Adam in normalized box "
         f"coordinates, projection onto the box after every step), {mc['n_starts']} random "
         f"starts × {mc['steps']} steps, best value over starts and steps kept.",
         "- **Input domain X:** a per-feature box in standardized feature space from the "
         "defender's clean holdout (the phase7e holdout); primary: min/max; pre-specified "
         "sensitivity analysis: 1st–99th percentile. No training data is used. Bounds: "
         "`box_bounds.json`.", "",
         "## Binary adaptation of the decision rule", "",
         "MM-BD's original rule fits a null distribution (e.g., Gamma) to the K − 1 "
         "non-maximal statistics and computes an order-statistic p-value for the largest. With "
         "K = 2 classes only one non-maximal statistic remains, so the null distribution cannot "
         "be fitted and the rule is not identifiable. The paper's paired reference-relative "
         "rule is used instead:", "",
         "- rel_c = r_c / median of r_c over the calibration clean models (same holdout, same box);",
         "- statistic S = max_c rel_c; the predicted target class is argmax_c rel_c;",
         f"- primary rule: flag if S > {cfg['rel_threshold']} (the factor of the paper's Neural "
         "Cleanse rule); secondary rule: flag if S exceeds the largest S among the calibration "
         "models, each normalized by the median of the other calibration models;",
         f"- calibration group: the first {cfg['calib_n']} clean models (existing seeds); test "
         "group: the remaining clean models (disjoint). For the AUC, the calibration models use "
         "their leave-one-out normalization. Leave-one-out: each clean model against the median "
         "of the other clean models; each poisoned model against the median of all clean models.",
         "", "## Separation (primary metric: ROC-AUC)", "",
         "| dataset | box | ROC-AUC [95% CI] | PR-AUC [95% CI] | Mann-Whitney U | p (one-sided) | "
         "rank-biserial r |", "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for b, a in r["boxes"].items():
            au, m = a["auc"], a["mann_whitney"]
            L.append(f"| {d} | {b}{' (primary)' if b == cfg['primary_box'] else ''} | "
                     f"{au['roc_auc']:.3f} {_ci(au['roc_ci95'])} | {au['pr_auc']:.3f} "
                     f"{_ci(au['pr_ci95'])} | {m['U']:.0f} | {m['p_one_sided_greater']:.2g} | "
                     f"{m['rank_biserial']:.2f} |")
    L += ["", "## Flag rate and false-positive rate", "",
          "| dataset | box | rule | threshold | TPR | FPR, test group | FPR, leave-one-out | "
          "TPR, leave-one-out |", "|---|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for b, a in r["boxes"].items():
            p, s = a["primary_rule"], a["secondary_rule"]
            L.append(f"| {d} | {b} | primary | {p['threshold']:.2f} | {_rate(p['tpr'])} | "
                     f"{_rate(p['fpr_test_group'])} | {_rate(p['fpr_leave_one_out'])} | "
                     f"{_rate(p['tpr_leave_one_out'])} |")
            L.append(f"| {d} | {b} | secondary | {s['threshold']:.2f} | {_rate(s['tpr'])} | "
                     f"{_rate(s['fpr_test_group'])} | – | – |")
    L += ["", "## Target-class identification and raw margins", "",
          "| dataset | box | target identified (argmax rel_c) | target identified (argmax raw r_c) | "
          "r_0 clean | r_0 poisoned | AUC r_0 | r_1 clean | r_1 poisoned | AUC r_1 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for b, a in r["boxes"].items():
            rm = a["raw_margins"]
            L.append(f"| {d} | {b} | {a['target_identified_rate']:.2f} | "
                     f"{a['target_identified_raw_argmax_rate']:.2f} | "
                     f"{rm['0']['clean']['mean']:.3g} | {rm['0']['poisoned']['mean']:.3g} | "
                     f"{rm['0']['auc_poisoned_gt_clean']['roc_auc']:.2f} | "
                     f"{rm['1']['clean']['mean']:.3g} | {rm['1']['poisoned']['mean']:.3g} | "
                     f"{rm['1']['auc_poisoned_gt_clean']['roc_auc']:.2f} |")
    L += ["", "The target class is class 0 (benign/normal) in every poisoned model. 'AUC r_c' is "
          "the ROC-AUC of the raw margin of class c for poisoned versus clean models (> 0.5: "
          "larger in poisoned models). Raw per-model values: `metrics.json`; plot: "
          "`mmbd_raw_margins.png`.", "",
          "## Agreement with Neural Cleanse (primary box, primary rule)", "",
          "| dataset | agreement | Cohen's κ | either flags: TPR / FPR (test group) | both flag: "
          "TPR / FPR (test group) |", "|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        g = r["agreement_with_nc"]
        k = "–" if g["cohen_kappa"] is None else f"{g['cohen_kappa']:.2f}"
        L.append(f"| {d} | {g['agreement_rate']:.2f} | {k} | {_rate(g['either_flags']['tpr'])} / "
                 f"{_rate(g['either_flags']['fpr_test_group'])} | {_rate(g['both_flag']['tpr'])} / "
                 f"{_rate(g['both_flag']['fpr_test_group'])} |")
    L += ["", "| dataset | model | group | NC ratio | NC flag | MM-BD S | MM-BD flag |",
          "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for row in r["agreement_with_nc"]["per_model"]:
            L.append(f"| {d} | `{row['model']}` | {row['group']} | {row['nc_ratio']:.2f} | "
                     f"{'yes' if row['nc_flag'] else 'no'} | {row['mmbd_S']:.2f} | "
                     f"{'yes' if row['mmbd_flag'] else 'no'} |")
    L += ["", "## Optimization checks", ""]
    for d, r in out["datasets"].items():
        for n, cc in r["convergence_check"].items():
            L.append(f"- {d} `{n}`, {cfg['mmbd']['convergence_check']['n_starts']} starts × "
                     f"{cfg['mmbd']['convergence_check']['steps']} steps vs the main setting: "
                     + "; ".join(f"r_{c} {v['base']:.4g} → {v['2x']:.4g} ({v['rel_change']:+.2%})"
                                 for c, v in cc.items()))
        for b, a in r["boxes"].items():
            fr = [min(p["convergence"].values()) for p in a["per_model"].values()]
            L.append(f"- {d}, box {b}: median box width {a['box_width']['median']:.3g}, max "
                     f"{a['box_width']['max']:.3g} (standardized units); fraction of starts within "
                     f"1% of the best margin, lowest over classes, median over models "
                     f"{np.median(fr):.2f}; time per model {a['time_per_model_s']['mean']:.1f} s.")
    L += ["", f"Total runtime {out['runtime_s'] / 60:.1f} min.", "", "## Artifacts", "",
          "- `metrics.json`, `summary.md`, `box_bounds.json`, `manifest.json`",
          "- `mmbd_raw_margins.png` (raw r_c per class, clean vs poisoned), "
          "`mmbd_statistic.png` (S with both thresholds)"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="v1.1 Experiment B: MM-BD")
    ap.add_argument("--config", default="configs/v11_mmbd.yaml")
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
