"""Consolidated tables of the v1.1 extended experiments (outputs ``experiments/v11_summary/``).

Reads only committed metrics files (no models, no data) and writes:

1. one detection table, rows = dataset x architecture, columns = Neural Cleanse,
   MM-BD (min/max box), MM-BD (1st-99th percentile box), STRIP, IBD-PSC; cells =
   ROC-AUC [95% CI] and the operating point under each pre-specified rule;
2. the target-class "contrast" table: natural (clean-model) target-class NC mask
   norm and MM-BD margin next to the poisoned-model values;
3. headline numbers of v1.0 (frozen files) next to the v1.1 values;
4. the class properties behind findings (d)-(f) for both architectures;
5. compute time and the number of new checkpoints.

Run:
    python -m src.eval.v11_report
"""

from __future__ import annotations

import json

import numpy as np

from src.eval.v11_common import write_json
from src.utils import resolve_path


def _j(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def _ci(c, nd=2):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def _r(r):
    return f"{r['k']}/{r['n']}"


def _mci(a, nd=3):
    return f"{a['mean']:.{nd}f} {_ci(a['boot_ci95'], nd)}"


DS = ("CICIDS2017", "UNSW-NB15")
ARCH = (("mlp", "MLP"), ("cnn1d", "1D-CNN"))


def load_all():
    return {
        "A": _j("experiments/v11_screening/metrics.json"),
        "B": _j("experiments/v11_mmbd/metrics.json"),
        "C": _j("experiments/v11_ibd_psc/metrics.json"),
        "Cchk": _j("experiments/v11_ibd_psc_checks/metrics.json"),
        "D": _j("experiments/v11_cnn/metrics.json"),
        "Dnc": _j("experiments/v11_cnn/nc/metrics.json"),
        "Dmm": _j("experiments/v11_cnn/mmbd/metrics.json"),
        "Drt": _j("experiments/v11_cnn/runtime/metrics.json"),
        "diag": _j("experiments/v11_nc_diagnostic/metrics.json"),
        "p7e": _j("experiments/phase7e/metrics.json"),
        "p7f": _j("experiments/phase7f/metrics.json"),
        "p7d": _j("experiments/phase7d/metrics.json"),
        "p7": _j("experiments/phase7/comparison_metrics.json"),
        "fig3": _j("experiments/figure3/figure3_data.json"),
        "manA": _j("experiments/v11_screening/models_manifest.json"),
        "manD": _j("experiments/v11_cnn/models_manifest.json"),
        "manDb": _j("experiments/v11_cnn/models_manifest_full_baseline.json"),
    }


def sources(S, d, arch):
    if arch == "mlp":
        nc = S["A"]["datasets"][d]
        return {"nc_rule": nc["n10"]["rule"], "nc_auc": nc["n10"]["auc"], "nc_per_model": nc["per_model"],
                "nc_target_id": nc["n10"]["target_is_smallest_mask_rate"],
                "mm": S["B"]["datasets"][d], "rt": S["C"]["datasets"][d]["summary"]["all_n10"],
                "two_sided": S["Cchk"]["datasets"][d]["summary"]["all_n10"]}
    nc = S["Dnc"]["datasets"][d]
    return {"nc_rule": nc["n10"]["rule"], "nc_auc": nc["n10"]["auc"], "nc_per_model": nc["per_model"],
            "nc_target_id": nc["n10"]["target_is_smallest_mask_rate"],
            "mm": S["Dmm"]["datasets"][d], "rt": S["Drt"]["datasets"][d]["summary"]["all_n10"],
            "two_sided": None}


def detection_rows(S):
    rows = []
    for d in DS:
        for arch, lbl in ARCH:
            s = sources(S, d, arch)
            nr = s["nc_rule"]
            row = {"dataset": d, "arch": lbl,
                   "nc": {"auc": s["nc_auc"]["roc_auc"], "ci": s["nc_auc"]["roc_ci95"],
                          "tpr": _r(nr["tpr"]), "fpr_test": _r(nr["fpr_test_group"]),
                          "fpr_loo": _r(nr["fpr_leave_one_out"]), "target_id": s["nc_target_id"]}}
            for box in ("minmax", "p1_p99"):
                a = s["mm"]["boxes"][f"{box}/seeded"]
                pre = s["mm"]["boxes"][f"{box}/random_only"]
                row[f"mmbd_{box}"] = {
                    "auc": a["auc"]["roc_auc"], "ci": a["auc"]["roc_ci95"],
                    "primary_tpr": _r(a["primary_rule"]["tpr"]), "primary_fpr": _r(a["primary_rule"]["fpr_test_group"]),
                    "primary_fpr_loo": _r(a["primary_rule"]["fpr_leave_one_out"]),
                    "secondary_thr": a["secondary_rule"]["threshold"],
                    "secondary_tpr": _r(a["secondary_rule"]["tpr"]),
                    "secondary_fpr": _r(a["secondary_rule"]["fpr_test_group"]),
                    "target_id": a["target_identified_rate"],
                    "prespecified_random_only_auc": None if "undefined" in pre else pre["auc"]["roc_auc"]}
            st, ib = s["rt"]["strip"], s["rt"]["ibd_psc"]
            row["strip"] = {"auc": st["auc"]["mean"], "ci": st["auc"]["boot_ci95"],
                            "tpr5": st["tpr_5fpr"]["mean"], "tpr5_ci": st["tpr_5fpr"]["boot_ci95"],
                            "fpr5": st["fpr_5fpr_negatives"]["mean"]}
            row["ibd"] = {"auc": ib["auc"]["mean"], "ci": ib["auc"]["boot_ci95"],
                          "tpr5": ib["tpr_5fpr"]["mean"], "fpr5": ib["fpr_5fpr_negatives"]["mean"],
                          "tprT": ib["tpr_T"]["mean"], "fprT": ib["fpr_T_negatives"]["mean"]}
            if s["two_sided"]:
                row["strip"]["two_sided_tpr"] = s["two_sided"]["strip"]["full"]["two_sided_tpr"]["mean"]
                row["strip"]["two_sided_fpr"] = s["two_sided"]["strip"]["full"]["two_sided_fpr_negatives"]["mean"]
                row["ibd"]["two_sided_tpr"] = s["two_sided"]["ibd_primary"]["full"]["two_sided_tpr"]["mean"]
                row["ibd"]["two_sided_fpr"] = s["two_sided"]["ibd_primary"]["full"]["two_sided_fpr_negatives"]["mean"]
            rows.append(row)
    return rows


def contrast_rows(S):
    rows = []
    for d in DS:
        for arch, lbl in ARCH:
            s = sources(S, d, arch)
            pm = s["nc_per_model"]
            mm = s["mm"]["boxes"]["minmax/seeded"]["per_model"]
            r = {"dataset": d, "arch": lbl}
            for role in ("clean", "poisoned"):
                names = [n for n, p in pm.items() if p["role"] == role]
                for c in ("0", "1"):
                    r[f"nc_{role}_{c}"] = float(np.mean([pm[n]["l1"][c] for n in names]))
                    r[f"nc_{role}_{c}_sd"] = float(np.std([pm[n]["l1"][c] for n in names], ddof=1))
                    r[f"mm_{role}_{c}"] = float(np.mean([mm[n]["margins"][c] for n in names]))
                    r[f"mm_{role}_{c}_sd"] = float(np.std([mm[n]["margins"][c] for n in names], ddof=1))
            r["nc_clean_ratio_range"] = [float(min(p["ratio"] for p in pm.values() if p["role"] == "clean")),
                                         float(max(p["ratio"] for p in pm.values() if p["role"] == "clean"))]
            r["nc_poisoned_ratio_range"] = [float(min(p["ratio"] for p in pm.values() if p["role"] == "poisoned")),
                                            float(max(p["ratio"] for p in pm.values() if p["role"] == "poisoned"))]
            rows.append(r)
    return rows


def compute(S):
    tA = sum(e.get("train_time_s", 0) for d in S["manA"]["datasets"].values() for e in d["new"])
    tD = sum(e.get("train_time_s", 0) for d in S["manD"]["datasets"].values() for e in d["new"])
    tDb = sum(e.get("train_time_s", 0) for d in S["manDb"]["datasets"].values() for e in d["new"])
    nA = sum(len(d["new"]) for d in S["manA"]["datasets"].values())
    nD = sum(len(d["new"]) for d in S["manD"]["datasets"].values())
    nDb = sum(len(d["new"]) for d in S["manDb"]["datasets"].values())
    ev = {"A (NC, n = 10 + 10)": S["A"]["runtime_s"], "B (MM-BD)": S["B"]["runtime_s"],
          "B diagnostic (exploratory NC)": S["diag"]["runtime_s"], "C (IBD-PSC + STRIP)": S["C"]["runtime_s"],
          "C checks": S["Cchk"]["runtime_s"], "D (evaluation)": S["D"]["runtime_s"]}
    return {"training_s": {"MLP (Experiment A)": tA, "1D-CNN screening (Experiment D)": tD,
                           "1D-CNN full-data baseline (Experiment D.1)": tDb},
            "evaluation_s": ev, "new_checkpoints": {"MLP": nA, "1D-CNN screening": nD,
                                                    "1D-CNN full-data baseline": nDb,
                                                    "total": nA + nD + nDb}}


def run():
    S = load_all()
    out = {"detection": detection_rows(S), "contrast": contrast_rows(S), "compute": compute(S)}
    exp = resolve_path("experiments/v11_summary")
    write_json(exp / "metrics.json", out)
    _summary(exp / "summary.md", S, out)
    print(f"wrote {exp}")


def _summary(path, S, out):
    L = ["# Extended experiments (v1.1): consolidated tables", "",
         "- **Run:** `python -m src.eval.v11_report` (reads committed metrics files only).",
         "- **Models:** per dataset and architecture, 10 clean and 10 benign/normal-target poisoned models "
         "(MLP: 5 v1.0 models + 5 new seeds; 1D-CNN: 10 new seeds). Model-level AUCs compare 10 clean "
         "with 10 poisoned models (stratified bootstrap 95% CI, 10,000 resamples). Runtime AUCs and TPRs are "
         "means over the 10 poisoned models (bootstrap 95% CI over models). Counts are k/n models.", "",
         "## 1. Detection: all detectors, both datasets, both architectures", "",
         "Pre-specified rules. Neural Cleanse (NC): flag if the mask-norm ratio > 2.0 × the median of the 5 "
         "calibration clean models. MM-BD primary: flag if S = max_c r_c / calibration median > 2.0; secondary: "
         "flag if S exceeds the largest leave-one-out S among the calibration models. FPR = 5 disjoint clean test "
         "models (and leave-one-out over all 10 clean models where given). STRIP and IBD-PSC: TPR at the 5% FPR "
         "threshold of the clean holdout (realized FPR on clean test inputs); IBD-PSC also at its fixed T = 0.9.", "",
         "| dataset × architecture | Neural Cleanse | MM-BD, min/max box | MM-BD, 1st–99th percentile box | STRIP (α 0.7) | IBD-PSC |",
         "|---|---|---|---|---|---|"]
    for r in out["detection"]:
        nc, a, b, st, ib = r["nc"], r["mmbd_minmax"], r["mmbd_p1_p99"], r["strip"], r["ibd"]

        def mm(x):
            return (f"AUC {x['auc']:.2f} {_ci(x['ci'])}<br>S > 2.0: TPR {x['primary_tpr']}, FPR {x['primary_fpr']} "
                    f"(LOO {x['primary_fpr_loo']})<br>LOO-max (S > {x['secondary_thr']:.2f}): TPR {x['secondary_tpr']}, "
                    f"FPR {x['secondary_fpr']}")
        L.append(f"| {r['dataset']} × {r['arch']} | AUC {nc['auc']:.2f} {_ci(nc['ci'])}<br>2.0×: TPR {nc['tpr']}, "
                 f"FPR {nc['fpr_test']} (LOO {nc['fpr_loo']}) | {mm(a)} | {mm(b)} | "
                 f"AUC {st['auc']:.2f} {_ci(st['ci'])}<br>TPR@5% {st['tpr5']:.3f} (FPR {st['fpr5']:.3f}) | "
                 f"AUC {ib['auc']:.2f} {_ci(ib['ci'])}<br>TPR@5% {ib['tpr5']:.3f} (FPR {ib['fpr5']:.3f})<br>"
                 f"T = 0.9: TPR {ib['tprT']:.2f}, FPR {ib['fprT']:.2f} |")
    L += ["", "Target class identified (smallest NC mask / largest MM-BD relative margin, poisoned models):", "",
          "| dataset × architecture | NC | MM-BD, min/max | MM-BD, 1st–99th | MM-BD min/max AUC, pre-specified random starts |",
          "|---|---|---|---|---|"]
    for r in out["detection"]:
        pre = r["mmbd_minmax"]["prespecified_random_only_auc"]
        L.append(f"| {r['dataset']} × {r['arch']} | {r['nc']['target_id']:.2f} | {r['mmbd_minmax']['target_id']:.2f} | "
                 f"{r['mmbd_p1_p99']['target_id']:.2f} | {'undefined (margins ≤ 0)' if pre is None else f'{pre:.2f}'} |")
    L += ["", "MM-BD uses holdout-seeded starts (a post-hoc correction of Experiment B, applied everywhere); the "
          "last column is the pre-specified random-start estimate. Two-sided percentile rule (exploratory, MLP "
          "only): " + "; ".join(
              f"{r['dataset']} STRIP TPR {r['strip']['two_sided_tpr']:.3f} (FPR {r['strip']['two_sided_fpr']:.3f}), "
              f"IBD-PSC TPR {r['ibd']['two_sided_tpr']:.3f} (FPR {r['ibd']['two_sided_fpr']:.3f})"
              for r in out["detection"] if "two_sided_tpr" in r["strip"]) + ".", "",
          "## 2. Target-class contrast (natural vs poisoned)", "",
          "Mean (SD) over 10 clean and 10 poisoned models. Target class = class 0 (benign/normal). NC: minimal mask "
          "L1 (target-excluded, count-matched; smaller = easier to reach). MM-BD: maximum margin r_c (min/max box, "
          "seeded; larger = easier to push to that class).", "",
          "| dataset × architecture | NC mask, target: clean → poisoned | NC mask, non-target: clean → poisoned | "
          "NC ratio range: clean / poisoned | MM-BD margin, target: clean → poisoned | MM-BD margin, non-target: clean → poisoned |",
          "|---|---|---|---|---|---|"]
    for r in out["contrast"]:
        L.append(f"| {r['dataset']} × {r['arch']} | {r['nc_clean_0']:.2f} ({r['nc_clean_0_sd']:.2f}) → "
                 f"{r['nc_poisoned_0']:.2f} ({r['nc_poisoned_0_sd']:.2f}) | {r['nc_clean_1']:.2f} ({r['nc_clean_1_sd']:.2f}) → "
                 f"{r['nc_poisoned_1']:.2f} ({r['nc_poisoned_1_sd']:.2f}) | "
                 f"{r['nc_clean_ratio_range'][0]:.1f}–{r['nc_clean_ratio_range'][1]:.1f} / "
                 f"{r['nc_poisoned_ratio_range'][0]:.1f}–{r['nc_poisoned_ratio_range'][1]:.1f} | "
                 f"{r['mm_clean_0']:.0f} ({r['mm_clean_0_sd']:.0f}) → {r['mm_poisoned_0']:.0f} ({r['mm_poisoned_0_sd']:.0f}) | "
                 f"{r['mm_clean_1']:.0f} ({r['mm_clean_1_sd']:.0f}) → {r['mm_poisoned_1']:.0f} ({r['mm_poisoned_1_sd']:.0f}) |")
    # ---- headline numbers: v1.0 vs v1.1
    A, p7e, p7f = S["A"]["datasets"], S["p7e"], S["p7f"]
    L += ["", "## 3. Headline numbers: v1.0 (frozen files) vs v1.1", "",
          "| quantity | dataset | v1.0 | v1.1 MLP (n = 10 + 10) | v1.1 1D-CNN (n = 10 + 10) |", "|---|---|---|---|---|"]
    for d in DS:
        a5 = p7e[d]["part3"]["auc_target0_evasion_vs_clean"]
        n10 = A[d]["n10"]
        dn = S["Dnc"]["datasets"][d]["n10"]
        L.append(f"| NC screening ROC-AUC [95% CI] | {d} | {a5['roc_auc']:.3f} {_ci(a5['roc_ci95'])} (5 + 5) | "
                 f"{n10['auc']['roc_auc']:.3f} {_ci(n10['auc']['roc_ci95'])} | {dn['auc']['roc_auc']:.3f} {_ci(dn['auc']['roc_ci95'])} |")
        p2 = p7e[d]["part2"]
        L.append(f"| NC flag rate / FPR test group / FPR LOO | {d} | {p2['tpr_target0_evasion']:.2f} / "
                 f"{p2['fpr_test_group']:.2f} (of 2) / {p2['fpr_leave_one_out']:.2f} (of 5) | {_r(n10['rule']['tpr'])} / "
                 f"{_r(n10['rule']['fpr_test_group'])} / {_r(n10['rule']['fpr_leave_one_out'])} | {_r(dn['rule']['tpr'])} / "
                 f"{_r(dn['rule']['fpr_test_group'])} / {_r(dn['rule']['fpr_leave_one_out'])} |")
        L.append(f"| NC Mann-Whitney p (one-sided), rank-biserial r | {d} | {A[d]['n5']['mann_whitney']['p_one_sided_greater']:.2g}, "
                 f"{A[d]['n5']['mann_whitney']['rank_biserial']:.2f} (recomputed) | {n10['mann_whitney']['p_one_sided_greater']:.2g}, "
                 f"{n10['mann_whitney']['rank_biserial']:.2f} | {dn['mann_whitney']['p_one_sided_greater']:.2g}, "
                 f"{dn['mann_whitney']['rank_biserial']:.2f} |")
        al = p7f[d]["part1_alpha_sweep"]
        rt = S["C"]["datasets"][d]["summary"]["all_n10"]["strip"]
        drt = S["Drt"]["datasets"][d]["summary"]["all_n10"]["strip"]
        L.append(f"| STRIP TPR @ 5% FPR (α 0.7) / best variant | {d} | {al['per_alpha']['alpha_0.7']['tpr5']['mean']:.3f} / "
                 f"{al['max_tpr5_over_alphas']:.3f} | {_mci(rt['tpr_5fpr'])} | {_mci(drt['tpr_5fpr'])} |")
        at, dat = A[d]["attack"]["all_n10"], S["D"]["datasets"][d]["attack"]
        art = ("0.992 ± 0.008 (6 full-data seeds, phase7)" if d == "CICIDS2017"
               else f"{S['p7d']['conditions']['target_normal']['asr']['mean']:.3f} (5 seeds, phase7d)")
        L.append(f"| ASR | {d} | {art} | {_mci(at['asr'])} | {_mci(dat['asr'])} |")
        L.append(f"| clean-accuracy shift (poisoned − clean) | {d} | – | {at['shift']['accuracy']['mean']:+.4f} "
                 f"{_ci(at['shift']['accuracy']['boot_ci95'], 4)} | {dat['shift']['accuracy']['mean']:+.4f} "
                 f"{_ci(dat['shift']['accuracy']['boot_ci95'], 4)} |")
    # ---- class properties
    fig = {(e["panel"], e["dataset"], e["class"]): e for e in S["fig3"]}
    cp = S["D"]["class_properties"]
    L += ["", "## 4. Class properties on clean models (findings d–f)", "",
          "| dataset (class-0 share) | property | v1.0 Figure 3 (n = 5): benign/normal vs attack | MLP n = 10: class 0 vs 1 | "
          "1D-CNN n = 10: class 0 vs 1 |", "|---|---|---|---|---|"]
    for d in DS:
        for prop, panel, lbl in (("nc_mask_l1", "reachability", "NC minimal mask (smaller = easier to reach)"),
                                 ("strip_entropy", "homogeneity", "STRIP entropy (smaller = more homogeneous)"),
                                 ("mmbd_margin", None, "MM-BD margin (larger = easier to reach)")):
            if panel:
                fb = fig.get((panel, d, "normal/benign"))
                fa = fig.get((panel, d, "attack"))
                art = f"{fb['value']:.3f} vs {fa['value']:.3f}" if fb and fa else "–"
            else:
                art = "–"
            m, c = cp[d]["mlp"][prop], cp[d]["cnn1d"][prop]
            L.append(f"| {d} ({cp[d]['class0_train_fraction']:.2f}) | {lbl} | {art} | {m['class0']['mean']:.3f} vs "
                     f"{m['class1']['mean']:.3f} ({m['models_with_class0_smaller']}/{m['n_models']} models class 0 smaller, "
                     f"p = {m['wilcoxon_p_two_sided']:.2g}) | {c['class0']['mean']:.3f} vs {c['class1']['mean']:.3f} "
                     f"({c['models_with_class0_smaller']}/{c['n_models']}, p = {c['wilcoxon_p_two_sided']:.2g}) |")
    L += ["", "MLP n = 10 (CICIDS2017) uses the condition-B clean models of the screening experiments, while the v1.0 "
          "CICIDS2017 entropy bar uses the clean reference models (`clean_entropy`); the UNSW-NB15 MLP entropies of seeds "
          "42–46 reproduce phase7d exactly.", "",
          "## 5. Compute and checkpoints", ""]
    c = out["compute"]
    for k, v in c["training_s"].items():
        L.append(f"- Training, {k}: {v / 3600:.2f} h")
    for k, v in c["evaluation_s"].items():
        L.append(f"- Evaluation, {k}: {v / 60:.1f} min")
    tot = sum(c["training_s"].values()) + sum(c["evaluation_s"].values())
    n = c["new_checkpoints"]
    L += [f"- **Total: {tot / 3600:.1f} h of CPU wall-clock time; {n['total']} new checkpoints** "
          f"({n['MLP']} MLP, {n['1D-CNN screening']} 1D-CNN screening, {n['1D-CNN full-data baseline']} 1D-CNN full-data baseline).",
          "", "## Artifacts", "", "- `metrics.json`, `summary.md`"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
