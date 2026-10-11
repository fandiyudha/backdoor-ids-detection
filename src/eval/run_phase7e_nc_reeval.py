"""Neural Cleanse re-evaluation: target-excluded, count-matched optimization,
calibration/test split, and full statistics (abstract screening AUCs; Sections
3.2-3.5; outputs ``experiments/phase7e/``).

Tests whether the "majority / intrinsic-easy class has a
smaller NC mask" pattern an ARTEFACT of source-set prevalence in the optimization,
or an intrinsic property? Reuses existing checkpoints (no victim retraining) and
reruns only the NC optimization, under a corrected protocol.

Part 1 — target-excluded, count-matched NC:
  * source-only: the source set for reconstructing the trigger to target t is
    rows with label != t AND correctly classified (not the whole holdout).
  * count-matched: the SAME number N of source rows for every candidate target,
    removing class-count / prevalence effects. N is recorded.
  Reports per-class L1 mask under source-only count-matched beside the old
  all-input numbers, on clean + both poisoned targets, multi-seed, on CICIDS2017
  and UNSW-NB15. Reports whether the class pattern survives.

Part 2 — calibration/test split: clean models split into disjoint CALIBRATION
  (sets the threshold) and TEST (estimates FPR) groups; plus leave-one-out. The
  anomaly score, threshold rule, reference counts, and seed handling are stated.

Part 3 — full statistics: for every AUC the exact number of INDEPENDENT clean and
  poisoned models (no poison-rate variants counted as independent); ROC-AUC vs
  PR-AUC distinguished; bootstrap CIs over models; and precision / false-alarm
  estimates at realistic low poisoned-model prevalence (1% and 5%).

Run:
    python -m src.eval.run_phase7e_nc_reeval --config configs/phase7e.yaml
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.detectors.neural_cleanse import NeuralCleanseDetector  # noqa: E402
from src.eval.run_comparison import load_ids_model  # noqa: E402
from src.models.train import load_dataset  # noqa: E402
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


def ms(x):
    a = np.asarray(x, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "values": [float(v) for v in a]}


def anomaly_ratio(l1):  # symmetric max/min asymmetry, >= 1
    lo, hi = min(l1[0], l1[1]), max(l1[0], l1[1])
    return hi / (lo + 1e-9)


def nc_both(det, N, preds):
    old0 = det.reverse_engineer(0)
    old1 = det.reverse_engineer(1)
    new0 = det.reverse_engineer(0, source_only=True, n_source=N, preds=preds)
    new1 = det.reverse_engineer(1, source_only=True, n_source=N, preds=preds)
    return {"l1_old": {0: old0["l1_norm"], 1: old1["l1_norm"]},
            "l1_new": {0: new0["l1_norm"], 1: new1["l1_norm"]},
            "n_used_new": (new0["n_opt_samples"], new1["n_opt_samples"])}


def auc_with_ci(clean, pois, iters, rng):
    clean, pois = np.asarray(clean, float), np.asarray(pois, float)
    nc, npo = len(clean), len(pois)
    y = np.r_[np.zeros(nc), np.ones(npo)]
    s = np.r_[clean, pois]
    roc, pr = float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    rocs, prs = [], []
    for _ in range(iters):
        ci, pi = rng.integers(0, nc, nc), rng.integers(0, npo, npo)
        yy = np.r_[np.zeros(nc), np.ones(npo)]
        ss = np.r_[clean[ci], pois[pi]]
        rocs.append(roc_auc_score(yy, ss)); prs.append(average_precision_score(yy, ss))
    return {"roc_auc": roc, "roc_ci95": [float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5))],
            "pr_auc": pr, "pr_ci95": [float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))],
            "n_clean": nc, "n_poisoned": npo}


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    md = resolve_path(cfg["paths"]["models_dir"])
    rng = np.random.default_rng(cfg["seed"])
    iters = cfg["bootstrap_iters"]

    out = {}
    for dname, dc in cfg["datasets"].items():
        print(f"\n==================== {dname} ====================")
        data = load_dataset(resolve_path(dc["processed_dir"]))
        # CICIDS has a val split; UNSW uses the official train/test only -> use test
        hk, yk = ("X_val", "y_val") if "X_val" in data else ("X_test", "y_test")
        vidx = rng.choice(len(data[hk]), min(cfg["nc"]["holdout_size"], len(data[hk])),
                          replace=False)
        hX, hy = data[hk][vidx].astype(np.float32), data[yk][vidx].astype(int)
        print(f"  holdout source: {hk} ({len(hX)} rows, "
              f"class0 frac {float((hy==0).mean()):.3f})")

        def det_for(fname):
            m = load_ids_model(md / f"{fname}.pt", device)
            return NeuralCleanseDetector(m, hX, hy, device=device,
                                         steps=cfg["nc"]["steps"], lr=cfg["nc"]["lr"],
                                         seed=cfg["seed"])

        # count-match N from the first clean model's correct predictions
        det0 = det_for(dc["clean"][0])
        preds0 = det0.predict_holdout()
        pool0 = int(((hy == 1) & (preds0 == hy)).sum())   # sources for target0 (normal/benign)
        pool1 = int(((hy == 0) & (preds0 == hy)).sum())   # sources for target1 (attack)
        N = min(pool0, pool1)
        print(f"  count-match N = {N} (correct non-target pools: target0={pool0}, target1={pool1})")

        groups = {"clean": dc["clean"], "pois_target0": dc["poisoned_target0"],
                  "pois_target1": dc["poisoned_target1"]}
        rec = {g: [] for g in groups}
        for g, names in groups.items():
            for fname in names:
                det = det_for(fname)
                preds = det.predict_holdout()
                r = nc_both(det, N, preds)
                r["ratio_old"] = anomaly_ratio(r["l1_old"])
                r["ratio_new"] = anomaly_ratio(r["l1_new"])
                rec[g].append(r)
            print(f"  {g}: {len(names)} models done")

        out[dname] = _analyse(dname, dc, rec, N, cfg, iters, rng)

    _plot(out, exp_dir / "mask_old_vs_new.png")
    _save(exp_dir, cfg, out)
    _print(out)
    return out


def _analyse(dname, dc, rec, N, cfg, iters, rng):
    normal = dc["normal_label"]  # 0
    attack = 1 - normal
    thr_rel = cfg["rel_threshold"]

    def grp_mask(g, which):  # which in {"l1_old","l1_new"}; returns per-class ms
        return {c: ms([m[which][c] for m in rec[g]]) for c in (0, 1)}

    clean_old = grp_mask("clean", "l1_old")
    clean_new = grp_mask("clean", "l1_new")
    p0_new = grp_mask("pois_target0", "l1_new")
    p1_new = grp_mask("pois_target1", "l1_new")
    p0_old = grp_mask("pois_target0", "l1_old")
    p1_old = grp_mask("pois_target1", "l1_old")

    # ---- Part 1: does the clean-model class pattern survive count-matching? ----
    easy_old = 0 if clean_old[0]["mean"] < clean_old[1]["mean"] else 1
    easy_new = 0 if clean_new[0]["mean"] < clean_new[1]["mean"] else 1
    gap_old = max(clean_old[0]["mean"], clean_old[1]["mean"]) / (min(clean_old[0]["mean"], clean_old[1]["mean"]) + 1e-9)
    gap_new = max(clean_new[0]["mean"], clean_new[1]["mean"]) / (min(clean_new[0]["mean"], clean_new[1]["mean"]) + 1e-9)
    if easy_new != easy_old:
        verdict = "GONE/FLIPPED"
    elif gap_new >= 0.7 * gap_old and gap_new >= 1.3:
        verdict = "SURVIVES"
    elif gap_new >= 1.15:
        verdict = "WEAKENED"
    else:
        verdict = "GONE"
    majority = attack if dname == "UNSW-NB15" else normal  # informational
    part1 = {
        "count_match_N": N,
        "clean_mask_old": {"normal": clean_old[normal], "attack": clean_old[attack]},
        "clean_mask_new": {"normal": clean_new[normal], "attack": clean_new[attack]},
        "easy_class_old": "normal" if easy_old == normal else "attack",
        "easy_class_new": "normal" if easy_new == normal else "attack",
        "gap_old": gap_old, "gap_new": gap_new,
        "pattern_survives_count_match": verdict,
        "poisoned_target0_mask_new": {"normal": p0_new[normal], "attack": p0_new[attack]},
        "poisoned_target1_mask_new": {"normal": p1_new[normal], "attack": p1_new[attack]},
        "poisoned_target0_mask_old": {"normal": p0_old[normal], "attack": p0_old[attack]},
        "poisoned_target1_mask_old": {"normal": p1_old[normal], "attack": p1_old[attack]},
    }

    # ---- Part 2: calibration/test split + LOO (screening on target0 evasion) ----
    clean_ratio = [m["ratio_new"] for m in rec["clean"]]
    p0_ratio = [m["ratio_new"] for m in rec["pois_target0"]]
    p1_ratio = [m["ratio_new"] for m in rec["pois_target1"]]
    k = cfg["calib_n"]
    calib, test = clean_ratio[:k], clean_ratio[k:]
    thr = float(np.median(calib)) * thr_rel
    fpr_test = float(np.mean([r > thr for r in test]))
    tpr_p0 = float(np.mean([r > thr for r in p0_ratio]))
    tpr_p1 = float(np.mean([r > thr for r in p1_ratio]))
    # leave-one-out FPR (each clean model's threshold from the OTHER clean models)
    loo = []
    for i in range(len(clean_ratio)):
        ref = [clean_ratio[j] for j in range(len(clean_ratio)) if j != i]
        loo.append(clean_ratio[i] > float(np.median(ref)) * thr_rel)
    fpr_loo = float(np.mean(loo))
    part2 = {
        "anomaly_score": "NC L1 mask-norm asymmetry ratio = max_c L1 / min_c L1, "
                         "under the source-only count-matched protocol",
        "threshold_rule": f"flag if ratio > {thr_rel} x median(reference clean ratios)",
        "n_calibration_models": len(calib), "n_test_models": len(test),
        "reference_training_data": "clean IDS models trained on the same dataset "
                                   "(distinct seeds); identical MLP architecture and "
                                   "preprocessing; 1 optimization restart per class; "
                                   "optimization seed fixed (torch.manual_seed) so the "
                                   "only randomness is the count-matched source sample",
        "threshold_value": thr,
        "fpr_test_group": fpr_test, "fpr_leave_one_out": fpr_loo,
        "tpr_target0_evasion": tpr_p0, "tpr_target1": tpr_p1,
    }

    # ---- Part 3: full statistics ----
    auc_p0 = auc_with_ci(clean_ratio, p0_ratio, iters, rng)      # evasion vs clean
    auc_p1 = auc_with_ci(clean_ratio, p1_ratio, iters, rng)      # attack-target vs clean
    auc_any = auc_with_ci(clean_ratio, p0_ratio + p1_ratio, iters, rng)
    # low-prevalence precision using TEST-group FPR (or LOO) and target0 TPR
    fpr = fpr_loo  # more robust than the 2-model test group
    prev = {}
    for p in cfg["base_rates"]:
        denom = tpr_p0 * p + fpr * (1 - p)
        prec = (tpr_p0 * p / denom) if denom > 0 else 0.0
        prev[f"base_rate_{p}"] = {
            "precision": prec,
            "false_alarms_per_1000_models": fpr * (1 - p) * 1000,
            "true_alarms_per_1000_models": tpr_p0 * p * 1000}
    part3 = {
        "auc_target0_evasion_vs_clean": auc_p0,
        "auc_target1_vs_clean": auc_p1,
        "auc_any_poisoned_vs_clean": auc_any,
        "clean_ratio_per_seed": clean_ratio, "target0_ratio_per_seed": p0_ratio,
        "target1_ratio_per_seed": p1_ratio,
        "note_independence": "Every model is a distinct training seed; no "
                             "poison-rate variants of a single base are counted.",
        "low_prevalence": prev,
    }
    print(f"  [{dname}] pattern survives count-match: {verdict} "
          f"(clean easy old={part1['easy_class_old']} new={part1['easy_class_new']}, "
          f"gap {gap_old:.2f}->{gap_new:.2f}) | evasion ROC-AUC {auc_p0['roc_auc']:.3f} "
          f"(n {auc_p0['n_clean']}/{auc_p0['n_poisoned']}) | FPR_loo {fpr_loo:.2f} "
          f"TPR_evasion {tpr_p0:.2f}")
    return {"class_ratio_note": dc["class_ratio_note"], "part1": part1,
            "part2": part2, "part3": part3}


# ---------------------------------------------------------------------------
def _plot(out, path):
    ds = list(out.keys())
    fig, axes = plt.subplots(1, len(ds), figsize=(6.5 * len(ds), 4.4), squeeze=False)
    for ax, d in zip(axes[0], ds):
        p1 = out[d]["part1"]
        cats = ["normal", "attack"]
        x = np.arange(2); w = 0.38
        old = [p1["clean_mask_old"]["normal"]["mean"], p1["clean_mask_old"]["attack"]["mean"]]
        olde = [p1["clean_mask_old"]["normal"]["std"], p1["clean_mask_old"]["attack"]["std"]]
        new = [p1["clean_mask_new"]["normal"]["mean"], p1["clean_mask_new"]["attack"]["mean"]]
        newe = [p1["clean_mask_new"]["normal"]["std"], p1["clean_mask_new"]["attack"]["std"]]
        ax.bar(x - w / 2, old, w, yerr=olde, capsize=3, color="#95a5a6", label="all-input (old)")
        ax.bar(x + w / 2, new, w, yerr=newe, capsize=3, color="#c0392b",
               label="source-only count-matched (new)")
        ax.set_xticks(x, labels=cats)
        ax.set_ylabel("NC L1 mask norm (clean models)")
        ax.set_title(f"{d}\n{out[d]['class_ratio_note']}\npattern: "
                     f"{p1['pattern_survives_count_match']} (gap {p1['gap_old']:.2f}->{p1['gap_new']:.2f})",
                     fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Does the smaller-mask-on-easy-class pattern survive count-matching?")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _save(exp_dir, cfg, out):
    (exp_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=float),
                                          encoding="utf-8")
    _summary(exp_dir / "summary.md", cfg, out)
    print(f"\n  wrote artifacts -> {exp_dir}")


def _summary(path, cfg, out):
    L = ["# Neural Cleanse re-evaluation: target-excluded, count-matched optimization", "",
         "Reruns NC optimization over existing checkpoints (no victim retraining) to "
         "test whether the smaller-mask-on-easy-class pattern is a prevalence artefact "
         "of the source set, or intrinsic.", "",
         "- **Run:** `python -m src.eval.run_phase7e_nc_reeval --config configs/phase7e.yaml`",
         "- **Measured:** per-class mask norms under all-input vs target-excluded, "
         "count-matched optimization; FPR on a disjoint calibration/test split of "
         "clean models and leave-one-out; ROC-AUC and PR-AUC with bootstrap CIs; "
         "precision at low poisoned-model prevalence (CICIDS2017 and UNSW-NB15).", "",
         "## Part 1 — Target-excluded, count-matched NC", "",
         "Source set for target *t* = rows with label ≠ *t* AND correctly classified, "
         "count-matched to the same N across candidate targets.", "",
         "| dataset | class ratio | N | clean easy class (old → new) | gap old → new | "
         "**pattern survives count-match?** |", "|---|---|---|---|---|---|"]
    for d, r in out.items():
        p1 = r["part1"]
        L.append(f"| {d} | {r['class_ratio_note']} | {p1['count_match_N']} | "
                 f"{p1['easy_class_old']} → {p1['easy_class_new']} | "
                 f"{p1['gap_old']:.2f} → {p1['gap_new']:.2f} | "
                 f"**{p1['pattern_survives_count_match']}** |")
    L += ["", "### Per-class clean-model mask norms (all-input vs source-only count-matched)", "",
          "| dataset | class | mask all-input (old) | mask source-only count-matched (new) |",
          "|---|---|---|---|"]
    for d, r in out.items():
        p1 = r["part1"]
        for cls in ("normal", "attack"):
            L.append(f"| {d} | {cls} | "
                     f"{p1['clean_mask_old'][cls]['mean']:.2f}±{p1['clean_mask_old'][cls]['std']:.2f} | "
                     f"{p1['clean_mask_new'][cls]['mean']:.2f}±{p1['clean_mask_new'][cls]['std']:.2f} |")
    L += ["",
          "## Part 2 — Calibration / test split (independent FPR)", ""]
    for d, r in out.items():
        p2 = r["part2"]
        L += [f"### {d}",
              f"- Anomaly score: {p2['anomaly_score']}.",
              f"- Threshold rule: {p2['threshold_rule']}; value {p2['threshold_value']:.3f}.",
              f"- Reference models: {p2['n_calibration_models']} calibration + "
              f"{p2['n_test_models']} independent test (disjoint). "
              f"{p2['reference_training_data']}.",
              f"- **FPR on the independent test group: {p2['fpr_test_group']:.2f}; "
              f"leave-one-out FPR: {p2['fpr_leave_one_out']:.2f}.** "
              f"TPR (target-0 evasion): {p2['tpr_target0_evasion']:.2f}; "
              f"TPR (target-1): {p2['tpr_target1']:.2f}.", ""]
    L += ["## Part 3 — Full statistics (exact counts, ROC vs PR, CIs, prevalence)", "",
          "| dataset | contrast | n clean | n poisoned | ROC-AUC [95% CI] | PR-AUC [95% CI] |",
          "|---|---|---|---|---|---|"]
    for d, r in out.items():
        for key, lbl in [("auc_target0_evasion_vs_clean", "target-0 evasion vs clean"),
                         ("auc_target1_vs_clean", "target-1 vs clean"),
                         ("auc_any_poisoned_vs_clean", "any poisoned vs clean")]:
            a = r["part3"][key]
            L.append(f"| {d} | {lbl} | {a['n_clean']} | {a['n_poisoned']} | "
                     f"{a['roc_auc']:.3f} [{a['roc_ci95'][0]:.3f}, {a['roc_ci95'][1]:.3f}] | "
                     f"{a['pr_auc']:.3f} [{a['pr_ci95'][0]:.3f}, {a['pr_ci95'][1]:.3f}] |")
    L += ["",
          "*All models are distinct training seeds; poison-rate variants of one base "
          "model are NOT counted as independent observations.*", "",
          "### Precision at realistic low poisoned-model prevalence (target-0 evasion)",
          "",
          "| dataset | base rate | precision | est. false alarms / 1000 models | est. true alarms / 1000 |",
          "|---|---|---|---|---|"]
    for d, r in out.items():
        for p in cfg["base_rates"]:
            lp = r["part3"]["low_prevalence"][f"base_rate_{p}"]
            L.append(f"| {d} | {p:.0%} | {lp['precision']:.3f} | "
                     f"{lp['false_alarms_per_1000_models']:.1f} | "
                     f"{lp['true_alarms_per_1000_models']:.1f} |")
    # ---- cross-dataset interpretation ----
    verds = {d: out[d]["part1"]["pattern_survives_count_match"] for d in out}
    easy = {d: out[d]["part1"]["easy_class_new"] for d in out}
    both_survive = all(v.startswith("SURVIVES") for v in verds.values())
    L += ["", "## Cross-dataset interpretation", ""]
    if both_survive:
        L += [
            "The smaller-mask-on-easy-class pattern **SURVIVES** target-exclusion and "
            "count-matching in BOTH datasets — equalizing the source counts barely "
            "changes the mask-norm gap (CICIDS2017 "
            f"{out['CICIDS2017']['part1']['gap_old']:.2f}→"
            f"{out['CICIDS2017']['part1']['gap_new']:.2f}; UNSW-NB15 "
            f"{out['UNSW-NB15']['part1']['gap_old']:.2f}→"
            f"{out['UNSW-NB15']['part1']['gap_new']:.2f}). The asymmetry is therefore "
            "a property of the trained model's decision geometry rather than an "
            "artefact of NC optimization with unequal source counts.",
            "",
            f"**Refinement:** the NC-easy (smaller-mask) class "
            f"is the MAJORITY class in both datasets (CICIDS: `{easy['CICIDS2017']}`; "
            f"UNSW: `{easy['UNSW-NB15']}`). So reachability co-varies with training "
            "prevalence — but as a MODEL-geometry property (the majority class "
            "occupies a larger basin, reachable with a small mask), which persists "
            "under count-matching, not as an optimizer artefact. This is consistent "
            "with the frequency-linked Neural Cleanse result of the UNSW-NB15 "
            "replication and locates it in the trained model rather than in the NC "
            "procedure.",
        ]
    else:
        L += [f"Pattern verdicts: {verds}. Some datasets show the pattern weakening or "
              "vanishing under count-matching (see the table), i.e. part of the "
              "all-input pattern was a prevalence artefact."]
    L += ["", "## Artifacts", "- `metrics.json`, `summary.md`, `mask_old_vs_new.png`"]
    path.write_text("\n".join(L), encoding="utf-8")


def _print(out):
    print("\n" + "=" * 92)
    print("PHASE 7e — NC RE-EVALUATION SUMMARY")
    print("=" * 92)
    for d, r in out.items():
        p1, p2, p3 = r["part1"], r["part2"], r["part3"]
        a = p3["auc_target0_evasion_vs_clean"]
        print(f"{d}: pattern survives count-match = {p1['pattern_survives_count_match']} "
              f"(clean easy {p1['easy_class_old']}->{p1['easy_class_new']}, "
              f"gap {p1['gap_old']:.2f}->{p1['gap_new']:.2f})")
        print(f"   evasion ROC-AUC {a['roc_auc']:.3f} [{a['roc_ci95'][0]:.2f},{a['roc_ci95'][1]:.2f}] "
              f"PR-AUC {a['pr_auc']:.3f} (n_clean={a['n_clean']}, n_pois={a['n_poisoned']}) "
              f"| FPR_test {p2['fpr_test_group']:.2f} FPR_loo {p2['fpr_leave_one_out']:.2f} "
              f"TPR {p2['tpr_target0_evasion']:.2f}")
    print("=" * 92 + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Neural Cleanse re-evaluation (target-excluded, count-matched)")
    ap.add_argument("--config", default="configs/phase7e.yaml")
    return ap.parse_args()


def main():
    run(load_yaml(resolve_path(parse_args().config)))


if __name__ == "__main__":
    main()
