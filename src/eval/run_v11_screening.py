"""v1.1 Experiment A: Neural Cleanse screening with n = 10 + 10 models per dataset
(outputs ``experiments/v11_screening/``).

Reruns the paper's final screening protocol (``run_phase7e_nc_reeval``) on the 5
existing and 5 new clean models and the 5 existing and 5 new poisoned
(benign/normal-target) models of each dataset:

* the same 3,000-row clean holdout as phase7e (its random draw is replayed);
* target-excluded, count-matched Neural Cleanse (N from the first existing clean
  model, as in phase7e), 500 steps, one restart per class, fixed optimization seed;
* anomaly statistic: max_c L1 / min_c L1 of the two reverse-engineered masks;
* paired reference-relative rule: flag if the statistic exceeds 2.0 x the median
  statistic of the calibration clean models; FPR on a disjoint clean test group
  and by leave-one-out.

Reports, side by side for n = 5 (the frozen phase7e values) and n = 10: ROC-AUC
and PR-AUC with stratified bootstrap 95% CIs, the flag rate and false-positive
rate with bootstrap and exact intervals, a Mann-Whitney U test with effect
sizes, and the attack success rate and clean-performance shift of the poisoned
models. The existing models' statistics are recomputed and checked against the
frozen phase7e values before anything else is reported.

Run (after ``python -u -m src.eval.v11_train_models --config configs/v11_screening.yaml``):
    python -u -m src.eval.run_v11_screening --config configs/v11_screening.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.detectors.neural_cleanse import NeuralCleanseDetector  # noqa: E402
from src.eval.run_causal_test_ids import asr_generic  # noqa: E402
from src.eval.v11_common import (  # noqa: E402
    CPU,
    auc_boot,
    clean_performance,
    environment,
    git_commit,
    load_dataset_trigger,
    load_model_cpu,
    load_training_data,
    mann_whitney,
    mean_ci,
    model_names,
    models_manifest_entry,
    phase7e_holdouts,
    reference_relative_rule,
    utc_now,
    write_json,
)
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402

PERF_KEYS = ("accuracy", "attack_recall", "attack_f1", "pr_auc")


def nc_count_matched(model, hX, hy, N, nc_cfg, seed):
    """phase7e's 'new' protocol for one model: source-only, count-matched NC."""
    det = NeuralCleanseDetector(model, hX, hy, device=CPU, steps=nc_cfg["steps"],
                                lr=nc_cfg["lr"], seed=seed)
    preds = det.predict_holdout()
    t0 = time.perf_counter()
    r = {c: det.reverse_engineer(c, source_only=True, n_source=N, preds=preds) for c in (0, 1)}
    elapsed = time.perf_counter() - t0
    l1 = {str(c): float(r[c]["l1_norm"]) for c in (0, 1)}
    lo, hi = min(l1.values()), max(l1.values())
    return {"l1": l1, "ratio": hi / (lo + 1e-9),
            "smallest_mask_class": int(min(l1, key=l1.get)),
            "reversal_success": {str(c): float(r[c]["reversal_success"]) for c in (0, 1)},
            "n_source": {str(c): int(r[c]["n_opt_samples"]) for c in (0, 1)},
            "time_s": elapsed}


def count_match_N(model, hX, hy, nc_cfg, seed):
    det = NeuralCleanseDetector(model, hX, hy, device=CPU, steps=nc_cfg["steps"],
                                lr=nc_cfg["lr"], seed=seed)
    p = det.predict_holdout()
    return min(int(((hy == 1) & (p == hy)).sum()), int(((hy == 0) & (p == hy)).sum()))


def attack_metrics(dc, names, md, iters, seed, log_prefix):
    """ASR, natural ASR, and clean-performance shift (poisoned minus same-seed clean)."""
    data = load_training_data(dc)
    trigger = load_dataset_trigger(dc, data["feature_names"], data["X_train"])
    Xte, yte = data["X_test"], data["y_test"]
    tgt, src = dc["poison"]["target"], dc["poison"]["source"]
    rows = []
    for cname, pname in zip(names["clean"], names["poisoned"]):
        cm, pm = load_model_cpu(md, cname), load_model_cpu(md, pname)
        cperf, pperf = clean_performance(cm, Xte, yte), clean_performance(pm, Xte, yte)
        rows.append({"clean": cname, "poisoned": pname,
                     "asr": asr_generic(pm, Xte, yte, trigger, tgt, src, CPU)["asr"],
                     "natural_asr_clean": asr_generic(cm, Xte, yte, trigger, tgt, src, CPU)["asr"],
                     "clean_perf": cperf, "poisoned_perf": pperf,
                     "shift": {k: pperf[k] - cperf[k] for k in PERF_KEYS}})
        print(f"  {log_prefix} {pname}: ASR {rows[-1]['asr']:.3f} | nat-ASR "
              f"{rows[-1]['natural_asr_clean']:.3f} | dAcc {rows[-1]['shift']['accuracy']:+.4f} "
              f"dF1 {rows[-1]['shift']['attack_f1']:+.4f} dPR-AUC {rows[-1]['shift']['pr_auc']:+.4f}",
              flush=True)

    def agg(sel):
        rs = [rows[i] for i in sel]
        out = {"asr": mean_ci([r["asr"] for r in rs], iters, seed),
               "natural_asr_clean": mean_ci([r["natural_asr_clean"] for r in rs], iters, seed),
               "shift": {k: mean_ci([r["shift"][k] for r in rs], iters, seed) for k in PERF_KEYS},
               "clean_perf": {k: mean_ci([r["clean_perf"][k] for r in rs], iters, seed)
                              for k in PERF_KEYS},
               "poisoned_perf": {k: mean_ci([r["poisoned_perf"][k] for r in rs], iters, seed)
                                 for k in PERF_KEYS}}
        return out

    n_exist = len(dc["existing"]["clean"])
    return {"per_pair": rows, "existing_n5": agg(range(n_exist)),
            "new_n5": agg(range(n_exist, len(rows))), "all_n10": agg(range(len(rows))),
            "n_test_rows": int(len(yte)),
            "asr_definition": "fraction of correctly classified source-class test rows "
                              "predicted as the target class once the trigger is stamped"}


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    t_start = time.perf_counter()
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    md = resolve_path(cfg["paths"]["models_dir"])
    iters, seed = cfg["bootstrap_iters"], cfg["seed"]
    frozen = json.loads(resolve_path(cfg["reference_n5"]["metrics"]).read_text(encoding="utf-8"))
    holdouts = phase7e_holdouts(cfg["holdout_replay_config"])

    out = {"datasets": {}}
    for dname, dc in cfg["datasets"].items():
        print(f"\n==================== {dname} ====================", flush=True)
        names = model_names(dc)
        missing = [n for r in names.values() for n in r if not (md / f"{n}.pt").exists()]
        if missing:
            raise FileNotFoundError(f"{dname}: missing checkpoints {missing}; run "
                                    "src.eval.v11_train_models first")
        h = holdouts[dname]
        N = count_match_N(load_model_cpu(md, names["clean"][0]), h["X"], h["y"], cfg["nc"], seed)
        fz = frozen[dname]
        assert N == fz["part1"]["count_match_N"], (N, fz["part1"]["count_match_N"])
        print(f"  holdout {h['source_split']} ({len(h['X'])} rows); count-match N = {N} "
              f"(phase7e: {fz['part1']['count_match_N']})", flush=True)

        nc = {}
        for role in ("clean", "poisoned"):
            for n in names[role]:
                nc[n] = nc_count_matched(load_model_cpu(md, n), h["X"], h["y"], N, cfg["nc"], seed)
                print(f"  NC {role:8s} {n}: L1 normal {nc[n]['l1']['0']:.2f} attack "
                      f"{nc[n]['l1']['1']:.2f} ratio {nc[n]['ratio']:.2f} "
                      f"({nc[n]['time_s']:.1f}s)", flush=True)
        clean_r = [nc[n]["ratio"] for n in names["clean"]]
        pois_r = [nc[n]["ratio"] for n in names["poisoned"]]
        k = len(dc["existing"]["clean"])

        # fidelity: existing models must reproduce the frozen phase7e statistics
        fz_clean, fz_pois = (fz["part3"]["clean_ratio_per_seed"],
                             fz["part3"]["target0_ratio_per_seed"])
        diffs = np.abs(np.r_[clean_r[:k], pois_r[:k]] - np.r_[fz_clean, fz_pois])
        fidelity = {"max_abs_diff_vs_phase7e": float(diffs.max()),
                    "identical": bool(diffs.max() == 0.0)}
        print(f"  fidelity vs phase7e: max |diff| = {fidelity['max_abs_diff_vs_phase7e']:.3e}",
              flush=True)

        n5 = {"source": "frozen phase7e values (experiments/phase7e/metrics.json)",
              "frozen_auc_2000_resamples": fz["part3"]["auc_target0_evasion_vs_clean"],
              "frozen_rule_calib3": fz["part2"],
              "auc": auc_boot(fz_clean, fz_pois, iters, seed),
              "mann_whitney": mann_whitney(fz_clean, fz_pois),
              "rule": reference_relative_rule(fz_clean, fz_pois, 3, cfg["rel_threshold"],
                                              iters, seed),
              "clean_ratio": fz_clean, "poisoned_ratio": fz_pois}
        n10 = {"auc": auc_boot(clean_r, pois_r, iters, seed),
               "mann_whitney": mann_whitney(clean_r, pois_r),
               "rule": reference_relative_rule(clean_r, pois_r, cfg["calib_n"],
                                               cfg["rel_threshold"], iters, seed),
               "clean_ratio": clean_r, "poisoned_ratio": pois_r,
               "new_only_auc": auc_boot(clean_r[k:], pois_r[k:], iters, seed)}
        thr10 = n10["rule"]["threshold"]
        n10["new_poisoned_flag_rate_at_calib_threshold"] = float(
            np.mean(np.asarray(pois_r[k:]) > thr10))
        target = dc["poison"]["target"]
        n10["target_is_smallest_mask_rate"] = float(np.mean(
            [nc[n]["smallest_mask_class"] == target for n in names["poisoned"]]))

        print(f"  n=5 : AUC {n5['auc']['roc_auc']:.3f} {n5['auc']['roc_ci95']} | MWU p "
              f"{n5['mann_whitney']['p_one_sided_greater']:.4f}", flush=True)
        print(f"  n=10: AUC {n10['auc']['roc_auc']:.3f} {n10['auc']['roc_ci95']} | MWU p "
              f"{n10['mann_whitney']['p_one_sided_greater']:.2e} | thr {thr10:.3f} TPR "
              f"{n10['rule']['tpr']['rate']:.2f} FPR_test {n10['rule']['fpr_test_group']['rate']:.2f}"
              f" FPR_loo {n10['rule']['fpr_leave_one_out']['rate']:.2f}", flush=True)

        print("  attack metrics ...", flush=True)
        atk = attack_metrics(dc, names, md, iters, seed, dname)
        out["datasets"][dname] = {
            "count_match_N": N, "holdout_source_split": h["source_split"],
            "holdout_size": int(len(h["X"])), "fidelity": fidelity,
            "per_model": {n: {**nc[n], "role": r} for r in ("clean", "poisoned")
                          for n in names[r]},
            "models": names, "n5": n5, "n10": n10, "attack": atk,
            "nc_time_per_model_s": mean_ci([nc[n]["time_s"] for n in nc], iters, seed)}

    out["runtime_s"] = time.perf_counter() - t_start
    _plot(out, cfg, exp_dir / "screening_ratios.png")
    write_json(exp_dir / "metrics.json", out)
    _manifest(exp_dir, cfg, out, md)
    _summary(exp_dir / "summary.md", cfg, out)
    print(f"\n  wrote artifacts -> {exp_dir} ({out['runtime_s']:.0f}s)", flush=True)
    return out


def _manifest(exp_dir, cfg, out, md):
    models = []
    for dname, d in out["datasets"].items():
        dc = cfg["datasets"][dname]
        seeds = list(dc["existing"]["seeds"]) + list(dc["new"]["seeds"])
        for role in ("clean", "poisoned"):
            for n, s in zip(d["models"][role], seeds):
                models.append(models_manifest_entry(n, md, dataset=dname, role=role, seed=s,
                                                    new=s in dc["new"]["seeds"]))
    write_json(exp_dir / "manifest.json", {
        "experiment": "v11_screening", "config": cfg.get("_config_path"),
        "command": "python -u -m src.eval.run_v11_screening --config "
                   f"{cfg.get('_config_path')}",
        "evaluated_at": utc_now(), "git_commit": git_commit(), "environment": environment(),
        "runtime_s": out["runtime_s"], "models": models,
        "training_manifest": "models_manifest.json"})


# ---------------------------------------------------------------------------
def _plot(out, cfg, path):
    ds = list(out["datasets"])
    fig, axes = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 4.2), squeeze=False)
    for ax, d in zip(axes[0], ds):
        r = out["datasets"][d]
        k = len(cfg["datasets"][d]["existing"]["clean"])
        rng = np.random.default_rng(0)
        for x, role, col in ((0, "clean", "#2b6cb0"), (1, "poisoned", "#c53030")):
            v = np.asarray(r["n10"][f"{role}_ratio"])
            jit = rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x + jit[:k], v[:k], marker="o", color=col, s=36,
                       label=f"{role}, existing seeds")
            ax.scatter(x + jit[k:], v[k:], marker="^", facecolor="none", edgecolor=col, s=44,
                       label=f"{role}, new seeds")
        ax.axhline(r["n10"]["rule"]["threshold"], color="#444", ls="--", lw=1,
                   label=f"threshold ({cfg['rel_threshold']} x calibration median)")
        ax.set_yscale("log")
        ax.set_xticks([0, 1], labels=["clean", "poisoned"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("NC mask-norm ratio (max/min)")
        a = r["n10"]["auc"]
        ax.set_title(f"{d}: ROC-AUC {a['roc_auc']:.3f} "
                     f"[{a['roc_ci95'][0]:.2f}, {a['roc_ci95'][1]:.2f}], n = 10 + 10", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _f(x, nd=3):
    return f"{x:.{nd}f}"


def _ci(c, nd=2):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def _rate(r):
    return (f"{r['k']}/{r['n']} = {r['rate']:.2f} (bootstrap {_ci(r['boot_ci95'])}; "
            f"exact {_ci(r['clopper_pearson95'])})")


def _summary(path, cfg, out):
    L = ["# Extended experiment A: Neural Cleanse screening with 10 clean and 10 poisoned "
         "models per dataset", "",
         "- **Run:** `python -u -m src.eval.v11_train_models --config configs/v11_screening.yaml`, "
         "then `python -u -m src.eval.run_v11_screening --config configs/v11_screening.yaml`",
         "- **Protocol:** the paper's final screening protocol (`phase7e`): the same "
         "3,000-row clean holdout, target-excluded and count-matched Neural Cleanse, the "
         "mask-norm ratio max/min as the anomaly statistic, and the paired "
         f"reference-relative rule (flag if the ratio exceeds {cfg['rel_threshold']} x the "
         "median ratio of the calibration clean models).",
         "- **Models:** per dataset, the 5 clean and 5 benign/normal-target poisoned MLP "
         "models of the paper (seeds 42-46) plus 5 clean and 5 poisoned models trained "
         "for this experiment with the same protocol and new seeds "
         f"({', '.join(str(s) for s in next(iter(cfg['datasets'].values()))['new']['seeds'])}).",
         "- **n = 5** columns use the frozen `phase7e` statistics; the AUC CI and the "
         f"Mann-Whitney test are recomputed from them with {cfg['bootstrap_iters']:,} "
         "stratified bootstrap resamples (phase7e used 2,000). **n = 10** columns use "
         "all models. Calibration/test split for n = 10 (fixed before evaluation): the 5 "
         "existing clean models calibrate the threshold, the 5 new clean models are the "
         "disjoint test group. For n = 5 the split is phase7e's 3 + 2.", "",
         "## Protocol fidelity", "",
         "| dataset | count-match N | max abs. difference of the recomputed existing-model "
         "statistics vs phase7e |", "|---|---|---|"]
    for d, r in out["datasets"].items():
        L.append(f"| {d} | {r['count_match_N']} | {r['fidelity']['max_abs_diff_vs_phase7e']:.1e} |")
    L += ["", "## Separation of clean and poisoned models", "",
          "| dataset | n (clean + poisoned) | ROC-AUC [95% CI] | PR-AUC [95% CI] | "
          "Mann-Whitney U | p (one-sided) | rank-biserial r |",
          "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for lbl, s in (("5 + 5", r["n5"]), ("10 + 10", r["n10"])):
            a, m = s["auc"], s["mann_whitney"]
            L.append(f"| {d} | {lbl} | {_f(a['roc_auc'])} {_ci(a['roc_ci95'])} | "
                     f"{_f(a['pr_auc'])} {_ci(a['pr_ci95'])} | {m['U']:.0f} | "
                     f"{m['p_one_sided_greater']:.2g} | {_f(m['rank_biserial'], 2)} |")
    L += ["", "The one-sided test asks whether poisoned models have larger ratios than clean "
          "models. With 5 + 5 models the smallest attainable one-sided p is 1/252 ≈ 0.004.",
          "", "## Flag rate and false-positive rate", "",
          "| dataset | n | threshold | flag rate (TPR), poisoned | FPR, disjoint test group | "
          "FPR, leave-one-out | TPR, leave-one-out |", "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for lbl, s in (("5 + 5 (calib 3, test 2)", r["n5"]), ("10 + 10 (calib 5, test 5)", r["n10"])):
            ru = s["rule"]
            L.append(f"| {d} | {lbl} | {_f(ru['threshold'])} | {_rate(ru['tpr'])} | "
                     f"{_rate(ru['fpr_test_group'])} | {_rate(ru['fpr_leave_one_out'])} | "
                     f"{_rate(ru['tpr_leave_one_out'])} |")
    L += ["", "Bootstrap intervals collapse to a point when all flags agree (for example 0 of "
          "5); the exact Clopper-Pearson interval is given for that reason.", ""]
    for d, r in out["datasets"].items():
        n10 = r["n10"]
        t = str(cfg["datasets"][d]["poison"]["target"])
        tm = {role: np.mean([m["l1"][t] for m in r["per_model"].values() if m["role"] == role])
              for role in ("clean", "poisoned")}
        L.append(f"- {d}: mean L1 mask of the target class: clean models {tm['clean']:.2f}, "
                 f"poisoned models {tm['poisoned']:.2f} "
                 f"({'smaller' if tm['poisoned'] < tm['clean'] else 'larger'} under the backdoor).")
        L.append(f"- {d}: on the new models only (5 + 5), ROC-AUC "
                 f"{_f(n10['new_only_auc']['roc_auc'])}; flag rate of the new poisoned models "
                 f"at the calibration threshold {_f(n10['new_poisoned_flag_rate_at_calib_threshold'], 2)}; "
                 f"the target class has the smallest mask in "
                 f"{_f(n10['target_is_smallest_mask_rate'], 2)} of the poisoned models.")
    L += ["", "## Attack success and clean-performance shift", "",
          "ASR: fraction of correctly classified source-class test rows predicted as the "
          "target class once the trigger is stamped (full test split). Shift: poisoned "
          "model minus the clean model of the same seed. Mean over models with a bootstrap "
          "95% CI.", "",
          "| dataset | models | ASR | natural ASR (clean models) | Δ accuracy | Δ attack F1 | "
          "Δ PR-AUC |", "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for lbl, key in (("existing (seeds 42-46)", "existing_n5"), ("new", "new_n5"),
                         ("all 10", "all_n10")):
            a = r["attack"][key]
            L.append(f"| {d} | {lbl} | {_f(a['asr']['mean'])} {_ci(a['asr']['boot_ci95'], 3)} | "
                     f"{_f(a['natural_asr_clean']['mean'])} | "
                     f"{a['shift']['accuracy']['mean']:+.4f} {_ci(a['shift']['accuracy']['boot_ci95'], 4)} | "
                     f"{a['shift']['attack_f1']['mean']:+.4f} {_ci(a['shift']['attack_f1']['boot_ci95'], 4)} | "
                     f"{a['shift']['pr_auc']['mean']:+.4f} {_ci(a['shift']['pr_auc']['boot_ci95'], 4)} |")
    L += ["", "## Per-model statistics", "",
          "| dataset | model | role | L1 mask normal | L1 mask attack | ratio | flagged (n = 10 rule) |",
          "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        thr = r["n10"]["rule"]["threshold"]
        for n, m in r["per_model"].items():
            L.append(f"| {d} | `{n}` | {m['role']} | {m['l1']['0']:.2f} | {m['l1']['1']:.2f} | "
                     f"{m['ratio']:.2f} | {'yes' if m['ratio'] > thr else 'no'} |")
    L += ["", f"Neural Cleanse time per model (both classes): "
          + "; ".join(f"{d} {r['nc_time_per_model_s']['mean']:.1f} s"
                      for d, r in out["datasets"].items())
          + f". Total evaluation runtime {out['runtime_s'] / 60:.1f} min.", "",
          "## Artifacts", "",
          "- `metrics.json`, `summary.md`, `screening_ratios.png`",
          "- `manifest.json` (models evaluated, checkpoint SHA-256, environment) and "
          "`models_manifest.json` (training of the new seeds)"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="v1.1 Experiment A: NC screening with n = 10 + 10")
    ap.add_argument("--config", default="configs/v11_screening.yaml")
    ap.add_argument("--summary-only", action="store_true",
                    help="rewrite summary.md from the existing metrics.json")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = load_yaml(resolve_path(a.config))
    cfg["_config_path"] = a.config
    if a.summary_only:  # regenerate summary.md from the committed metrics.json
        exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
        out = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
        _summary(exp_dir / "summary.md", cfg, out)
        return
    run(cfg)


if __name__ == "__main__":
    main()
