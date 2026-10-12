"""EXPLORATORY diagnostic: Neural Cleanse with a lower attack-success requirement
on UNSW-NB15 (outputs ``experiments/v11_nc_diagnostic/``).

Does not change the paper's Neural Cleanse detection rule (success threshold
0.99). Reruns the target-excluded, count-matched mask optimization of
Experiment A on the same 10 clean + 10 poisoned UNSW-NB15 models with the
reversal-success requirement lowered to 0.90 for both classes, and reports
whether the normal-class (target) mask then shrinks under the backdoor and
whether the target becomes the smallest-mask class.

Run:
    python -u -m src.eval.run_v11_nc_diagnostic --config configs/v11_nc_diagnostic.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.detectors.neural_cleanse import NeuralCleanseDetector
from src.eval.run_v11_screening import count_match_N
from src.eval.v11_common import (
    CPU,
    auc_boot,
    environment,
    git_commit,
    load_model_cpu,
    mann_whitney,
    mean_ci,
    model_names,
    phase7e_holdouts,
    utc_now,
    write_json,
)
from src.utils import load_yaml, resolve_path, set_global_seed


def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    t0 = time.perf_counter()
    mcfg = load_yaml(resolve_path(cfg["models_from"]))
    ref = json.loads(resolve_path(cfg["reference_results"]).read_text(encoding="utf-8"))
    dname = cfg["dataset"]
    dc = mcfg["datasets"][dname]
    names = model_names(dc)
    md = resolve_path(cfg["paths"]["models_dir"])
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    h = phase7e_holdouts(mcfg["holdout_replay_config"])[dname]
    iters, seed, nc = cfg["bootstrap_iters"], cfg["seed"], cfg["nc"]
    N = count_match_N(load_model_cpu(md, names["clean"][0]), h["X"], h["y"], mcfg["nc"], seed)
    target = str(dc["poison"]["target"])

    per = {}
    for role in ("clean", "poisoned"):
        for n in names[role]:
            det = NeuralCleanseDetector(load_model_cpu(md, n), h["X"], h["y"], device=CPU,
                                        steps=nc["steps"], lr=nc["lr"], seed=seed,
                                        success_threshold=nc["success_threshold"])
            preds = det.predict_holdout()
            r = {c: det.reverse_engineer(c, source_only=True, n_source=N, preds=preds)
                 for c in (0, 1)}
            l1 = {str(c): float(r[c]["l1_norm"]) for c in (0, 1)}
            per[n] = {"role": role, "l1": l1,
                      "ratio": max(l1.values()) / (min(l1.values()) + 1e-9),
                      "smallest_mask_class": int(min(l1, key=l1.get)),
                      "reversal_success": {str(c): float(r[c]["reversal_success"]) for c in (0, 1)},
                      "l1_at_0.99": ref["datasets"][dname]["per_model"][n]["l1"]}
            print(f"  {role:8s} {n}: L1 normal {l1['0']:.2f} (0.99: {per[n]['l1_at_0.99']['0']:.2f}) "
                  f"attack {l1['1']:.2f} | success {per[n]['reversal_success']}", flush=True)

    def col(role, key, c):
        return [p[key][c] for p in per.values() if p["role"] == role]

    res = {"label": "EXPLORATORY", "dataset": dname, "count_match_N": N,
           "success_threshold": nc["success_threshold"], "per_model": per,
           "target_mask": {thr: {role: mean_ci(col(role, key, target), iters, seed)
                                 for role in ("clean", "poisoned")}
                           for thr, key in (("0.90", "l1"), ("0.99", "l1_at_0.99"))},
           "nontarget_mask_0.90": {role: mean_ci(col(role, "l1", "1"), iters, seed)
                                   for role in ("clean", "poisoned")},
           "target_mask_poisoned_vs_clean_0.90": mann_whitney(
               col("clean", "l1", target), col("poisoned", "l1", target)),
           "target_is_smallest_mask_rate_poisoned": float(np.mean(
               [p["smallest_mask_class"] == int(target) for p in per.values()
                if p["role"] == "poisoned"])),
           "target_is_smallest_mask_rate_clean": float(np.mean(
               [p["smallest_mask_class"] == int(target) for p in per.values()
                if p["role"] == "clean"])),
           "ratio_auc_0.90": auc_boot([p["ratio"] for p in per.values() if p["role"] == "clean"],
                                      [p["ratio"] for p in per.values() if p["role"] == "poisoned"],
                                      iters, seed),
           "min_reversal_success_poisoned_target": float(min(
               p["reversal_success"][target] for p in per.values() if p["role"] == "poisoned")),
           "runtime_s": time.perf_counter() - t0}
    write_json(exp_dir / "metrics.json", res)
    write_json(exp_dir / "manifest.json", {
        "experiment": exp_dir.name, "label": "EXPLORATORY", "config": cfg.get("_config_path"),
        "command": f"python -u -m src.eval.run_v11_nc_diagnostic --config {cfg.get('_config_path')}",
        "evaluated_at": utc_now(), "git_commit": git_commit(), "environment": environment(),
        "models": names})
    _summary(exp_dir / "summary.md", cfg, res)
    print(f"  wrote artifacts -> {exp_dir}", flush=True)
    return res


def _summary(path, cfg, r):
    tm = r["target_mask"]
    L = ["# EXPLORATORY diagnostic: Neural Cleanse success requirement on UNSW-NB15", "",
         "**Exploratory analysis.** It does not change the paper's Neural Cleanse detection rule "
         "(success requirement 0.99), and no number here replaces a screening result.", "",
         f"- **Run:** `python -u -m src.eval.run_v11_nc_diagnostic --config {cfg.get('_config_path')}`",
         f"- **Setting:** the 10 clean and 10 normal-target poisoned UNSW-NB15 MLP models of "
         f"Experiment A, the phase7e holdout, target-excluded count-matched Neural Cleanse "
         f"(N = {r['count_match_N']}), 500 steps; the reversal-success requirement is lowered from "
         f"0.99 to {r['success_threshold']:.2f} for both classes.", "",
         "## Target-class (normal) mask", "",
         "| success requirement | clean models | poisoned models | direction under the backdoor |",
         "|---|---|---|---|"]
    for thr in ("0.99", "0.90"):
        c, p = tm[thr]["clean"], tm[thr]["poisoned"]
        L.append(f"| {thr} | {c['mean']:.2f} [{c['boot_ci95'][0]:.2f}, {c['boot_ci95'][1]:.2f}] | "
                 f"{p['mean']:.2f} [{p['boot_ci95'][0]:.2f}, {p['boot_ci95'][1]:.2f}] | "
                 f"{'smaller' if p['mean'] < c['mean'] else 'larger'} |")
    m = r["target_mask_poisoned_vs_clean_0.90"]
    nt = r["nontarget_mask_0.90"]
    L += ["", f"At {r['success_threshold']:.2f}: Mann-Whitney U (poisoned > clean) = {m['U']:.0f}, "
          f"one-sided p = {m['p_one_sided_greater']:.2g}, two-sided p = {m['p_two_sided']:.2g}, "
          f"rank-biserial r = {m['rank_biserial']:.2f}. Attack-class mask: clean "
          f"{nt['clean']['mean']:.2f}, poisoned {nt['poisoned']['mean']:.2f}.", "",
          "## Is the target identifiable?", "",
          f"- Target (normal) is the smallest-mask class in **{r['target_is_smallest_mask_rate_poisoned']:.2f}** "
          f"of the poisoned models (clean models: {r['target_is_smallest_mask_rate_clean']:.2f}).",
          f"- Lowest reversal success reached for the target class in a poisoned model: "
          f"{r['min_reversal_success_poisoned_target']:.3f}.",
          f"- Mask-ratio separation at {r['success_threshold']:.2f} (descriptive only): ROC-AUC "
          f"{r['ratio_auc_0.90']['roc_auc']:.3f} [{r['ratio_auc_0.90']['roc_ci95'][0]:.2f}, "
          f"{r['ratio_auc_0.90']['roc_ci95'][1]:.2f}].", "",
          "## Per-model masks", "",
          "| model | role | L1 normal (0.99) | L1 normal (0.90) | L1 attack (0.90) | smallest-mask class (0.90) |",
          "|---|---|---|---|---|---|"]
    for n, p in r["per_model"].items():
        L.append(f"| `{n}` | {p['role']} | {p['l1_at_0.99']['0']:.2f} | {p['l1']['0']:.2f} | "
                 f"{p['l1']['1']:.2f} | {'normal' if p['smallest_mask_class'] == 0 else 'attack'} |")
    L += ["", "## Artifacts", "", "- `metrics.json`, `summary.md`, `manifest.json`"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="EXPLORATORY NC success-requirement diagnostic")
    ap.add_argument("--config", default="configs/v11_nc_diagnostic.yaml")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = load_yaml(resolve_path(a.config))
    cfg["_config_path"] = a.config
    run(cfg)


if __name__ == "__main__":
    main()
