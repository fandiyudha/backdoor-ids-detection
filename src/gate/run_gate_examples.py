"""Run the screening-gate prototype on example candidates (outputs ``experiments/v11_gate/``).

Writes one JSON verdict per candidate and a summary. With the 5 calibration clean
models as references, the gate's per-detector flags must equal the per-model
verdicts of Experiments A/B (MLP) and D (1D-CNN); the summary reports this check.

Run:
    python -u -m src.gate.run_gate_examples --config configs/v11_gate.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.eval.v11_common import git_commit, utc_now, write_json
from src.gate.screen_model import screen
from src.utils import load_yaml, resolve_path


def _expected(cfg, ex, name):
    src = cfg["experiment_metrics"][ex["arch"]]
    nc = json.loads(resolve_path(src["nc"]).read_text(encoding="utf-8"))["datasets"][ex["dataset"]]
    mm = json.loads(resolve_path(src["mmbd"]).read_text(encoding="utf-8"))["datasets"][ex["dataset"]]
    row = next(r for r in mm["agreement_with_nc"]["per_model"] if r["model"] == name)
    return {"nc_flag": row["nc_flag"], "mmbd_flag": row["mmbd_flag"], "role": row["role"],
            "group": row["group"]}


def run(cfg):
    md = resolve_path(cfg["paths"]["models_dir"])
    exp = resolve_path(cfg["paths"]["experiment_dir"])
    cache_dir = resolve_path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ex in cfg["examples"]:
        refs = [md / f"{n}.pt" for n in ex["references"]]
        cache = cache_dir / f"refs_{ex['dataset']}_{ex['arch']}.json"
        for cand in ex["candidates"]:
            print(f"\n== {ex['dataset']} / {ex['arch']}: {cand}", flush=True)
            v = screen(md / f"{cand}.pt", refs, f"{cfg['holdout']}:{ex['dataset']}", ex["dataset"],
                       cache, log=lambda m: print("  " + m, flush=True))
            write_json(exp / "verdicts" / f"{ex['dataset']}_{cand}.json", v)
            e = _expected(cfg, ex, cand)
            d = v["detectors"]
            rows.append({"dataset": ex["dataset"], "arch": ex["arch"], "candidate": cand,
                         "truth": e["role"], "group": e["group"],
                         "nc_statistic": d["neural_cleanse"]["statistic"],
                         "nc_threshold": d["neural_cleanse"]["threshold"], "nc_flag": d["neural_cleanse"]["flag"],
                         "mmbd_statistic": d["mmbd"]["statistic"], "mmbd_flag": d["mmbd"]["flag"],
                         "either": v["decision"]["either_detector_flags"],
                         "both": v["decision"]["both_detectors_flag"],
                         "matches_experiment": e["nc_flag"] == d["neural_cleanse"]["flag"]
                         and e["mmbd_flag"] == d["mmbd"]["flag"],
                         "context": v["context"]["per_class"], "runtime_s": v["runtime_s"],
                         "measured": v["measured_performance"]})
            print(f"  NC {rows[-1]['nc_statistic']:.2f} (thr {rows[-1]['nc_threshold']:.2f}) flag "
                  f"{rows[-1]['nc_flag']} | MM-BD S {rows[-1]['mmbd_statistic']:.2f} flag {rows[-1]['mmbd_flag']} "
                  f"| matches experiment {rows[-1]['matches_experiment']}", flush=True)
    out = {"examples": rows, "all_match_experiments": all(r["matches_experiment"] for r in rows),
           "evaluated_at": utc_now(), "git_commit": git_commit(), "config": cfg.get("_config_path")}
    write_json(exp / "metrics.json", out)
    _summary(exp / "summary.md", cfg, out)
    print(f"\n  all flags match the experiments: {out['all_match_experiments']}", flush=True)
    return out


def _summary(path, cfg, out):
    L = ["# Extended experiment E: model-intake screening gate (prototype)", "",
         f"- **Run:** `python -u -m src.gate.run_gate_examples --config {cfg.get('_config_path')}`; "
         "single candidate: `python -m src.gate.screen_model --candidate <ckpt> --references <folder> "
         "--holdout phase7e:<dataset> --dataset <dataset> --out verdict.json`.",
         "- **What it does:** runs the two model-level detectors (Neural Cleanse, paper protocol; MM-BD, "
         "Experiment B) on a candidate checkpoint with a clean holdout and a set of clean reference models, "
         "applies their reference-relative rules (2.0 x the reference median), and combines them with an "
         "\"either detector flags\" and a \"both detectors flag\" rule. Each verdict carries the TPR/FPR "
         "measured for the same dataset and architecture in Experiments A, B, and D.",
         "- **Context (not a decision):** each verdict also lists, per class, the reference models' median "
         "and range of the minimal NC mask norm and of the MM-BD margin, and the candidate's value relative "
         "to them. No threshold has been validated for these values and they do not enter the decision.",
         "- **Status:** a prototype for future work, not a product. References here are the 5 calibration "
         "clean models of each experiment, so the per-detector flags must reproduce the experiments' "
         f"per-model verdicts: **all match: {out['all_match_experiments']}**.", "",
         "## Example verdicts", "",
         "| dataset | architecture | candidate | truth | NC ratio (threshold) | NC flag | MM-BD S (threshold 2.0) | "
         "MM-BD flag | either | both | matches experiment |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["examples"]:
        L.append(f"| {r['dataset']} | {r['arch']} | `{r['candidate']}` | {r['truth']} | "
                 f"{r['nc_statistic']:.2f} ({r['nc_threshold']:.2f}) | {'yes' if r['nc_flag'] else 'no'} | "
                 f"{r['mmbd_statistic']:.2f} | {'yes' if r['mmbd_flag'] else 'no'} | "
                 f"{'yes' if r['either'] else 'no'} | {'yes' if r['both'] else 'no'} | {r['matches_experiment']} |")
    L += ["", "## Measured operating points attached to the decision", "",
          "| dataset | architecture | rule | TPR (exact 95% CI) | FPR, clean test group (exact 95% CI) |",
          "|---|---|---|---|---|"]
    seen = set()
    for r in out["examples"]:
        k = (r["dataset"], r["arch"])
        if k in seen or not r["measured"].get("available"):
            continue
        seen.add(k)
        for rule, lbl in (("neural_cleanse", "Neural Cleanse"), ("mmbd", "MM-BD"),
                          ("either_flags", "either detector flags"), ("both_flag", "both detectors flag")):
            t, f = r["measured"][rule]["tpr"], r["measured"][rule]["fpr_test_group"]
            L.append(f"| {r['dataset']} | {r['arch']} | {lbl} | {t['k']}/{t['n']} = {t['rate']:.2f} "
                     f"[{t['clopper_pearson95'][0]:.2f}, {t['clopper_pearson95'][1]:.2f}] | {f['k']}/{f['n']} = "
                     f"{f['rate']:.2f} [{f['clopper_pearson95'][0]:.2f}, {f['clopper_pearson95'][1]:.2f}] |")
    L += ["", "## Context (not a decision): natural per-class statistics of the references", "",
          "Reference median [min, max] over the clean reference models, and the candidate's value relative "
          "to the median. Class 0 = benign/normal (the backdoor target), class 1 = attack.", "",
          "| dataset | architecture | candidate | class | NC mask L1: reference median [range] | candidate / median | "
          "MM-BD margin: reference median [range] | candidate / median |", "|---|---|---|---|---|---|---|---|"]
    for r in out["examples"]:
        for c in ("0", "1"):
            n = r["context"][f"class_{c}"]["nc_minimal_mask_l1"]
            m = r["context"][f"class_{c}"]["mmbd_margin"]
            L.append(f"| {r['dataset']} | {r['arch']} | `{r['candidate']}` | {c} | {n['reference_median']:.2f} "
                     f"[{n['reference_min']:.2f}, {n['reference_max']:.2f}] | {n['candidate_over_reference_median']:.2f} | "
                     f"{m['reference_median']:.1f} [{m['reference_min']:.1f}, {m['reference_max']:.1f}] | "
                     f"{m['candidate_over_reference_median']:.2f} |")
    L += ["", f"Runtime per candidate after the reference statistics are cached: "
          f"{np.median([r['runtime_s'] for r in out['examples']]):.0f} s (median; reference statistics are "
          "computed once per dataset and architecture and cached under `results/gate_cache/`).", "",
          "## Artifacts", "", "- `metrics.json`, `summary.md`, `verdicts/<dataset>_<candidate>.json`"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Screening-gate prototype examples")
    ap.add_argument("--config", default="configs/v11_gate.yaml")
    a = ap.parse_args()
    cfg = load_yaml(resolve_path(a.config))
    cfg["_config_path"] = a.config
    run(cfg)


if __name__ == "__main__":
    main()
