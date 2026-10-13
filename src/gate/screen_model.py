"""Model-intake screening gate (prototype for future work, not a product).

Screens one candidate IDS model before deployment with the two model-level
detectors of the v1.1 experiments and their calibrated reference-relative rules:

* Neural Cleanse (target-excluded, count-matched; paper protocol): flag if the
  candidate's mask-norm ratio max/min exceeds 2.0 x the median ratio of the clean
  reference models.
* MM-BD (min/max box from the clean holdout, holdout-seeded starts; Experiment B):
  rel_c = r_c / median over references of r_c, S = max_c rel_c; flag if S > 2.0.
  The largest leave-one-out S among the references (Experiment B's secondary
  rule) is reported as well, but is not part of the decision.

Overall decision under two combination rules, "either detector flags" and "both
detectors flag", each reported with the TPR/FPR measured for the same dataset and
architecture in Experiments A, B (MLP) and D (1D-CNN). Those rates were measured
with 5 calibration reference models and 10 + 10 test models; with other
reference sets they are indicative only.

A separate "context (not a decision)" section gives, per class, the reference
models' natural statistics (median and range of the minimal NC mask norm and of
the MM-BD margin) and the candidate's value relative to them. No threshold has
been validated for these values; they are reported to support future work on
anticipating low-contrast regimes (e.g. the CICIDS2017 1D-CNN, where the clean
benign mask is already near zero).

Usage:
    python -m src.gate.screen_model --candidate results/models/<name>.pt \\
        --references <folder or comma-separated .pt files> \\
        --holdout phase7e:CICIDS2017 | <file.npz with arrays X, y> \\
        --dataset CICIDS2017 [--out verdict.json] [--cache refs.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.detectors.base import MMBDScreen, NeuralCleanseScreen
from src.eval.v11_common import environment, sha256_file, utc_now
from src.models.registry import load_model
from src.utils import resolve_path

REL_THRESHOLD = 2.0
SETTINGS = {"neural_cleanse": {"steps": 500, "lr": 0.1, "seed": 42, "success_threshold": 0.99},
            "mmbd": {"box": "minmax", "n_starts": 50, "steps": 500, "lr": 0.02, "seed": 42,
                     "holdout_starts": 50},
            "rel_threshold": REL_THRESHOLD}
# where the measured operating points of each (dataset, architecture) live
MEASURED = {"mlp": {"nc": "experiments/v11_screening/metrics.json",
                    "mmbd": "experiments/v11_mmbd/metrics.json"},
            "cnn1d": {"nc": "experiments/v11_cnn/nc/metrics.json",
                      "mmbd": "experiments/v11_cnn/mmbd/metrics.json"}}
DISCLAIMER = ("Prototype for future work, not a validated product. Measured rates come from small "
              "samples (10 clean + 10 poisoned models per dataset and architecture, one trigger type, "
              "one poison ratio) and do not transfer to other attacks, datasets, or architectures "
              "without new validation.")


# ---------------------------------------------------------------------------
def load_holdout(spec: str):
    if spec.startswith("phase7e:"):
        from src.eval.v11_common import phase7e_holdouts

        d = spec.split(":", 1)[1]
        h = phase7e_holdouts("configs/phase7e.yaml")[d]
        return h["X"], h["y"], f"phase7e holdout of {d} ({h['source_split']}, {len(h['X'])} rows)"
    z = np.load(spec)
    return z["X"].astype(np.float32), z["y"].astype(int), f"{spec} ({len(z['X'])} rows)"


def list_references(spec: str) -> List[Path]:
    p = Path(spec)
    if p.is_dir():
        refs = sorted(p.glob("*.pt"))
    else:
        refs = [Path(s) for s in spec.split(",") if s]
    if len(refs) < 3:
        raise SystemExit("at least 3 clean reference models are needed")
    return refs


def _arch(model) -> str:
    return model.config().get("arch", "mlp")


def _holdout_hash(X, y) -> str:
    return hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.asarray(y, np.int64).tobytes()).hexdigest()


def score(model, holdout, N) -> dict:
    nc = NeuralCleanseScreen(N, **SETTINGS["neural_cleanse"]).score_model(model, holdout)
    mm = MMBDScreen(**SETTINGS["mmbd"]).score_model(model, holdout)
    return {"nc_ratio": nc["statistic"], "nc_mask_l1": nc["per_class"]["mask_l1"],
            "mmbd_margin": mm["per_class"]["margin"]}


def reference_stats(refs: List[Path], holdout, cache: Path | None, log) -> dict:
    hh = _holdout_hash(*holdout)
    shas = [sha256_file(r) for r in refs]
    key = {"holdout_sha256": hh, "references_sha256": shas, "settings": SETTINGS}
    if cache and cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("key") == key:
            log(f"reference statistics loaded from cache {cache}")
            return c
    m0 = load_model(refs[0])
    N = NeuralCleanseScreen.count_match_n(m0, holdout, seed=SETTINGS["neural_cleanse"]["seed"])
    per = {}
    for r, s in zip(refs, shas):
        t = time.perf_counter()
        per[r.name] = {**score(load_model(r), holdout, N), "sha256": s}
        log(f"reference {r.name}: NC ratio {per[r.name]['nc_ratio']:.2f} ({time.perf_counter() - t:.1f}s)")
    out = {"key": key, "count_match_N": N, "per_reference": per, "arch": _arch(m0)}
    if cache:
        cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _measured(dataset: str, arch: str) -> dict:
    src = MEASURED.get(arch)
    if not src:
        return {"available": False, "note": f"no measurements for architecture {arch!r}"}
    try:
        nc = json.loads(resolve_path(src["nc"]).read_text(encoding="utf-8"))["datasets"][dataset]["n10"]["rule"]
        mm = json.loads(resolve_path(src["mmbd"]).read_text(encoding="utf-8"))["datasets"][dataset]
    except (FileNotFoundError, KeyError):
        return {"available": False, "note": f"no measurements for {dataset} / {arch}"}
    prim = mm["boxes"]["minmax/seeded"]["primary_rule"]
    ag = mm["agreement_with_nc"]

    def rate(r):
        return {"rate": r["rate"], "k": r["k"], "n": r["n"], "clopper_pearson95": r["clopper_pearson95"]}

    return {"available": True, "sources": src,
            "neural_cleanse": {"tpr": rate(nc["tpr"]), "fpr_test_group": rate(nc["fpr_test_group"])},
            "mmbd": {"tpr": rate(prim["tpr"]), "fpr_test_group": rate(prim["fpr_test_group"])},
            "either_flags": {"tpr": rate(ag["either_flags"]["tpr"]),
                             "fpr_test_group": rate(ag["either_flags"]["fpr_test_group"])},
            "both_flag": {"tpr": rate(ag["both_flag"]["tpr"]),
                          "fpr_test_group": rate(ag["both_flag"]["fpr_test_group"])},
            "note": "measured with 5 calibration clean reference models; 10 poisoned models (TPR) and "
                    "5 disjoint clean test models (FPR); exact Clopper-Pearson 95% intervals"}


def _context(ref: dict, cand: dict) -> dict:
    out = {}
    for c in ("0", "1"):
        row = {}
        for key, lbl in (("nc_mask_l1", "nc_minimal_mask_l1"), ("mmbd_margin", "mmbd_margin")):
            v = np.array([p[key][c] for p in ref["per_reference"].values()], float)
            med = float(np.median(v))
            row[lbl] = {"reference_median": med, "reference_min": float(v.min()),
                        "reference_max": float(v.max()), "candidate": float(cand[key][c]),
                        "candidate_over_reference_median": float(cand[key][c] / med) if med else None,
                        "candidate_outside_reference_range": bool(cand[key][c] < v.min() or cand[key][c] > v.max())}
        out[f"class_{c}"] = row
    return {"label": "context (not a decision)",
            "note": "Natural per-class statistics of the clean reference models and the candidate's value "
                    "relative to them. No threshold has been validated for these values; they are not used "
                    "in the decision. A reference median near zero for a class (e.g. the benign mask of the "
                    "CICIDS2017 1D-CNN) marks a low-contrast regime in which a backdoor targeting that class "
                    "has little room to change the statistic.",
            "per_class": out}


def screen(candidate: Path, refs: List[Path], holdout_spec: str, dataset: str,
           cache: Path | None = None, log=print) -> dict:
    t0 = time.perf_counter()
    X, y, hdesc = load_holdout(holdout_spec)
    holdout = (X, y)
    ref = reference_stats(refs, holdout, cache, log)
    cm = load_model(candidate)
    arch = _arch(cm)
    if arch != ref["arch"]:
        log(f"warning: candidate architecture {arch} differs from the references ({ref['arch']})")
    cand = score(cm, holdout, ref["count_match_N"])

    # Neural Cleanse rule
    rr = np.array([p["nc_ratio"] for p in ref["per_reference"].values()], float)
    nc_thr = REL_THRESHOLD * float(np.median(rr))
    nc_flag = bool(cand["nc_ratio"] > nc_thr)
    # MM-BD rule
    R = np.array([[p["mmbd_margin"]["0"], p["mmbd_margin"]["1"]] for p in ref["per_reference"].values()], float)
    med = np.median(R, axis=0)
    c_r = np.array([cand["mmbd_margin"]["0"], cand["mmbd_margin"]["1"]], float)
    valid = bool((med > 0).all() and (c_r > 0).all() and (R > 0).all())
    if valid:
        rel = c_r / med
        S, tgt = float(rel.max()), int(rel.argmax())
        loo = [float((R[i] / np.median(np.delete(R, i, axis=0), axis=0)).max()) for i in range(len(R))]
        mm_flag = bool(S > REL_THRESHOLD)
    else:
        rel, S, tgt, loo, mm_flag = None, None, None, [], False

    verdict = {
        "gate": "model-intake screening gate (prototype)", "evaluated_at": utc_now(),
        "candidate": {"path": str(candidate), "sha256": sha256_file(candidate), "architecture": arch},
        "dataset": dataset, "holdout": hdesc,
        "references": {"n": len(refs), "count_match_N": ref["count_match_N"],
                       "models": {k: v["sha256"] for k, v in ref["per_reference"].items()}},
        "settings": SETTINGS,
        "detectors": {
            "neural_cleanse": {"statistic": cand["nc_ratio"], "statistic_name": "mask-norm ratio max/min",
                               "threshold": nc_thr, "rule": f"flag if ratio > {REL_THRESHOLD} x median of "
                                                            "the reference ratios",
                               "flag": nc_flag, "smallest_mask_class": int(min(cand["nc_mask_l1"],
                                                                              key=cand["nc_mask_l1"].get))},
            "mmbd": {"statistic": S, "statistic_name": "S = max_c r_c / reference median of r_c",
                     "threshold": REL_THRESHOLD, "rule": f"flag if S > {REL_THRESHOLD}",
                     "flag": mm_flag, "predicted_target_class": tgt,
                     "relative_margins": None if rel is None else {"0": float(rel[0]), "1": float(rel[1])},
                     "valid": valid,
                     "secondary_threshold_reported_only": max(loo) if loo else None}},
        "decision": {"either_detector_flags": nc_flag or mm_flag, "both_detectors_flag": nc_flag and mm_flag},
        "measured_performance": _measured(dataset, arch),
        "context": _context(ref, cand),
        "disclaimer": DISCLAIMER, "runtime_s": time.perf_counter() - t0,
        "environment": environment()}
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description="Model-intake screening gate (prototype)")
    ap.add_argument("--candidate", required=True, help="candidate checkpoint (.pt)")
    ap.add_argument("--references", required=True,
                    help="folder of clean reference checkpoints, or comma-separated .pt files")
    ap.add_argument("--holdout", required=True,
                    help="'phase7e:<dataset>' or an .npz file with standardized arrays X and y")
    ap.add_argument("--dataset", required=True, choices=["CICIDS2017", "UNSW-NB15"],
                    help="selects the measured operating points reported with the decision")
    ap.add_argument("--out", default=None, help="write the JSON verdict here (default: stdout)")
    ap.add_argument("--cache", default=None, help="JSON cache for the reference statistics")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(True)
    log = (lambda m: print(m, file=sys.stderr, flush=True))
    v = screen(Path(a.candidate), list_references(a.references), a.holdout, a.dataset,
               Path(a.cache) if a.cache else None, log)
    s = json.dumps(v, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(s, encoding="utf-8")
        log(f"verdict -> {a.out}: either={v['decision']['either_detector_flags']} "
            f"both={v['decision']['both_detectors_flag']}")
    else:
        print(s)
    return v


if __name__ == "__main__":
    main()
