"""Check that the detector wrappers in ``src.detectors.base`` reproduce committed outputs.

* NeuralCleanseScreen: per-class mask L1 of two MLP models vs
  ``experiments/v11_screening/metrics.json``;
* MMBDScreen: per-class margins (min/max box, seeded) of two MLP models vs
  ``experiments/v11_mmbd/metrics.json``;
* StripRuntime: per-input entropies of the first 5,000 clean benign test rows vs
  ``experiments/phase7f/fig3_entropy_raw.csv`` (the article's Figure 2 data);
* IBDPSCRuntime: per-input PSC scores of the first 5,000 clean benign test rows
  vs ``experiments/v11_ibd_psc/psc_raw_CICIDS2017.csv``.

Needs the preprocessed data and the checkpoints. Run:
    python -m tests.check_detector_wrappers
"""

from __future__ import annotations

import csv
import json

import numpy as np

from src.detectors.base import IBDPSCRuntime, MMBDScreen, NeuralCleanseScreen, StripRuntime
from src.eval.v11_common import load_model_cpu, phase7e_holdouts
from src.models.train import load_dataset
from src.utils import resolve_path

MD = resolve_path("results/models")


def _csv_group(path, group, ndigits):
    with open(resolve_path(path), encoding="utf-8") as fh:
        vals = [float(r[1]) for r in csv.reader(fh) if r and r[0] == group]
    return np.asarray(vals), ndigits


def main() -> None:
    ok = True
    h = phase7e_holdouts("configs/phase7e.yaml")["CICIDS2017"]
    hold = (h["X"], h["y"])
    scr = json.loads(resolve_path("experiments/v11_screening/metrics.json").read_text(encoding="utf-8"))
    mm = json.loads(resolve_path("experiments/v11_mmbd/metrics.json").read_text(encoding="utf-8"))
    N = scr["datasets"]["CICIDS2017"]["count_match_N"]
    nc, mb = NeuralCleanseScreen(N), MMBDScreen()
    for name in ("causal_ids_B_imbal_tgtBenign_clean_seed43", "v11_mlp_cicids_pois_tgt0_seed101"):
        m = load_model_cpu(MD, name)
        got = nc.score_model(m, hold)["per_class"]["mask_l1"]
        ref = scr["datasets"]["CICIDS2017"]["per_model"][name]["l1"]
        same = got == ref
        ok &= same
        print(f"NeuralCleanseScreen {name}: identical={same}")
        got = mb.score_model(m, hold)["per_class"]["margin"]
        ref = mm["datasets"]["CICIDS2017"]["boxes"]["minmax/seeded"]["per_model"][name]["margins"]
        same = got == ref
        ok &= same
        print(f"MMBDScreen {name}: identical={same}")

    # runtime wrappers: phase7f / Experiment C draws on CICIDS2017
    data = load_dataset(resolve_path("data/processed"))
    rng = np.random.default_rng(42)
    hidx = rng.choice(len(data["X_val"]), 3000, replace=False)
    holdout, holdout_y = data["X_val"][hidx].astype(np.float32), data["y_val"][hidx].astype(int)
    cb = data["X_test"][np.where(data["y_test"] == 0)[0][:5000]].astype(np.float32)
    rep = load_model_cpu(MD, "causal_ids_B_imbal_tgtBenign_seed42")

    e = StripRuntime(holdout).score_inputs(rep, cb)
    ref, nd = _csv_group("experiments/phase7f/fig3_entropy_raw.csv", "clean_benign", 6)
    same = bool(np.array_equal(np.round(e, nd), np.round(ref, nd)))
    ok &= same
    print(f"StripRuntime (phase7f Figure 2 clean benign, n={len(ref)}): identical to 6 decimals={same}")

    s = IBDPSCRuntime(holdout[:100], holdout_y[:100]).score_inputs(rep, cb)
    ref, nd = _csv_group("experiments/v11_ibd_psc/psc_raw_CICIDS2017.csv", "clean_benign", 10)
    same = bool(np.array_equal(np.round(s, nd), np.round(ref, nd)))
    ok &= same
    print(f"IBDPSCRuntime (Experiment C clean benign, n={len(ref)}): identical to 10 decimals={same}")

    print("ALL IDENTICAL" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
