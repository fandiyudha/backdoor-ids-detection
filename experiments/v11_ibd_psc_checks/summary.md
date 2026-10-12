# Follow-up checks to extended experiment C

- **Run:** `python -u -m src.eval.run_v11_ibd_psc_checks --config configs/v11_ibd_psc_checks.yaml`
- **Scope:** three secondary analyses on the models, holdouts, positives, and negatives of Experiment C (which replays the phase7f draws). None replaces a result of Experiment C.
- **Reproduction:** STRIP and primary IBD-PSC scores recomputed here match Experiment C: CICIDS2017 max abs. difference 0.0e+00; UNSW-NB15 max abs. difference 0.0e+00.

## 1. Sensitivity analysis: ξ-faithful IBD-PSC

All BN layers amplified; ω increased over [1.5, 2, 3, 5, 8, 12, 20, 30, 50] until the error rate on the first 100 clean holdout rows (against the true labels) exceeds ξ = 0.6; ω* = the first such value. Five amplified models scale all BN layers by ω* × [1.0, 1.25, 1.5, 1.75, 2.0].

- CICIDS2017: ω* per model [None, None, None, None, None, None, None, None, None, None]; ξ reached in 0/10 models.
- CICIDS2017: clean error rate by ω (lowest–highest over models): ω 1.5: 0.03–0.07, ω 2: 0.03–0.09, ω 3: 0.03–0.09, ω 5: 0.03–0.10, ω 8: 0.03–0.11, ω 12: 0.03–0.11, ω 20: 0.03–0.12, ω 30: 0.03–0.12, ω 50: 0.03–0.12.
- UNSW-NB15: ω* per model [None, None, None, None, None, None, None, None, None, None]; ξ reached in 0/10 models.
- UNSW-NB15: clean error rate by ω (lowest–highest over models): ω 1.5: 0.10–0.15, ω 2: 0.11–0.15, ω 3: 0.11–0.15, ω 5: 0.11–0.15, ω 8: 0.11–0.15, ω 12: 0.11–0.15, ω 20: 0.12–0.15, ω 30: 0.12–0.15, ω 50: 0.12–0.16.

| dataset | models | detector | n | ROC-AUC | TPR @ 5% FPR | realized FPR | TPR @ T = 0.9 | FPR @ T = 0.9 |
|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | existing 5 | IBD-PSC (Experiment C) | 5 | 0.326 [0.283, 0.372] | 0.001 [0.000, 0.004] | 0.013 | 0.979 [0.972, 0.984] | 0.951 [0.947, 0.953] |
| CICIDS2017 | existing 5 | IBD-PSC (ξ-faithful) | 0 | ξ not reached | – | – | – | – |
| CICIDS2017 | all 10 | IBD-PSC (Experiment C) | 10 | 0.373 [0.322, 0.431] | 0.001 [0.000, 0.003] | 0.015 | 0.976 [0.968, 0.982] | 0.952 [0.949, 0.955] |
| CICIDS2017 | all 10 | IBD-PSC (ξ-faithful) | 0 | ξ not reached | – | – | – | – |
| UNSW-NB15 | existing 5 | IBD-PSC (Experiment C) | 5 | 0.221 [0.203, 0.241] | 0.003 [0.003, 0.004] | 0.052 | 0.595 [0.529, 0.683] | 0.793 [0.787, 0.797] |
| UNSW-NB15 | existing 5 | IBD-PSC (ξ-faithful) | 0 | ξ not reached | – | – | – | – |
| UNSW-NB15 | all 10 | IBD-PSC (Experiment C) | 10 | 0.204 [0.185, 0.221] | 0.003 [0.003, 0.003] | 0.048 | 0.515 [0.426, 0.596] | 0.792 [0.788, 0.795] |
| UNSW-NB15 | all 10 | IBD-PSC (ξ-faithful) | 0 | ξ not reached | – | – | – | – |

CICIDS2017 controlled conditions, ξ-faithful variant (5 models each):

| condition | target | ω* per model | ROC-AUC | TPR @ 5% FPR | TPR @ T | FPR @ T | clean target-class PSC |
|---|---|---|---|---|---|---|---|
| A_balanced_tgtBenign | benign | [None, None, None, None, None] | ξ not reached | – | – | – | – |
| B_imbal_tgtBenign | benign | [None, None, None, None, None] | ξ not reached | – | – | – | – |
| C_imbal_tgtAttack | attack | [None, None, None, None, None] | ξ not reached | – | – | – | – |

## 2. Exploratory: two-sided percentile rule

Flag an input if its score lies outside the [2.5th, 97.5th] percentile range of the clean-holdout scores (nominal FPR 5%). Mean over models with a bootstrap 95% CI.

| dataset | detector | n | TPR (two-sided) | of which below / above the range | realized FPR (two-sided) |
|---|---|---|---|---|---|
| CICIDS2017 | STRIP (α 0.7) | 10 | 0.012 [0.008, 0.016] | 0.004 / 0.007 | 0.042 [0.040, 0.045] |
| CICIDS2017 | IBD-PSC (Experiment C) | 10 | 0.016 [0.011, 0.020] | 0.016 / 0.000 | 0.024 [0.021, 0.026] |
| CICIDS2017 | IBD-PSC (ξ-faithful) | 0 | ξ not reached | – | – |
| UNSW-NB15 | STRIP (α 0.7) | 10 | 0.056 [0.041, 0.070] | 0.003 / 0.052 | 0.049 [0.046, 0.052] |
| UNSW-NB15 | IBD-PSC (Experiment C) | 10 | 0.117 [0.080, 0.156] | 0.115 / 0.002 | 0.040 [0.032, 0.046] |
| UNSW-NB15 | IBD-PSC (ξ-faithful) | 0 | ξ not reached | – | – |

## 3. UNSW-NB15 holdout/negative overlap

- CICIDS2017: 0 of the negatives also appear in the holdout (holdout drawn from `X_val`).
- UNSW-NB15: 51 of the negatives also appear in the holdout (holdout drawn from `X_test`).

Per-input scores and holdout thresholds are unchanged; only the negative set shrinks. Positive-only metrics (TPR) are therefore identical.

| dataset | models | detector | metric | with overlap | de-duplicated |
|---|---|---|---|---|---|
| UNSW-NB15 | existing 5 | STRIP (α 0.7) | ROC-AUC | 0.2354 | 0.2342 |
| UNSW-NB15 | existing 5 | STRIP (α 0.7) | FPR at the 5% threshold | 0.0473 | 0.0483 |
| UNSW-NB15 | existing 5 | STRIP (α 0.7) | FPR, two-sided rule | 0.0460 | 0.0467 |
| UNSW-NB15 | existing 5 | IBD-PSC (Experiment C) | ROC-AUC | 0.2212 | 0.2191 |
| UNSW-NB15 | existing 5 | IBD-PSC (Experiment C) | FPR at the 5% threshold | 0.0523 | 0.0527 |
| UNSW-NB15 | existing 5 | IBD-PSC (Experiment C) | FPR, two-sided rule | 0.0460 | 0.0450 |
| UNSW-NB15 | existing 5 | IBD-PSC (Experiment C) | FPR at T = 0.9 | 0.7929 | 0.7956 |
| UNSW-NB15 | all 10 | STRIP (α 0.7) | ROC-AUC | 0.2453 | 0.2441 |
| UNSW-NB15 | all 10 | STRIP (α 0.7) | FPR at the 5% threshold | 0.0472 | 0.0480 |
| UNSW-NB15 | all 10 | STRIP (α 0.7) | FPR, two-sided rule | 0.0491 | 0.0495 |
| UNSW-NB15 | all 10 | IBD-PSC (Experiment C) | ROC-AUC | 0.2035 | 0.2017 |
| UNSW-NB15 | all 10 | IBD-PSC (Experiment C) | FPR at the 5% threshold | 0.0485 | 0.0488 |
| UNSW-NB15 | all 10 | IBD-PSC (Experiment C) | FPR, two-sided rule | 0.0397 | 0.0387 |
| UNSW-NB15 | all 10 | IBD-PSC (Experiment C) | FPR at T = 0.9 | 0.7915 | 0.7940 |

Total runtime 0.7 min.

## Artifacts

- `metrics.json`, `summary.md`, `manifest.json`, `psc_distributions_xi_faithful.png`
