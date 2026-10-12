# Extended experiment B: MM-BD maximum-margin detection

- **Run:** `python -u -m src.eval.run_v11_mmbd --config configs/v11_cnn_mmbd.yaml`
- **Reference:** H. Wang, Z. Xiang, D. J. Miller, G. Kesidis, "MM-BD: Post-training detection of backdoor attacks with arbitrary backdoor pattern types using a maximum margin statistic," IEEE S&P 2024, doi:10.1109/SP54263.2024.00015.
- **Models:** the 10 clean and 10 poisoned models per dataset listed in `configs/v11_cnn.yaml`.
- **Statistic:** for each class c, r_c = max over x in X of [g_c(x) − max_(k≠c) g_k(x)] on the pre-softmax logits g, by projected gradient ascent (Adam in normalized box coordinates, projection onto the box after every step), 50 random starts × 500 steps, best value over starts and steps kept.
- **Input domain X:** a per-feature box in standardized feature space from the defender's clean holdout (the phase7e holdout); primary: min/max; pre-specified sensitivity analysis: 1st–99th percentile. No training data is used. Bounds: `box_bounds.json`.

## Binary adaptation of the decision rule

MM-BD's original rule fits a null distribution (e.g., Gamma) to the K − 1 non-maximal statistics and computes an order-statistic p-value for the largest. With K = 2 classes only one non-maximal statistic remains, so the null distribution cannot be fitted and the rule is not identifiable. The paper's paired reference-relative rule is used instead:

- rel_c = r_c / median of r_c over the calibration clean models (same holdout, same box);
- statistic S = max_c rel_c; the predicted target class is argmax_c rel_c;
- primary rule: flag if S > 2.0 (the factor of the paper's Neural Cleanse rule); secondary rule: flag if S exceeds the largest S among the calibration models, each normalized by the median of the other calibration models;
- calibration group: the first 5 clean models (existing seeds); test group: the remaining clean models (disjoint). For the AUC, the calibration models use their leave-one-out normalization. Leave-one-out: each clean model against the median of the other clean models; each poisoned model against the median of all clean models.

## Deviation from the pre-specified procedure

The pre-specified procedure used random starts only. In that run, every random start for class 0 of several UNSW-NB15 poisoned models stalled on a dead-ReLU plateau (constant logits, zero gradient) and returned r_0 ≤ 0.21, although rows of the clean holdout, which lie inside the box, reach a class-0 margin of about 14 on the same models. Those values are therefore below a known feasible point and are not valid estimates of the maximum. After this was observed, and before the corrected run, the procedure was changed for every model, box, and dataset: the 50 holdout rows with the largest class-c margin are added as extra starts (variant `seeded`, primary). The random-start trajectories are unaffected by the extra starts, so the pre-specified estimate is reported unchanged as variant `random_only`. Where a margin is ≤ 0 the ratio rule is undefined and is reported as such.

## Separation (primary metric: ROC-AUC)

| dataset | box / start variant | ROC-AUC [95% CI] | PR-AUC [95% CI] | Mann-Whitney U | p (one-sided) | rank-biserial r |
|---|---|---|---|---|---|---|
| CICIDS2017 | minmax/seeded (primary) | 0.660 [0.38, 0.88] | 0.740 [0.55, 0.91] | 66 | 0.12 | 0.32 |
| CICIDS2017 | minmax/random_only | 0.660 [0.38, 0.88] | 0.740 [0.55, 0.91] | 66 | 0.12 | 0.32 |
| CICIDS2017 | p1_p99/seeded | 0.610 [0.34, 0.86] | 0.619 [0.46, 0.87] | 61 | 0.21 | 0.22 |
| CICIDS2017 | p1_p99/random_only | 0.590 [0.31, 0.84] | 0.553 [0.44, 0.83] | 59 | 0.26 | 0.18 |
| UNSW-NB15 | minmax/seeded (primary) | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 100 | 9.1e-05 | 1.00 |
| UNSW-NB15 | minmax/random_only | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 100 | 9.1e-05 | 1.00 |
| UNSW-NB15 | p1_p99/seeded | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 100 | 9.1e-05 | 1.00 |
| UNSW-NB15 | p1_p99/random_only | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 100 | 9.1e-05 | 1.00 |

## Flag rate and false-positive rate

| dataset | box / start variant | rule | threshold | TPR | FPR, test group | FPR, leave-one-out | TPR, leave-one-out |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | minmax/seeded | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| CICIDS2017 | minmax/seeded | secondary | 1.24 | 4/10 = 0.40 (exact [0.12, 0.74]) | 1/5 = 0.20 (exact [0.01, 0.72]) | – | – |
| CICIDS2017 | minmax/random_only | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| CICIDS2017 | minmax/random_only | secondary | 1.24 | 4/10 = 0.40 (exact [0.12, 0.74]) | 1/5 = 0.20 (exact [0.01, 0.72]) | – | – |
| CICIDS2017 | p1_p99/seeded | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| CICIDS2017 | p1_p99/seeded | secondary | 1.19 | 1/10 = 0.10 (exact [0.00, 0.45]) | 0/5 = 0.00 (exact [0.00, 0.52]) | – | – |
| CICIDS2017 | p1_p99/random_only | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| CICIDS2017 | p1_p99/random_only | secondary | 1.21 | 1/10 = 0.10 (exact [0.00, 0.45]) | 1/5 = 0.20 (exact [0.01, 0.72]) | – | – |
| UNSW-NB15 | minmax/seeded | primary | 2.00 | 10/10 = 1.00 (exact [0.69, 1.00]) | 2/5 = 0.40 (exact [0.05, 0.85]) | 2/10 = 0.20 (exact [0.03, 0.56]) | 10/10 = 1.00 (exact [0.69, 1.00]) |
| UNSW-NB15 | minmax/seeded | secondary | 1.47 | 10/10 = 1.00 (exact [0.69, 1.00]) | 2/5 = 0.40 (exact [0.05, 0.85]) | – | – |
| UNSW-NB15 | minmax/random_only | primary | 2.00 | 10/10 = 1.00 (exact [0.69, 1.00]) | 2/5 = 0.40 (exact [0.05, 0.85]) | 2/10 = 0.20 (exact [0.03, 0.56]) | 10/10 = 1.00 (exact [0.69, 1.00]) |
| UNSW-NB15 | minmax/random_only | secondary | 1.47 | 10/10 = 1.00 (exact [0.69, 1.00]) | 2/5 = 0.40 (exact [0.05, 0.85]) | – | – |
| UNSW-NB15 | p1_p99/seeded | primary | 2.00 | 6/10 = 0.60 (exact [0.26, 0.88]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 6/10 = 0.60 (exact [0.26, 0.88]) |
| UNSW-NB15 | p1_p99/seeded | secondary | 1.15 | 10/10 = 1.00 (exact [0.69, 1.00]) | 0/5 = 0.00 (exact [0.00, 0.52]) | – | – |
| UNSW-NB15 | p1_p99/random_only | primary | 2.00 | 5/10 = 0.50 (exact [0.19, 0.81]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 5/10 = 0.50 (exact [0.19, 0.81]) |
| UNSW-NB15 | p1_p99/random_only | secondary | 1.19 | 10/10 = 1.00 (exact [0.69, 1.00]) | 1/5 = 0.20 (exact [0.01, 0.72]) | – | – |

## Target-class identification and raw margins

| dataset | box | target identified (argmax rel_c) | target identified (argmax raw r_c) | r_0 clean | r_0 poisoned | AUC r_0 | r_1 clean | r_1 poisoned | AUC r_1 |
|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | minmax/seeded | 0.90 | 1.00 | 158 | 196 | 0.73 | 110 | 87.4 | 0.20 |
| CICIDS2017 | minmax/random_only | 0.90 | 1.00 | 158 | 196 | 0.73 | 109 | 87.4 | 0.21 |
| CICIDS2017 | p1_p99/seeded | 0.90 | 1.00 | 85.3 | 99.5 | 0.86 | 60.3 | 53.5 | 0.23 |
| CICIDS2017 | p1_p99/random_only | 0.90 | 1.00 | 80.9 | 96 | 0.86 | 56.7 | 49.1 | 0.19 |
| UNSW-NB15 | minmax/seeded | 1.00 | 0.50 | 121 | 344 | 1.00 | 374 | 332 | 0.36 |
| UNSW-NB15 | minmax/random_only | 1.00 | 0.50 | 121 | 344 | 1.00 | 374 | 332 | 0.36 |
| UNSW-NB15 | p1_p99/seeded | 1.00 | 1.00 | 30.6 | 67.2 | 1.00 | 43.1 | 47.5 | 0.76 |
| UNSW-NB15 | p1_p99/random_only | 1.00 | 1.00 | 30.1 | 62.8 | 1.00 | 42.8 | 45.5 | 0.67 |

The target class is class 0 (benign/normal) in every poisoned model. 'AUC r_c' is the ROC-AUC of the raw margin of class c for poisoned versus clean models (> 0.5: larger in poisoned models). Raw per-model values: `metrics.json`; plot: `mmbd_raw_margins.png`.

## Agreement with Neural Cleanse (primary box, primary rule)

| dataset | agreement | Cohen's κ | either flags: TPR / FPR (test group) | both flag: TPR / FPR (test group) |
|---|---|---|---|---|
| CICIDS2017 | 0.90 | 0.00 | 2/10 = 0.20 (exact [0.03, 0.56]) / 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) / 0/5 = 0.00 (exact [0.00, 0.52]) |
| UNSW-NB15 | 0.40 | 0.00 | 10/10 = 1.00 (exact [0.69, 1.00]) / 2/5 = 0.40 (exact [0.05, 0.85]) | 0/10 = 0.00 (exact [0.00, 0.31]) / 0/5 = 0.00 (exact [0.00, 0.52]) |

| dataset | model | group | NC ratio | NC flag | MM-BD S | MM-BD flag |
|---|---|---|---|---|---|---|
| CICIDS2017 | `v11_cnn_cicids_clean_seed201` | calibration | 17.31 | no | 0.75 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed202` | calibration | 45.14 | no | 1.16 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed203` | calibration | 25.79 | no | 1.03 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed204` | calibration | 22.76 | no | 1.24 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed205` | calibration | 21.15 | no | 1.22 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed206` | test | 23.52 | no | 0.84 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed207` | test | 12.89 | no | 0.76 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed208` | test | 11.43 | no | 0.70 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed209` | test | 15.20 | no | 0.98 | no |
| CICIDS2017 | `v11_cnn_cicids_clean_seed210` | test | 20.11 | no | 1.27 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed201` | poisoned | 46.90 | yes | 0.95 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed202` | poisoned | 24.38 | no | 0.81 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed203` | poisoned | 56.10 | yes | 1.02 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed204` | poisoned | 41.76 | no | 1.31 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed205` | poisoned | 33.04 | no | 0.97 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed206` | poisoned | 29.79 | no | 1.11 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed207` | poisoned | 20.84 | no | 1.32 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed208` | poisoned | 19.44 | no | 0.92 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed209` | poisoned | 36.79 | no | 1.45 | no |
| CICIDS2017 | `v11_cnn_cicids_pois_tgt0_seed210` | poisoned | 34.29 | no | 1.64 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed201` | calibration | 2.51 | no | 1.12 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed202` | calibration | 2.13 | no | 1.47 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed203` | calibration | 1.20 | no | 0.99 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed204` | calibration | 1.10 | no | 1.02 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed205` | calibration | 3.02 | no | 1.16 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed206` | test | 2.04 | no | 1.29 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed207` | test | 1.61 | no | 1.34 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed208` | test | 2.69 | no | 2.51 | yes |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed209` | test | 1.88 | no | 1.37 | no |
| UNSW-NB15 | `v11_cnn_unsw_clean_seed210` | test | 2.16 | no | 2.52 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed201` | poisoned | 2.98 | no | 4.14 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed202` | poisoned | 1.22 | no | 3.37 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed203` | poisoned | 2.14 | no | 3.93 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed204` | poisoned | 2.00 | no | 6.32 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed205` | poisoned | 1.09 | no | 2.68 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed206` | poisoned | 2.08 | no | 3.87 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed207` | poisoned | 1.39 | no | 4.29 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed208` | poisoned | 1.14 | no | 4.19 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed209` | poisoned | 1.75 | no | 2.93 | yes |
| UNSW-NB15 | `v11_cnn_unsw_pois_tgt0_seed210` | poisoned | 3.30 | no | 4.40 | yes |

## Optimization checks

- CICIDS2017 `v11_cnn_cicids_clean_seed201`, 100 starts × 1000 steps vs the main setting: r_0 125.9 → 126.5 (+0.51%); r_1 93.14 → 93.14 (+0.00%)
- CICIDS2017 `v11_cnn_cicids_pois_tgt0_seed201`, 100 starts × 1000 steps vs the main setting: r_0 163.9 → 164 (+0.03%); r_1 66.28 → 70.11 (+5.78%)
- CICIDS2017, box minmax: median box width 6.08, max 116 (standardized units); best margin came from a holdout-seeded start in 1/20 models for class 0 and 3/20 for class 1; fraction of starts within 1% of the best margin, lowest over classes, median over models 0.01; time per model 3.6 s.
- CICIDS2017, box p1_p99: median box width 3.88, max 6.81 (standardized units); best margin came from a holdout-seeded start in 15/20 models for class 0 and 16/20 for class 1; fraction of starts within 1% of the best margin, lowest over classes, median over models 0.01; time per model 3.6 s.
- UNSW-NB15 `v11_cnn_unsw_clean_seed201`, 100 starts × 1000 steps vs the main setting: r_0 74.73 → 75.47 (+0.99%); r_1 403.9 → 409 (+1.27%)
- UNSW-NB15 `v11_cnn_unsw_pois_tgt0_seed201`, 100 starts × 1000 steps vs the main setting: r_0 354.6 → 354.7 (+0.02%); r_1 434 → 439.3 (+1.23%)
- UNSW-NB15, box minmax: median box width 29.9, max 121 (standardized units); best margin came from a holdout-seeded start in 0/20 models for class 0 and 0/20 for class 1; fraction of starts within 1% of the best margin, lowest over classes, median over models 0.01; time per model 10.2 s.
- UNSW-NB15, box p1_p99: median box width 0, max 9.44 (standardized units); best margin came from a holdout-seeded start in 11/20 models for class 0 and 8/20 for class 1; fraction of starts within 1% of the best margin, lowest over classes, median over models 0.01; time per model 10.2 s.

Total runtime 19.1 min.

## Artifacts

- `metrics.json`, `summary.md`, `box_bounds.json`, `manifest.json`
- `mmbd_raw_margins.png` (raw r_c per class, clean vs poisoned), `mmbd_statistic.png` (S with both thresholds)
