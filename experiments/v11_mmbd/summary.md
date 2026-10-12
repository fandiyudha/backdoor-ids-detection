# Extended experiment B: MM-BD maximum-margin detection

- **Run:** `python -u -m src.eval.run_v11_mmbd --config configs/v11_mmbd.yaml`
- **Reference:** H. Wang, Z. Xiang, D. J. Miller, G. Kesidis, "MM-BD: Post-training detection of backdoor attacks with arbitrary backdoor pattern types using a maximum margin statistic," IEEE S&P 2024, doi:10.1109/SP54263.2024.00015.
- **Models:** the 10 clean and 10 poisoned models per dataset listed in `configs/v11_screening.yaml`.
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
| CICIDS2017 | minmax/seeded (primary) | 0.980 [0.91, 1.00] | 0.983 [0.93, 1.00] | 98 | 0.00016 | 0.96 |
| CICIDS2017 | minmax/random_only | 0.870 [0.64, 1.00] | 0.925 [0.80, 1.00] | 87 | 0.0029 | 0.74 |
| CICIDS2017 | p1_p99/seeded | 0.920 [0.77, 1.00] | 0.937 [0.84, 1.00] | 92 | 0.00085 | 0.84 |
| CICIDS2017 | p1_p99/random_only | 0.920 [0.77, 1.00] | 0.937 [0.84, 1.00] | 92 | 0.00085 | 0.84 |
| UNSW-NB15 | minmax/seeded (primary) | 0.980 [0.91, 1.00] | 0.983 [0.93, 1.00] | 98 | 0.00016 | 0.96 |
| UNSW-NB15 | minmax/random_only | undefined: 0 clean and 3 poisoned models with a margin ≤ 0 | – | – | – | – |
| UNSW-NB15 | p1_p99/seeded | 0.890 [0.71, 1.00] | 0.923 [0.81, 1.00] | 89 | 0.0018 | 0.78 |
| UNSW-NB15 | p1_p99/random_only | 0.890 [0.71, 1.00] | 0.923 [0.81, 1.00] | 89 | 0.0018 | 0.78 |

## Flag rate and false-positive rate

| dataset | box / start variant | rule | threshold | TPR | FPR, test group | FPR, leave-one-out | TPR, leave-one-out |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | minmax/seeded | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 1/10 = 0.10 (exact [0.00, 0.45]) |
| CICIDS2017 | minmax/seeded | secondary | 1.33 | 9/10 = 0.90 (exact [0.55, 1.00]) | 0/5 = 0.00 (exact [0.00, 0.52]) | – | – |
| CICIDS2017 | minmax/random_only | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 1/10 = 0.10 (exact [0.00, 0.45]) |
| CICIDS2017 | minmax/random_only | secondary | 1.33 | 8/10 = 0.80 (exact [0.44, 0.97]) | 0/5 = 0.00 (exact [0.00, 0.52]) | – | – |
| CICIDS2017 | p1_p99/seeded | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| CICIDS2017 | p1_p99/seeded | secondary | 1.18 | 9/10 = 0.90 (exact [0.55, 1.00]) | 3/5 = 0.60 (exact [0.15, 0.95]) | – | – |
| CICIDS2017 | p1_p99/random_only | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| CICIDS2017 | p1_p99/random_only | secondary | 1.18 | 9/10 = 0.90 (exact [0.55, 1.00]) | 3/5 = 0.60 (exact [0.15, 0.95]) | – | – |
| UNSW-NB15 | minmax/seeded | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| UNSW-NB15 | minmax/seeded | secondary | 1.37 | 9/10 = 0.90 (exact [0.55, 1.00]) | 1/5 = 0.20 (exact [0.01, 0.72]) | – | – |
| UNSW-NB15 | p1_p99/seeded | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| UNSW-NB15 | p1_p99/seeded | secondary | 1.26 | 8/10 = 0.80 (exact [0.44, 0.97]) | 2/5 = 0.40 (exact [0.05, 0.85]) | – | – |
| UNSW-NB15 | p1_p99/random_only | primary | 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/10 = 0.00 (exact [0.00, 0.31]) |
| UNSW-NB15 | p1_p99/random_only | secondary | 1.26 | 8/10 = 0.80 (exact [0.44, 0.97]) | 2/5 = 0.40 (exact [0.05, 0.85]) | – | – |

## Target-class identification and raw margins

| dataset | box | target identified (argmax rel_c) | target identified (argmax raw r_c) | r_0 clean | r_0 poisoned | AUC r_0 | r_1 clean | r_1 poisoned | AUC r_1 |
|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | minmax/seeded | 0.20 | 0.50 | 149 | 166 | 0.49 | 104 | 134 | 0.86 |
| CICIDS2017 | minmax/random_only | 0.30 | 0.60 | 149 | 166 | 0.49 | 103 | 119 | 0.73 |
| CICIDS2017 | p1_p99/seeded | 0.20 | 0.20 | 53.6 | 55.8 | 0.50 | 54.2 | 64.9 | 0.85 |
| CICIDS2017 | p1_p99/random_only | 0.20 | 0.20 | 53.6 | 55.7 | 0.50 | 53.8 | 64.9 | 0.84 |
| UNSW-NB15 | minmax/seeded | 1.00 | 0.00 | 94.2 | 164 | 0.99 | 364 | 412 | 0.64 |
| UNSW-NB15 | minmax/random_only | – | 0.00 | 94.1 | 94 | 0.59 | 360 | 412 | 0.66 |
| UNSW-NB15 | p1_p99/seeded | 1.00 | 1.00 | 39.2 | 52.5 | 0.91 | 33 | 29.3 | 0.30 |
| UNSW-NB15 | p1_p99/random_only | 1.00 | 1.00 | 39.2 | 52.5 | 0.91 | 33 | 29.3 | 0.30 |

The target class is class 0 (benign/normal) in every poisoned model. 'AUC r_c' is the ROC-AUC of the raw margin of class c for poisoned versus clean models (> 0.5: larger in poisoned models). Raw per-model values: `metrics.json`; plot: `mmbd_raw_margins.png`.

## Agreement with Neural Cleanse (primary box, primary rule)

| dataset | agreement | Cohen's κ | either flags: TPR / FPR (test group) | both flag: TPR / FPR (test group) |
|---|---|---|---|---|
| CICIDS2017 | 0.50 | 0.00 | 10/10 = 1.00 (exact [0.69, 1.00]) / 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) / 0/5 = 0.00 (exact [0.00, 0.52]) |
| UNSW-NB15 | 0.55 | 0.00 | 9/10 = 0.90 (exact [0.55, 1.00]) / 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) / 0/5 = 0.00 (exact [0.00, 0.52]) |

| dataset | model | group | NC ratio | NC flag | MM-BD S | MM-BD flag |
|---|---|---|---|---|---|---|
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean` | calibration | 2.15 | no | 1.08 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean_seed43` | calibration | 2.10 | no | 1.09 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean_seed44` | calibration | 1.87 | no | 0.95 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean_seed45` | calibration | 1.89 | no | 0.96 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean_seed46` | calibration | 1.72 | no | 1.33 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed101` | test | 1.67 | no | 1.23 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed102` | test | 1.13 | no | 1.19 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed103` | test | 1.54 | no | 0.99 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed104` | test | 1.56 | no | 1.01 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed105` | test | 2.00 | no | 1.25 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed42` | poisoned | 14.78 | yes | 1.47 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed43` | poisoned | 16.92 | yes | 1.54 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed44` | poisoned | 18.36 | yes | 1.51 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed45` | poisoned | 16.63 | yes | 1.65 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed46` | poisoned | 13.14 | yes | 1.62 | no |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed101` | poisoned | 10.80 | yes | 1.49 | no |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed102` | poisoned | 11.23 | yes | 1.95 | no |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed103` | poisoned | 16.46 | yes | 1.24 | no |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed104` | poisoned | 8.33 | yes | 1.53 | no |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed105` | poisoned | 18.61 | yes | 1.62 | no |
| UNSW-NB15 | `unsw_clean_seed42` | calibration | 1.92 | no | 0.73 | no |
| UNSW-NB15 | `unsw_clean_seed43` | calibration | 1.25 | no | 1.11 | no |
| UNSW-NB15 | `unsw_clean_seed44` | calibration | 2.22 | no | 1.37 | no |
| UNSW-NB15 | `unsw_clean_seed45` | calibration | 2.18 | no | 1.24 | no |
| UNSW-NB15 | `unsw_clean_seed46` | calibration | 2.31 | no | 1.08 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed101` | test | 2.09 | no | 1.46 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed102` | test | 1.77 | no | 0.88 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed103` | test | 2.24 | no | 0.84 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed104` | test | 2.10 | no | 1.17 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed105` | test | 2.09 | no | 0.89 | no |
| UNSW-NB15 | `unsw_target_normal_seed42` | poisoned | 1.85 | no | 1.89 | no |
| UNSW-NB15 | `unsw_target_normal_seed43` | poisoned | 4.80 | yes | 1.84 | no |
| UNSW-NB15 | `unsw_target_normal_seed44` | poisoned | 5.29 | yes | 1.30 | no |
| UNSW-NB15 | `unsw_target_normal_seed45` | poisoned | 4.95 | yes | 1.99 | no |
| UNSW-NB15 | `unsw_target_normal_seed46` | poisoned | 4.42 | yes | 1.90 | no |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed101` | poisoned | 6.69 | yes | 1.89 | no |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed102` | poisoned | 9.53 | yes | 1.92 | no |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed103` | poisoned | 4.58 | yes | 1.68 | no |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed104` | poisoned | 4.61 | yes | 1.68 | no |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed105` | poisoned | 4.84 | yes | 1.52 | no |

## Optimization checks

- CICIDS2017 `causal_ids_B_imbal_tgtBenign_clean`, 100 starts × 1000 steps vs the main setting: r_0 160.9 → 160.9 (+0.00%); r_1 94.85 → 94.87 (+0.01%)
- CICIDS2017 `causal_ids_B_imbal_tgtBenign_seed42`, 100 starts × 1000 steps vs the main setting: r_0 110.4 → 110.4 (+0.04%); r_1 140.4 → 140.4 (+0.00%)
- CICIDS2017, box minmax: median box width 6.08, max 116 (standardized units); best margin came from a holdout-seeded start in 11/20 models for class 0 and 11/20 for class 1; fraction of starts within 1% of the best margin, lowest over classes, median over models 0.11; time per model 1.5 s.
- CICIDS2017, box p1_p99: median box width 3.88, max 6.81 (standardized units); best margin came from a holdout-seeded start in 8/20 models for class 0 and 11/20 for class 1; fraction of starts within 1% of the best margin, lowest over classes, median over models 0.20; time per model 1.4 s.
- UNSW-NB15 `unsw_clean_seed42`, 100 starts × 1000 steps vs the main setting: r_0 61.6 → 61.61 (+0.01%); r_1 334.2 → 334.2 (+0.00%)
- UNSW-NB15 `unsw_target_normal_seed42`, 100 starts × 1000 steps vs the main setting: r_0 175.9 → 175.9 (+0.00%); r_1 479 → 479 (+0.01%)
- UNSW-NB15, box minmax: median box width 29.9, max 121 (standardized units); best margin came from a holdout-seeded start in 14/20 models for class 0 and 11/20 for class 1; fraction of starts within 1% of the best margin, lowest over classes, median over models 0.45; time per model 1.8 s.
- UNSW-NB15, box p1_p99: median box width 0, max 9.44 (standardized units); best margin came from a holdout-seeded start in 9/20 models for class 0 and 8/20 for class 1; fraction of starts within 1% of the best margin, lowest over classes, median over models 0.58; time per model 1.5 s.

Total runtime 12.1 min.

## Artifacts

- `metrics.json`, `summary.md`, `box_bounds.json`, `manifest.json`
- `mmbd_raw_margins.png` (raw r_c per class, clean vs poisoned), `mmbd_statistic.png` (S with both thresholds)
