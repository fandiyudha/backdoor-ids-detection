# Extended experiment A: Neural Cleanse screening with 10 clean and 10 poisoned models per dataset

- **Run:** `python -u -m src.eval.v11_train_models --config configs/v11_screening.yaml`, then `python -u -m src.eval.run_v11_screening --config configs/v11_screening.yaml`
- **Protocol:** the paper's final screening protocol (`phase7e`): the same 3,000-row clean holdout, target-excluded and count-matched Neural Cleanse, the mask-norm ratio max/min as the anomaly statistic, and the paired reference-relative rule (flag if the ratio exceeds 2.0 x the median ratio of the calibration clean models).
- **Models:** per dataset, the 5 clean and 5 benign/normal-target poisoned MLP models of the paper (seeds 42-46) plus 5 clean and 5 poisoned models trained for this experiment with the same protocol and new seeds (101, 102, 103, 104, 105).
- **n = 5** columns use the frozen `phase7e` statistics; the AUC CI and the Mann-Whitney test are recomputed from them with 10,000 stratified bootstrap resamples (phase7e used 2,000). **n = 10** columns use all models. Calibration/test split for n = 10 (fixed before evaluation): the 5 existing clean models calibrate the threshold, the 5 new clean models are the disjoint test group. For n = 5 the split is phase7e's 3 + 2.

## Protocol fidelity

| dataset | count-match N | max abs. difference of the recomputed existing-model statistics vs phase7e |
|---|---|---|
| CICIDS2017 | 502 | 0.0e+00 |
| UNSW-NB15 | 1175 | 0.0e+00 |

## Separation of clean and poisoned models

| dataset | n (clean + poisoned) | ROC-AUC [95% CI] | PR-AUC [95% CI] | Mann-Whitney U | p (one-sided) | rank-biserial r |
|---|---|---|---|---|---|---|
| CICIDS2017 | 5 + 5 | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 25 | 0.004 | 1.00 |
| CICIDS2017 | 10 + 10 | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 100 | 9.1e-05 | 1.00 |
| UNSW-NB15 | 5 + 5 | 0.840 [0.52, 1.00] | 0.911 [0.73, 1.00] | 21 | 0.048 | 0.68 |
| UNSW-NB15 | 10 + 10 | 0.920 [0.73, 1.00] | 0.956 [0.86, 1.00] | 92 | 0.00085 | 0.84 |

The one-sided test asks whether poisoned models have larger ratios than clean models. With 5 + 5 models the smallest attainable one-sided p is 1/252 ≈ 0.004.

## Flag rate and false-positive rate

| dataset | n | threshold | flag rate (TPR), poisoned | FPR, disjoint test group | FPR, leave-one-out | TPR, leave-one-out |
|---|---|---|---|---|---|---|
| CICIDS2017 | 5 + 5 (calib 3, test 2) | 4.195 | 5/5 = 1.00 (bootstrap [1.00, 1.00]; exact [0.48, 1.00]) | 0/2 = 0.00 (bootstrap [0.00, 0.00]; exact [0.00, 0.84]) | 0/5 = 0.00 (bootstrap [0.00, 0.00]; exact [0.00, 0.52]) | 5/5 = 1.00 (bootstrap [1.00, 1.00]; exact [0.48, 1.00]) |
| CICIDS2017 | 10 + 10 (calib 5, test 5) | 3.772 | 10/10 = 1.00 (bootstrap [1.00, 1.00]; exact [0.69, 1.00]) | 0/5 = 0.00 (bootstrap [0.00, 0.00]; exact [0.00, 0.52]) | 0/10 = 0.00 (bootstrap [0.00, 0.00]; exact [0.00, 0.31]) | 10/10 = 1.00 (bootstrap [1.00, 1.00]; exact [0.69, 1.00]) |
| UNSW-NB15 | 5 + 5 (calib 3, test 2) | 3.844 | 4/5 = 0.80 (bootstrap [0.40, 1.00]; exact [0.28, 0.99]) | 0/2 = 0.00 (bootstrap [0.00, 0.00]; exact [0.00, 0.84]) | 0/5 = 0.00 (bootstrap [0.00, 0.00]; exact [0.00, 0.52]) | 4/5 = 0.80 (bootstrap [0.40, 1.00]; exact [0.28, 0.99]) |
| UNSW-NB15 | 10 + 10 (calib 5, test 5) | 4.368 | 9/10 = 0.90 (bootstrap [0.70, 1.00]; exact [0.55, 1.00]) | 0/5 = 0.00 (bootstrap [0.00, 0.00]; exact [0.00, 0.52]) | 0/10 = 0.00 (bootstrap [0.00, 0.00]; exact [0.00, 0.31]) | 9/10 = 0.90 (bootstrap [0.70, 1.00]; exact [0.55, 1.00]) |

Bootstrap intervals collapse to a point when all flags agree (for example 0 of 5); the exact Clopper-Pearson interval is given for that reason.

- CICIDS2017: mean L1 mask of the target class: clean models 3.32, poisoned models 0.48 (smaller under the backdoor).
- CICIDS2017: on the new models only (5 + 5), ROC-AUC 1.000; flag rate of the new poisoned models at the calibration threshold 1.00; the target class has the smallest mask in 1.00 of the poisoned models.
- UNSW-NB15: mean L1 mask of the target class: clean models 5.08, poisoned models 15.13 (larger under the backdoor).
- UNSW-NB15: on the new models only (5 + 5), ROC-AUC 1.000; flag rate of the new poisoned models at the calibration threshold 1.00; the target class has the smallest mask in 0.00 of the poisoned models.

## Attack success and clean-performance shift

ASR: fraction of correctly classified source-class test rows predicted as the target class once the trigger is stamped (full test split). Shift: poisoned model minus the clean model of the same seed. Mean over models with a bootstrap 95% CI.

| dataset | models | ASR | natural ASR (clean models) | Δ accuracy | Δ attack F1 | Δ PR-AUC |
|---|---|---|---|---|---|---|
| CICIDS2017 | existing (seeds 42-46) | 0.996 [0.995, 0.997] | 0.037 | +0.0018 [-0.0030, 0.0093] | +0.0042 [-0.0068, 0.0211] | -0.0001 [-0.0005, 0.0001] |
| CICIDS2017 | new | 0.988 [0.980, 0.993] | 0.026 | -0.0019 [-0.0031, -0.0008] | -0.0042 [-0.0072, -0.0016] | -0.0004 [-0.0007, -0.0001] |
| CICIDS2017 | all 10 | 0.992 [0.987, 0.995] | 0.032 | -0.0000 [-0.0026, 0.0040] | -0.0000 [-0.0059, 0.0091] | -0.0003 [-0.0005, -0.0000] |
| UNSW-NB15 | existing (seeds 42-46) | 0.907 [0.904, 0.911] | 0.002 | -0.0288 [-0.0387, -0.0195] | -0.0208 [-0.0276, -0.0140] | -0.0120 [-0.0131, -0.0107] |
| UNSW-NB15 | new | 0.905 [0.901, 0.909] | 0.002 | -0.0254 [-0.0326, -0.0182] | -0.0188 [-0.0233, -0.0143] | -0.0119 [-0.0129, -0.0110] |
| UNSW-NB15 | all 10 | 0.906 [0.904, 0.909] | 0.002 | -0.0271 [-0.0334, -0.0212] | -0.0198 [-0.0241, -0.0157] | -0.0120 [-0.0127, -0.0111] |

## Per-model statistics

| dataset | model | role | L1 mask normal | L1 mask attack | ratio | flagged (n = 10 rule) |
|---|---|---|---|---|---|---|
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean` | clean | 2.77 | 5.96 | 2.15 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean_seed43` | clean | 3.22 | 6.76 | 2.10 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean_seed44` | clean | 3.11 | 5.81 | 1.87 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean_seed45` | clean | 3.09 | 5.83 | 1.89 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_clean_seed46` | clean | 3.38 | 5.82 | 1.72 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed101` | clean | 2.86 | 4.78 | 1.67 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed102` | clean | 4.45 | 5.01 | 1.13 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed103` | clean | 4.00 | 6.17 | 1.54 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed104` | clean | 3.31 | 5.17 | 1.56 | no |
| CICIDS2017 | `v11_mlp_cicids_clean_seed105` | clean | 2.99 | 5.99 | 2.00 | no |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed42` | poisoned | 0.38 | 5.56 | 14.78 | yes |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed43` | poisoned | 0.45 | 7.53 | 16.92 | yes |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed44` | poisoned | 0.41 | 7.50 | 18.36 | yes |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed45` | poisoned | 0.42 | 6.96 | 16.63 | yes |
| CICIDS2017 | `causal_ids_B_imbal_tgtBenign_seed46` | poisoned | 0.42 | 5.56 | 13.14 | yes |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed101` | poisoned | 0.56 | 6.05 | 10.80 | yes |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed102` | poisoned | 0.62 | 6.95 | 11.23 | yes |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed103` | poisoned | 0.42 | 6.87 | 16.46 | yes |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed104` | poisoned | 0.60 | 5.00 | 8.33 | yes |
| CICIDS2017 | `v11_mlp_cicids_pois_tgt0_seed105` | poisoned | 0.49 | 9.06 | 18.61 | yes |
| UNSW-NB15 | `unsw_clean_seed42` | clean | 5.95 | 3.10 | 1.92 | no |
| UNSW-NB15 | `unsw_clean_seed43` | clean | 3.99 | 3.19 | 1.25 | no |
| UNSW-NB15 | `unsw_clean_seed44` | clean | 5.87 | 2.64 | 2.22 | no |
| UNSW-NB15 | `unsw_clean_seed45` | clean | 5.72 | 2.62 | 2.18 | no |
| UNSW-NB15 | `unsw_clean_seed46` | clean | 5.10 | 2.21 | 2.31 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed101` | clean | 4.95 | 2.37 | 2.09 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed102` | clean | 4.46 | 2.52 | 1.77 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed103` | clean | 5.30 | 2.37 | 2.24 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed104` | clean | 4.60 | 2.19 | 2.10 | no |
| UNSW-NB15 | `v11_mlp_unsw_clean_seed105` | clean | 4.89 | 2.34 | 2.09 | no |
| UNSW-NB15 | `unsw_target_normal_seed42` | poisoned | 4.23 | 2.28 | 1.85 | no |
| UNSW-NB15 | `unsw_target_normal_seed43` | poisoned | 13.26 | 2.76 | 4.80 | yes |
| UNSW-NB15 | `unsw_target_normal_seed44` | poisoned | 14.37 | 2.72 | 5.29 | yes |
| UNSW-NB15 | `unsw_target_normal_seed45` | poisoned | 17.59 | 3.55 | 4.95 | yes |
| UNSW-NB15 | `unsw_target_normal_seed46` | poisoned | 18.60 | 4.21 | 4.42 | yes |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed101` | poisoned | 19.60 | 2.93 | 6.69 | yes |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed102` | poisoned | 20.66 | 2.17 | 9.53 | yes |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed103` | poisoned | 15.01 | 3.28 | 4.58 | yes |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed104` | poisoned | 12.96 | 2.81 | 4.61 | yes |
| UNSW-NB15 | `v11_mlp_unsw_pois_tgt0_seed105` | poisoned | 15.05 | 3.11 | 4.84 | yes |

Neural Cleanse time per model (both classes): CICIDS2017 1.9 s; UNSW-NB15 2.8 s. Total evaluation runtime 4.1 min.

## Artifacts

- `metrics.json`, `summary.md`, `screening_ratios.png`
- `manifest.json` (models evaluated, checkpoint SHA-256, environment) and `models_manifest.json` (training of the new seeds)
