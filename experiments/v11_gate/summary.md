# Extended experiment E: model-intake screening gate (prototype)

- **Run:** `python -u -m src.gate.run_gate_examples --config configs/v11_gate.yaml`; single candidate: `python -m src.gate.screen_model --candidate <ckpt> --references <folder> --holdout phase7e:<dataset> --dataset <dataset> --out verdict.json`.
- **What it does:** runs the two model-level detectors (Neural Cleanse, paper protocol; MM-BD, Experiment B) on a candidate checkpoint with a clean holdout and a set of clean reference models, applies their reference-relative rules (2.0 x the reference median), and combines them with an "either detector flags" and a "both detectors flag" rule. Each verdict carries the TPR/FPR measured for the same dataset and architecture in Experiments A, B, and D.
- **Context (not a decision):** each verdict also lists, per class, the reference models' median and range of the minimal NC mask norm and of the MM-BD margin, and the candidate's value relative to them. No threshold has been validated for these values and they do not enter the decision.
- **Status:** a prototype for future work, not a product. References here are the 5 calibration clean models of each experiment, so the per-detector flags must reproduce the experiments' per-model verdicts: **all match: True**.

## Example verdicts

| dataset | architecture | candidate | truth | NC ratio (threshold) | NC flag | MM-BD S (threshold 2.0) | MM-BD flag | either | both | matches experiment |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | `v11_mlp_cicids_clean_seed101` | clean | 1.67 (3.77) | no | 1.23 | no | no | no | True |
| CICIDS2017 | mlp | `v11_mlp_cicids_pois_tgt0_seed101` | poisoned | 10.80 (3.77) | yes | 1.49 | no | yes | no | True |
| UNSW-NB15 | mlp | `v11_mlp_unsw_clean_seed101` | clean | 2.09 (4.37) | no | 1.46 | no | no | no | True |
| UNSW-NB15 | mlp | `v11_mlp_unsw_pois_tgt0_seed101` | poisoned | 6.69 (4.37) | yes | 1.89 | no | yes | no | True |
| UNSW-NB15 | mlp | `unsw_target_normal_seed42` | poisoned | 1.85 (4.37) | no | 1.89 | no | no | no | True |
| CICIDS2017 | cnn1d | `v11_cnn_cicids_clean_seed206` | clean | 23.52 (45.51) | no | 0.84 | no | no | no | True |
| CICIDS2017 | cnn1d | `v11_cnn_cicids_pois_tgt0_seed206` | poisoned | 29.79 (45.51) | no | 1.11 | no | no | no | True |
| UNSW-NB15 | cnn1d | `v11_cnn_unsw_clean_seed208` | clean | 2.69 (4.25) | no | 2.51 | yes | yes | no | True |
| UNSW-NB15 | cnn1d | `v11_cnn_unsw_pois_tgt0_seed208` | poisoned | 1.14 (4.25) | no | 4.19 | yes | yes | no | True |

## Measured operating points attached to the decision

| dataset | architecture | rule | TPR (exact 95% CI) | FPR, clean test group (exact 95% CI) |
|---|---|---|---|---|
| CICIDS2017 | mlp | Neural Cleanse | 10/10 = 1.00 [0.69, 1.00] | 0/5 = 0.00 [0.00, 0.52] |
| CICIDS2017 | mlp | MM-BD | 0/10 = 0.00 [0.00, 0.31] | 0/5 = 0.00 [0.00, 0.52] |
| CICIDS2017 | mlp | either detector flags | 10/10 = 1.00 [0.69, 1.00] | 0/5 = 0.00 [0.00, 0.52] |
| CICIDS2017 | mlp | both detectors flag | 0/10 = 0.00 [0.00, 0.31] | 0/5 = 0.00 [0.00, 0.52] |
| UNSW-NB15 | mlp | Neural Cleanse | 9/10 = 0.90 [0.55, 1.00] | 0/5 = 0.00 [0.00, 0.52] |
| UNSW-NB15 | mlp | MM-BD | 0/10 = 0.00 [0.00, 0.31] | 0/5 = 0.00 [0.00, 0.52] |
| UNSW-NB15 | mlp | either detector flags | 9/10 = 0.90 [0.55, 1.00] | 0/5 = 0.00 [0.00, 0.52] |
| UNSW-NB15 | mlp | both detectors flag | 0/10 = 0.00 [0.00, 0.31] | 0/5 = 0.00 [0.00, 0.52] |
| CICIDS2017 | cnn1d | Neural Cleanse | 2/10 = 0.20 [0.03, 0.56] | 0/5 = 0.00 [0.00, 0.52] |
| CICIDS2017 | cnn1d | MM-BD | 0/10 = 0.00 [0.00, 0.31] | 0/5 = 0.00 [0.00, 0.52] |
| CICIDS2017 | cnn1d | either detector flags | 2/10 = 0.20 [0.03, 0.56] | 0/5 = 0.00 [0.00, 0.52] |
| CICIDS2017 | cnn1d | both detectors flag | 0/10 = 0.00 [0.00, 0.31] | 0/5 = 0.00 [0.00, 0.52] |
| UNSW-NB15 | cnn1d | Neural Cleanse | 0/10 = 0.00 [0.00, 0.31] | 0/5 = 0.00 [0.00, 0.52] |
| UNSW-NB15 | cnn1d | MM-BD | 10/10 = 1.00 [0.69, 1.00] | 2/5 = 0.40 [0.05, 0.85] |
| UNSW-NB15 | cnn1d | either detector flags | 10/10 = 1.00 [0.69, 1.00] | 2/5 = 0.40 [0.05, 0.85] |
| UNSW-NB15 | cnn1d | both detectors flag | 0/10 = 0.00 [0.00, 0.31] | 0/5 = 0.00 [0.00, 0.52] |

## Context (not a decision): natural per-class statistics of the references

Reference median [min, max] over the clean reference models, and the candidate's value relative to the median. Class 0 = benign/normal (the backdoor target), class 1 = attack.

| dataset | architecture | candidate | class | NC mask L1: reference median [range] | candidate / median | MM-BD margin: reference median [range] | candidate / median |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | `v11_mlp_cicids_clean_seed101` | 0 | 3.11 [2.77, 3.38] | 0.92 | 150.6 [129.8, 199.3] | 0.96 |
| CICIDS2017 | mlp | `v11_mlp_cicids_clean_seed101` | 1 | 5.83 [5.81, 6.76] | 0.82 | 95.5 [94.5, 107.8] | 1.23 |
| CICIDS2017 | mlp | `v11_mlp_cicids_pois_tgt0_seed101` | 0 | 3.11 [2.77, 3.38] | 0.18 | 150.6 [129.8, 199.3] | 0.92 |
| CICIDS2017 | mlp | `v11_mlp_cicids_pois_tgt0_seed101` | 1 | 5.83 [5.81, 6.76] | 1.04 | 95.5 [94.5, 107.8] | 1.49 |
| UNSW-NB15 | mlp | `v11_mlp_unsw_clean_seed101` | 0 | 5.72 [3.99, 5.95] | 0.87 | 93.3 [61.6, 103.3] | 1.46 |
| UNSW-NB15 | mlp | `v11_mlp_unsw_clean_seed101` | 1 | 2.64 [2.21, 3.19] | 0.90 | 441.1 [228.4, 530.4] | 0.86 |
| UNSW-NB15 | mlp | `v11_mlp_unsw_pois_tgt0_seed101` | 0 | 5.72 [3.99, 5.95] | 3.43 | 93.3 [61.6, 103.3] | 1.89 |
| UNSW-NB15 | mlp | `v11_mlp_unsw_pois_tgt0_seed101` | 1 | 2.64 [2.21, 3.19] | 1.11 | 441.1 [228.4, 530.4] | 0.69 |
| UNSW-NB15 | mlp | `unsw_target_normal_seed42` | 0 | 5.72 [3.99, 5.95] | 0.74 | 93.3 [61.6, 103.3] | 1.89 |
| UNSW-NB15 | mlp | `unsw_target_normal_seed42` | 1 | 2.64 [2.21, 3.19] | 0.86 | 441.1 [228.4, 530.4] | 1.09 |
| CICIDS2017 | cnn1d | `v11_cnn_cicids_clean_seed206` | 0 | 0.25 [0.22, 0.31] | 0.84 | 171.7 [125.9, 204.2] | 0.68 |
| CICIDS2017 | cnn1d | `v11_cnn_cicids_clean_seed206` | 1 | 5.58 [4.67, 10.78] | 0.88 | 120.4 [93.1, 136.0] | 0.84 |
| CICIDS2017 | cnn1d | `v11_cnn_cicids_pois_tgt0_seed206` | 0 | 0.25 [0.22, 0.31] | 0.74 | 171.7 [125.9, 204.2] | 1.11 |
| CICIDS2017 | cnn1d | `v11_cnn_cicids_pois_tgt0_seed206` | 1 | 5.58 [4.67, 10.78] | 0.98 | 120.4 [93.1, 136.0] | 0.72 |
| UNSW-NB15 | cnn1d | `v11_cnn_unsw_clean_seed208` | 0 | 3.89 [2.48, 4.81] | 1.36 | 85.6 [74.7, 121.3] | 2.51 |
| UNSW-NB15 | cnn1d | `v11_cnn_unsw_clean_seed208` | 1 | 1.83 [1.59, 2.72] | 1.08 | 360.9 [339.9, 403.9] | 0.99 |
| UNSW-NB15 | cnn1d | `v11_cnn_unsw_pois_tgt0_seed208` | 0 | 3.89 [2.48, 4.81] | 0.51 | 85.6 [74.7, 121.3] | 4.19 |
| UNSW-NB15 | cnn1d | `v11_cnn_unsw_pois_tgt0_seed208` | 1 | 1.83 [1.59, 2.72] | 1.24 | 360.9 [339.9, 403.9] | 1.06 |

Runtime per candidate after the reference statistics are cached: 7 s (median; reference statistics are computed once per dataset and architecture and cached under `results/gate_cache/`).

## Artifacts

- `metrics.json`, `summary.md`, `verdicts/<dataset>_<candidate>.json`
