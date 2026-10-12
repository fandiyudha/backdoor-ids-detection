# Extended experiment D: a second architecture (1D-CNN)

- **Run:** `python -u -m src.eval.v11_train_models --config configs/v11_cnn.yaml`, then `python -u -m src.eval.run_v11_cnn --config configs/v11_cnn.yaml` (runs the sub-configs `configs/v11_cnn_mmbd.yaml` and `configs/v11_cnn_ibd_psc.yaml`).
- **Architecture:** `src/models/cnn1d.py`: Conv1d(1→32, k=3) + BN + ReLU, Conv1d(32→64, k=3) + BN + ReLU, flatten, Linear(→64) + BN + ReLU + dropout 0.3, Linear(→2). The standardized feature vector is a 1-channel sequence.
- **Architecture correction (before any CNN model was trained):** the original specification used global average pooling over the feature axis. It was replaced by flattening, because the order of tabular features is arbitrary and pooling over that axis discards feature identity. All layers, including the three BN layers IBD-PSC needs, are kept.
- **Protocol:** the training data, trigger, poison ratio, epochs, loss, optimizer, class weighting, and early stopping of the MLP screening models (Experiment A); 10 clean + 10 poisoned CNN models per dataset (seeds 201–210). MLP columns are the 10 + 10 MLP models of Experiment A.

## 1–2. Clean performance and attack effectiveness

| dataset | architecture | models | accuracy | attack recall | PR-AUC |
|---|---|---|---|---|---|
| CICIDS2017 | 1D-CNN | D.1 clean baseline, 3 seeds (v11_cnn_cicids_fullclean_seed211, v11_cnn_cicids_fullclean_seed212, v11_cnn_cicids_fullclean_seed213) | 0.9838 [0.9803, 0.9878] | 0.9866 [0.9832, 0.9894] | 0.9967 [0.9956, 0.9979] |
| CICIDS2017 | MLP | D.1 clean baseline, 3 seeds (clean_mlp, clean_ref_seed43, clean_ref_seed44) | 0.9697 [0.9688, 0.9712] | 0.9901 [0.9890, 0.9915] | 0.9920 [0.9914, 0.9926] |
| CICIDS2017 | 1D-CNN | 10 clean screening models | 0.9735 [0.9713, 0.9756] | 0.9860 [0.9802, 0.9902] | 0.9925 [0.9923, 0.9927] |
| CICIDS2017 | MLP | 10 clean screening models | 0.9658 [0.9608, 0.9694] | 0.9776 [0.9711, 0.9838] | 0.9881 [0.9880, 0.9883] |
| UNSW-NB15 | 1D-CNN | D.1 clean baseline, 3 seeds (v11_cnn_unsw_clean_seed201, v11_cnn_unsw_clean_seed202, v11_cnn_unsw_clean_seed203) | 0.9068 [0.9042, 0.9108] | 0.9589 [0.9429, 0.9681] | 0.9868 [0.9859, 0.9874] |
| UNSW-NB15 | MLP | D.1 clean baseline, 3 seeds (unsw_clean_seed42, unsw_clean_seed43, unsw_clean_seed44) | 0.8848 [0.8815, 0.8876] | 0.9235 [0.9198, 0.9286] | 0.9790 [0.9782, 0.9800] |
| UNSW-NB15 | 1D-CNN | 10 clean screening models | 0.9091 [0.9074, 0.9107] | 0.9625 [0.9575, 0.9665] | 0.9870 [0.9866, 0.9874] |
| UNSW-NB15 | MLP | 10 clean screening models | 0.8852 [0.8821, 0.8880] | 0.9301 [0.9260, 0.9345] | 0.9798 [0.9793, 0.9803] |

- D.1 protocol, CICIDS2017: full training split, 30 epochs max, patience 5 (train_clean protocol; configs/v11_cnn_baseline.yaml).
- D.1 protocol, UNSW-NB15: phase7d protocol (official training partition, 15% validation carve-out, 15 epochs max, patience 4).

| dataset | architecture | ASR | natural ASR (clean models) | Δ accuracy | Δ attack F1 | Δ PR-AUC |
|---|---|---|---|---|---|---|
| CICIDS2017 | 1D-CNN | 1.000 [0.999, 1.000] | 0.053 [0.033, 0.074] | +0.0016 [-0.0010, 0.0041] | +0.0041 [-0.0023, 0.0101] | +0.0001 [-0.0004, 0.0006] |
| CICIDS2017 | MLP | 0.992 [0.987, 0.995] | 0.032 [0.024, 0.038] | -0.0000 [-0.0026, 0.0040] | -0.0000 [-0.0059, 0.0091] | -0.0003 [-0.0005, -0.0000] |
| UNSW-NB15 | 1D-CNN | 0.957 [0.893, 0.993] | 0.006 [0.004, 0.007] | -0.0445 [-0.0641, -0.0264] | -0.0318 [-0.0450, -0.0196] | -0.0022 [-0.0031, -0.0014] |
| UNSW-NB15 | MLP | 0.906 [0.904, 0.909] | 0.002 [0.001, 0.002] | -0.0271 [-0.0334, -0.0212] | -0.0198 [-0.0241, -0.0157] | -0.0120 [-0.0127, -0.0111] |

## 3. Screening (10 clean + 10 poisoned CNN models)

| dataset | detector | ROC-AUC [95% CI] | Mann-Whitney p (one-sided), r | rule | TPR | FPR, test group | FPR, leave-one-out | target identified |
|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | Neural Cleanse | 0.820 [0.61, 0.98] | 0.0086, 0.64 | 2.0 × calibration median | 2/10 = 0.20 (exact [0.03, 0.56]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 1/10 = 0.10 (exact [0.00, 0.45]) | 1.00 |
| CICIDS2017 | MM-BD (min/max box, seeded) | 0.660 [0.38, 0.88] | 0.12, 0.32 | S > 2.00 | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0.90 |
| CICIDS2017 | MM-BD, secondary rule | | | S > 1.24 | 4/10 = 0.40 (exact [0.12, 0.74]) | 1/5 = 0.20 (exact [0.01, 0.72]) | – | |
| UNSW-NB15 | Neural Cleanse | 0.420 [0.16, 0.68] | 0.74, -0.16 | 2.0 × calibration median | 0/10 = 0.00 (exact [0.00, 0.31]) | 0/5 = 0.00 (exact [0.00, 0.52]) | 0/10 = 0.00 (exact [0.00, 0.31]) | 0.20 |
| UNSW-NB15 | MM-BD (min/max box, seeded) | 1.000 [1.00, 1.00] | 9.1e-05, 1.00 | S > 2.00 | 10/10 = 1.00 (exact [0.69, 1.00]) | 2/5 = 0.40 (exact [0.05, 0.85]) | 2/10 = 0.20 (exact [0.03, 0.56]) | 1.00 |
| UNSW-NB15 | MM-BD, secondary rule | | | S > 1.47 | 10/10 = 1.00 (exact [0.69, 1.00]) | 2/5 = 0.40 (exact [0.05, 0.85]) | – | |
- CICIDS2017: NC target-class mask, clean 0.29 vs poisoned 0.19; NC/MM-BD agreement 0.90; either flags: TPR 2/10 = 0.20 (exact [0.03, 0.56]), FPR 0/5 = 0.00 (exact [0.00, 0.52]); both flag: TPR 0/10 = 0.00 (exact [0.00, 0.31]), FPR 0/5 = 0.00 (exact [0.00, 0.52]).
- UNSW-NB15: NC target-class mask, clean 3.93 vs poisoned 1.95; NC/MM-BD agreement 0.40; either flags: TPR 10/10 = 1.00 (exact [0.69, 1.00]), FPR 2/5 = 0.40 (exact [0.05, 0.85]); both flag: TPR 0/10 = 0.00 (exact [0.00, 0.31]), FPR 0/5 = 0.00 (exact [0.00, 0.52]).

Details: `nc/metrics.json`, `mmbd/summary.md`.

## 4. Runtime detection (all 10 poisoned CNN models; phase7f runtime setting)

| dataset | detector | ROC-AUC | TPR @ 5% FPR | realized FPR | TPR @ T = 0.9 | FPR @ T = 0.9 |
|---|---|---|---|---|---|---|
| CICIDS2017 | STRIP | 0.833 [0.798, 0.861] | 0.268 [0.221, 0.313] | 0.053 | – | – |
| CICIDS2017 | IBD-PSC | 0.524 [0.480, 0.565] | 0.000 [0.000, 0.000] | 0.000 | 0.322 [0.235, 0.413] | 0.451 [0.397, 0.509] |
| CICIDS2017 | IBD-PSC adaptation | k per model [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]; ξ reached in 0/10 | | | | |
| UNSW-NB15 | STRIP | 0.286 [0.236, 0.340] | 0.023 [0.014, 0.032] | 0.049 | – | – |
| UNSW-NB15 | IBD-PSC | 0.134 [0.088, 0.186] | 0.002 [0.001, 0.002] | 0.033 | 0.321 [0.204, 0.439] | 0.868 [0.816, 0.910] |
| UNSW-NB15 | IBD-PSC adaptation | k per model [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]; ξ reached in 0/10 | | | | |

Details: `runtime/summary.md`.

## 5. Class properties on clean models (10 models per architecture)

Per-class minimal mask L1: target-excluded count-matched Neural Cleanse (phase7e holdout). STRIP entropy: blend α 0.7, N = 100, 3,000-row clean holdout, first 2,000 clean test inputs per class (protocol of the article's Figure 3, panel b). MM-BD margin: min/max box, seeded starts. Paired Wilcoxon test across models (class 0 vs class 1).

| dataset | class 0 train fraction | architecture | property | class 0 (benign/normal) | class 1 (attack) | smaller class | models with class 0 smaller | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | 0.831 | mlp | NC mask L1 | 3.318 [3.041, 3.649] | 5.730 [5.384, 6.074] | 0 | 10/10 | 0.002 |
| CICIDS2017 | 0.831 | mlp | STRIP entropy | 0.077 [0.074, 0.080] | 0.117 [0.114, 0.119] | 0 | 10/10 | 0.002 |
| CICIDS2017 | 0.831 | mlp | MM-BD margin | 149.402 [133.247, 165.893] | 103.799 [98.298, 109.804] | 1 | 0/10 | 0.002 |
| CICIDS2017 | 0.831 | cnn1d | NC mask L1 | 0.287 [0.250, 0.326] | 5.819 [4.927, 7.053] | 0 | 10/10 | 0.002 |
| CICIDS2017 | 0.831 | cnn1d | STRIP entropy | 0.066 [0.061, 0.072] | 0.177 [0.164, 0.191] | 0 | 10/10 | 0.002 |
| CICIDS2017 | 0.831 | cnn1d | MM-BD margin | 158.460 [137.362, 179.775] | 110.340 [96.961, 124.355] | 1 | 0/10 | 0.002 |
| UNSW-NB15 | 0.319 | mlp | NC mask L1 | 5.083 [4.706, 5.452] | 2.556 [2.368, 2.773] | 1 | 0/10 | 0.002 |
| UNSW-NB15 | 0.319 | mlp | STRIP entropy | 0.078 [0.074, 0.083] | 0.234 [0.229, 0.239] | 0 | 10/10 | 0.002 |
| UNSW-NB15 | 0.319 | mlp | MM-BD margin | 94.174 [82.769, 106.239] | 363.878 [308.681, 421.565] | 0 | 10/10 | 0.002 |
| UNSW-NB15 | 0.319 | cnn1d | NC mask L1 | 3.925 [3.352, 4.489] | 2.033 [1.808, 2.270] | 1 | 1/10 | 0.0039 |
| UNSW-NB15 | 0.319 | cnn1d | STRIP entropy | 0.160 [0.151, 0.167] | 0.255 [0.241, 0.268] | 0 | 10/10 | 0.002 |
| UNSW-NB15 | 0.319 | cnn1d | MM-BD margin | 120.808 [93.337, 154.604] | 374.448 [345.767, 408.127] | 0 | 10/10 | 0.002 |

Majority class: CICIDS2017 class 0; UNSW-NB15 class 1. Figure: `class_properties.png`.

Evaluation runtime 127.6 min (excluding training; see `models_manifest.json`).

## Artifacts

- `metrics.json`, `summary.md`, `manifest.json`, `models_manifest.json`, `class_properties.png`
- `nc/metrics.json`; `mmbd/` (Experiment B outputs for the CNN); `runtime/` (Experiment C outputs for the CNN)
