# Extended experiment C: IBD-PSC input-level runtime detection

- **Run:** `python -u -m src.eval.run_v11_ibd_psc --config configs/v11_ibd_psc.yaml`
- **Reference:** L. Hou, R. Feng, Z. Hua, W. Luo, L. Y. Zhang, Y. Li, "IBD-PSC: Input-level backdoor detection via parameter-oriented scaling consistency," ICML 2024, PMLR 235:18992–19022; reference code in BackdoorBox, adapted to tabular inputs.
- **Setting:** the runtime setting of the paper's STRIP evaluation (`phase7f`): the same 3,000-row clean holdout, the trigger stamped on the first 1,500 source-class (attack) test rows as positives, and the same 1,500 random clean test rows as negatives, on every benign/normal-target poisoned model (5 existing + 5 new per dataset). Thresholds are set on the clean holdout only. STRIP (blend, α 0.7, N = 100) is rerun on the same models and inputs.

## Adaptation to the tabular models

- Amplification: γ and β of the deepest m `BatchNorm1d` layers multiplied by ω = 1.5.
- Adaptive start k: m = 1, 2, … until the error rate (against the true labels) on the first 100 rows of the clean holdout exceeds ξ = 0.6. If ξ is never exceeded with all BN layers scaled, k = number of BN layers.
- Amplified models: model i scales the deepest k + i − 1 layers (the paper's description; BackdoorBox scales one layer more). The models have 3 BN layers, so at most 3 − k + 1 layer counts exist; the set is completed by also using ω ∈ {1.5, 2.0, 2.5}, ordered by ω and then by layer count, and the first n = 5 are kept.
- PSC(x) = mean over the amplified models of the softmax probability of the label the original model predicts (softmax in float64 to avoid ties at 1.0). Flag if PSC ≥ T (fixed threshold T = 0.9 of the IBD-PSC paper) or if PSC exceeds the 95th percentile of the clean-holdout PSC (5% FPR operating point).

- CICIDS2017: highest clean error rate of any amplified model during the selection of k (ω = 1.5): 0.07 (ξ = 0.6).
- CICIDS2017: k per model [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]; ξ exceeded in 0/10 models; amplified models per detector [3]; first model's set (3 layers, ω 1.5), (3 layers, ω 2.0), (3 layers, ω 2.5).
- UNSW-NB15: highest clean error rate of any amplified model during the selection of k (ω = 1.5): 0.17 (ξ = 0.6).
- UNSW-NB15: k per model [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]; ξ exceeded in 0/10 models; amplified models per detector [3]; first model's set (3 layers, ω 1.5), (3 layers, ω 2.0), (3 layers, ω 2.5).

## Detection (mean over poisoned models, bootstrap 95% CI over models)

| dataset | models | detector | ROC-AUC | TPR @ 5% FPR | realized FPR (negatives) | TPR @ T = 0.9 | FPR @ T = 0.9 (negatives) | per-input cost (ms) |
|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | existing 5 | IBD-PSC | 0.326 [0.283, 0.372] | 0.001 [0.000, 0.004] | 0.013 | 0.979 [0.972, 0.984] | 0.951 [0.947, 0.953] | 0.003 |
| CICIDS2017 | existing 5 | STRIP | 0.621 [0.553, 0.681] | 0.009 [0.000, 0.025] | 0.039 | – | – | 0.225 |
| CICIDS2017 | new 5 | IBD-PSC | 0.421 [0.338, 0.498] | 0.001 [0.000, 0.004] | 0.016 | 0.972 [0.961, 0.984] | 0.953 [0.948, 0.959] | 0.008 |
| CICIDS2017 | new 5 | STRIP | 0.642 [0.591, 0.698] | 0.040 [0.005, 0.090] | 0.038 | – | – | 0.177 |
| CICIDS2017 | all 10 | IBD-PSC | 0.373 [0.322, 0.431] | 0.001 [0.000, 0.003] | 0.015 | 0.976 [0.968, 0.982] | 0.952 [0.949, 0.955] | 0.005 |
| CICIDS2017 | all 10 | STRIP | 0.632 [0.590, 0.671] | 0.024 [0.005, 0.051] | 0.039 | – | – | 0.201 |
| UNSW-NB15 | existing 5 | IBD-PSC | 0.221 [0.203, 0.241] | 0.003 [0.003, 0.004] | 0.052 | 0.595 [0.529, 0.683] | 0.793 [0.787, 0.797] | 0.004 |
| UNSW-NB15 | existing 5 | STRIP | 0.235 [0.224, 0.244] | 0.004 [0.004, 0.005] | 0.047 | – | – | 0.336 |
| UNSW-NB15 | new 5 | IBD-PSC | 0.186 [0.164, 0.202] | 0.003 [0.003, 0.003] | 0.045 | 0.436 [0.308, 0.536] | 0.790 [0.785, 0.794] | 0.004 |
| UNSW-NB15 | new 5 | STRIP | 0.255 [0.251, 0.260] | 0.004 [0.003, 0.004] | 0.047 | – | – | 0.370 |
| UNSW-NB15 | all 10 | IBD-PSC | 0.204 [0.185, 0.221] | 0.003 [0.003, 0.003] | 0.048 | 0.515 [0.426, 0.596] | 0.792 [0.788, 0.795] | 0.004 |
| UNSW-NB15 | all 10 | STRIP | 0.245 [0.237, 0.253] | 0.004 [0.004, 0.004] | 0.047 | – | – | 0.353 |

STRIP reproduction check on the existing models against `phase7f` (α 0.7): CICIDS2017 max abs. difference 0.0e+00; UNSW-NB15 max abs. difference 0.0e+00.
Holdout/negative row overlap (UNSW-NB15 draws both from the test split, as in phase7f): CICIDS2017 0 rows; UNSW-NB15 51 rows. Per-input cost is wall-clock time on the CPU, including all forward passes (IBD-PSC: amplified models + the original model; STRIP: N perturbations).

## PSC distributions (first poisoned model per dataset)

| dataset | group | n | mean | median | std | fraction ≥ T |
|---|---|---|---|---|---|---|
| CICIDS2017 | clean benign | 5000 | 0.9713 | 1.0000 | 0.1082 | 0.935 |
| CICIDS2017 | clean attack | 5000 | 0.9811 | 1.0000 | 0.0813 | 0.959 |
| CICIDS2017 | triggered | 5000 | 0.9877 | 1.0000 | 0.0845 | 0.980 |
| UNSW-NB15 | clean benign | 5000 | 0.9939 | 1.0000 | 0.0468 | 0.983 |
| UNSW-NB15 | clean attack | 5000 | 0.9482 | 0.9998 | 0.1314 | 0.873 |
| UNSW-NB15 | triggered | 5000 | 0.8468 | 0.9666 | 0.2290 | 0.679 |

ROC-AUC triggered vs clean benign/normal on these inputs: CICIDS2017 0.359; UNSW-NB15 0.039. Figures: `psc_distributions.png`, `psc_roc.png`; raw scores: `psc_raw_<dataset>.csv`.

## CICIDS2017 controlled conditions (phase6c/phase6d checkpoints, 5 models each)

Same draws as phase6d: 3,000-row validation holdout, 2,000 random clean test rows as negatives, the trigger on the first 2,000 source-class test rows as positives.

| condition | target | clean target-class STRIP entropy (phase6c) | clean target-class PSC | IBD-PSC ROC-AUC | IBD-PSC TPR @ 5% FPR | IBD-PSC TPR @ T | STRIP TPR @ 5% FPR (phase6d) | k per model |
|---|---|---|---|---|---|---|---|---|
| A_balanced_tgtBenign | benign | 0.049 | 0.968 [0.966, 0.970] | 0.503 [0.472, 0.533] | 0.009 [0.002, 0.019] | 0.996 [0.996, 0.998] | 0.033 | [3, 3, 3, 3, 3] |
| B_imbal_tgtBenign | benign | 0.079 | 0.971 [0.967, 0.975] | 0.330 [0.290, 0.373] | 0.001 [0.000, 0.003] | 0.979 [0.972, 0.984] | 0.008 | [3, 3, 3, 3, 3] |
| C_imbal_tgtAttack | attack | 0.122 | 0.975 [0.974, 0.976] | 0.754 [0.709, 0.790] | 0.000 [0.000, 0.000] | 0.992 [0.991, 0.994] | 0.068 | [3, 3, 3, 3, 3] |

Figure: `controlled_conditions.png`.

Total runtime 0.7 min.

## Artifacts

- `metrics.json`, `summary.md`, `manifest.json`
- `psc_distributions.png`, `psc_roc.png`, `controlled_conditions.png`, `psc_raw_<dataset>.csv`
