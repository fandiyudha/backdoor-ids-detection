# Extended experiment C: IBD-PSC input-level runtime detection

- **Run:** `python -u -m src.eval.run_v11_ibd_psc --config configs/v11_cnn_ibd_psc.yaml`
- **Reference:** L. Hou, R. Feng, Z. Hua, W. Luo, L. Y. Zhang, Y. Li, "IBD-PSC: Input-level backdoor detection via parameter-oriented scaling consistency," ICML 2024, PMLR 235:18992–19022; reference code in BackdoorBox, adapted to tabular inputs.
- **Setting:** the runtime setting of the paper's STRIP evaluation (`phase7f`): the same 3,000-row clean holdout, the trigger stamped on the first 1,500 source-class (attack) test rows as positives, and the same 1,500 random clean test rows as negatives, on every benign/normal-target poisoned model (5 existing + 5 new per dataset). Thresholds are set on the clean holdout only. STRIP (blend, α 0.7, N = 100) is rerun on the same models and inputs.

## Adaptation to the tabular models

- Amplification: γ and β of the deepest m `BatchNorm1d` layers multiplied by ω = 1.5.
- Adaptive start k: m = 1, 2, … until the error rate (against the true labels) on the first 100 rows of the clean holdout exceeds ξ = 0.6. If ξ is never exceeded with all BN layers scaled, k = number of BN layers.
- Amplified models: model i scales the deepest k + i − 1 layers (the paper's description; BackdoorBox scales one layer more). The models have 3 BN layers, so at most 3 − k + 1 layer counts exist; the set is completed by also using ω ∈ {1.5, 2.0, 2.5}, ordered by ω and then by layer count, and the first n = 5 are kept.
- PSC(x) = mean over the amplified models of the softmax probability of the label the original model predicts (softmax in float64 to avoid ties at 1.0). Flag if PSC ≥ T (fixed threshold T = 0.9 of the IBD-PSC paper) or if PSC exceeds the 95th percentile of the clean-holdout PSC (5% FPR operating point).

- CICIDS2017: highest clean error rate of any amplified model during the selection of k (ω = 1.5): 0.56 (ξ = 0.6).
- CICIDS2017: k per model [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]; ξ exceeded in 0/10 models; amplified models per detector [3]; first model's set (3 layers, ω 1.5), (3 layers, ω 2.0), (3 layers, ω 2.5).
- UNSW-NB15: highest clean error rate of any amplified model during the selection of k (ω = 1.5): 0.44 (ξ = 0.6).
- UNSW-NB15: k per model [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]; ξ exceeded in 0/10 models; amplified models per detector [3]; first model's set (3 layers, ω 1.5), (3 layers, ω 2.0), (3 layers, ω 2.5).

## Detection (mean over poisoned models, bootstrap 95% CI over models)

| dataset | models | detector | ROC-AUC | TPR @ 5% FPR | realized FPR (negatives) | TPR @ T = 0.9 | FPR @ T = 0.9 (negatives) | per-input cost (ms) |
|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | all 10 | IBD-PSC | 0.524 [0.480, 0.565] | 0.000 [0.000, 0.000] | 0.000 | 0.322 [0.235, 0.413] | 0.451 [0.397, 0.509] | 0.078 |
| CICIDS2017 | all 10 | STRIP | 0.833 [0.798, 0.861] | 0.268 [0.221, 0.313] | 0.053 | – | – | 2.336 |
| UNSW-NB15 | all 10 | IBD-PSC | 0.134 [0.088, 0.186] | 0.002 [0.001, 0.002] | 0.033 | 0.321 [0.204, 0.439] | 0.868 [0.816, 0.910] | 0.233 |
| UNSW-NB15 | all 10 | STRIP | 0.286 [0.236, 0.340] | 0.023 [0.014, 0.032] | 0.049 | – | – | 7.221 |

No pre-existing models in this run, so there is no reproduction check against phase7f.
Holdout/negative row overlap (UNSW-NB15 draws both from the test split, as in phase7f): CICIDS2017 0 rows; UNSW-NB15 51 rows. Per-input cost is wall-clock time on the CPU, including all forward passes (IBD-PSC: amplified models + the original model; STRIP: N perturbations).

## PSC distributions (first poisoned model per dataset)

| dataset | group | n | mean | median | std | fraction ≥ T |
|---|---|---|---|---|---|---|
| CICIDS2017 | clean benign | 5000 | 0.3873 | 0.1447 | 0.4315 | 0.295 |
| CICIDS2017 | clean attack | 5000 | 0.9908 | 1.0000 | 0.0942 | 0.990 |
| CICIDS2017 | triggered | 5000 | 0.4807 | 0.3334 | 0.2714 | 0.165 |
| UNSW-NB15 | clean benign | 5000 | 0.9946 | 1.0000 | 0.0681 | 0.993 |
| UNSW-NB15 | clean attack | 5000 | 0.9741 | 1.0000 | 0.1301 | 0.954 |
| UNSW-NB15 | triggered | 5000 | 0.4999 | 0.3599 | 0.3943 | 0.314 |

ROC-AUC triggered vs clean benign/normal on these inputs: CICIDS2017 0.639; UNSW-NB15 0.032. Figures: `psc_distributions.png`, `psc_roc.png`; raw scores: `psc_raw_<dataset>.csv`.

Total runtime 10.0 min.

## Artifacts

- `metrics.json`, `summary.md`, `manifest.json`
- `psc_distributions.png`, `psc_roc.png`, `controlled_conditions.png`, `psc_raw_<dataset>.csv`
