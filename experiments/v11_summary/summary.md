# Extended experiments (v1.1): consolidated tables

- **Run:** `python -m src.eval.v11_report` (reads committed metrics files only).
- **Models:** per dataset and architecture, 10 clean and 10 benign/normal-target poisoned models (MLP: 5 v1.0 models + 5 new seeds; 1D-CNN: 10 new seeds). Model-level AUCs compare 10 clean with 10 poisoned models (stratified bootstrap 95% CI, 10,000 resamples). Runtime AUCs and TPRs are means over the 10 poisoned models (bootstrap 95% CI over models). Counts are k/n models.

## 1. Detection: all detectors, both datasets, both architectures

Pre-specified rules. Neural Cleanse (NC): flag if the mask-norm ratio > 2.0 × the median of the 5 calibration clean models. MM-BD primary: flag if S = max_c r_c / calibration median > 2.0; secondary: flag if S exceeds the largest leave-one-out S among the calibration models. FPR = 5 disjoint clean test models (and leave-one-out over all 10 clean models where given). STRIP and IBD-PSC: TPR at the 5% FPR threshold of the clean holdout (realized FPR on clean test inputs); IBD-PSC also at its fixed T = 0.9.

| dataset × architecture | Neural Cleanse | MM-BD, min/max box | MM-BD, 1st–99th percentile box | STRIP (α 0.7) | IBD-PSC |
|---|---|---|---|---|---|
| CICIDS2017 × MLP | AUC 1.00 [1.00, 1.00]<br>2.0×: TPR 10/10, FPR 0/5 (LOO 0/10) | AUC 0.98 [0.91, 1.00]<br>S > 2.0: TPR 0/10, FPR 0/5 (LOO 0/10)<br>LOO-max (S > 1.33): TPR 9/10, FPR 0/5 | AUC 0.92 [0.77, 1.00]<br>S > 2.0: TPR 0/10, FPR 0/5 (LOO 0/10)<br>LOO-max (S > 1.18): TPR 9/10, FPR 3/5 | AUC 0.63 [0.59, 0.67]<br>TPR@5% 0.024 (FPR 0.039) | AUC 0.37 [0.32, 0.43]<br>TPR@5% 0.001 (FPR 0.015)<br>T = 0.9: TPR 0.98, FPR 0.95 |
| CICIDS2017 × 1D-CNN | AUC 0.82 [0.61, 0.98]<br>2.0×: TPR 2/10, FPR 0/5 (LOO 1/10) | AUC 0.66 [0.38, 0.88]<br>S > 2.0: TPR 0/10, FPR 0/5 (LOO 0/10)<br>LOO-max (S > 1.24): TPR 4/10, FPR 1/5 | AUC 0.61 [0.34, 0.86]<br>S > 2.0: TPR 0/10, FPR 0/5 (LOO 0/10)<br>LOO-max (S > 1.19): TPR 1/10, FPR 0/5 | AUC 0.83 [0.80, 0.86]<br>TPR@5% 0.268 (FPR 0.053) | AUC 0.52 [0.48, 0.56]<br>TPR@5% 0.000 (FPR 0.000)<br>T = 0.9: TPR 0.32, FPR 0.45 |
| UNSW-NB15 × MLP | AUC 0.92 [0.73, 1.00]<br>2.0×: TPR 9/10, FPR 0/5 (LOO 0/10) | AUC 0.98 [0.91, 1.00]<br>S > 2.0: TPR 0/10, FPR 0/5 (LOO 0/10)<br>LOO-max (S > 1.37): TPR 9/10, FPR 1/5 | AUC 0.89 [0.71, 1.00]<br>S > 2.0: TPR 0/10, FPR 0/5 (LOO 0/10)<br>LOO-max (S > 1.26): TPR 8/10, FPR 2/5 | AUC 0.25 [0.24, 0.25]<br>TPR@5% 0.004 (FPR 0.047) | AUC 0.20 [0.19, 0.22]<br>TPR@5% 0.003 (FPR 0.048)<br>T = 0.9: TPR 0.52, FPR 0.79 |
| UNSW-NB15 × 1D-CNN | AUC 0.42 [0.16, 0.68]<br>2.0×: TPR 0/10, FPR 0/5 (LOO 0/10) | AUC 1.00 [1.00, 1.00]<br>S > 2.0: TPR 10/10, FPR 2/5 (LOO 2/10)<br>LOO-max (S > 1.47): TPR 10/10, FPR 2/5 | AUC 1.00 [1.00, 1.00]<br>S > 2.0: TPR 6/10, FPR 0/5 (LOO 0/10)<br>LOO-max (S > 1.15): TPR 10/10, FPR 0/5 | AUC 0.29 [0.24, 0.34]<br>TPR@5% 0.023 (FPR 0.049) | AUC 0.13 [0.09, 0.19]<br>TPR@5% 0.002 (FPR 0.033)<br>T = 0.9: TPR 0.32, FPR 0.87 |

Target class identified (smallest NC mask / largest MM-BD relative margin, poisoned models):

| dataset × architecture | NC | MM-BD, min/max | MM-BD, 1st–99th | MM-BD min/max AUC, pre-specified random starts |
|---|---|---|---|---|
| CICIDS2017 × MLP | 1.00 | 0.20 | 0.20 | 0.87 |
| CICIDS2017 × 1D-CNN | 1.00 | 0.90 | 0.90 | 0.66 |
| UNSW-NB15 × MLP | 0.00 | 1.00 | 1.00 | undefined (margins ≤ 0) |
| UNSW-NB15 × 1D-CNN | 0.20 | 1.00 | 1.00 | 1.00 |

MM-BD uses holdout-seeded starts (a post-hoc correction of Experiment B, applied everywhere); the last column is the pre-specified random-start estimate. Two-sided percentile rule (exploratory, MLP only): CICIDS2017 STRIP TPR 0.012 (FPR 0.042), IBD-PSC TPR 0.016 (FPR 0.024); UNSW-NB15 STRIP TPR 0.056 (FPR 0.049), IBD-PSC TPR 0.117 (FPR 0.040).

## 2. Target-class contrast (natural vs poisoned)

Mean (SD) over 10 clean and 10 poisoned models. Target class = class 0 (benign/normal). NC: minimal mask L1 (target-excluded, count-matched; smaller = easier to reach). MM-BD: maximum margin r_c (min/max box, seeded; larger = easier to push to that class).

| dataset × architecture | NC mask, target: clean → poisoned | NC mask, non-target: clean → poisoned | NC ratio range: clean / poisoned | MM-BD margin, target: clean → poisoned | MM-BD margin, non-target: clean → poisoned |
|---|---|---|---|---|---|
| CICIDS2017 × MLP | 3.32 (0.52) → 0.48 (0.09) | 5.73 (0.59) → 6.70 (1.20) | 1.1–2.2 / 8.3–18.6 | 149 (28) → 166 (61) | 104 (10) → 134 (23) |
| CICIDS2017 × 1D-CNN | 0.29 (0.06) → 0.19 (0.03) | 5.82 (1.90) → 6.23 (1.57) | 11.4–45.1 / 19.4–56.1 | 158 (36) → 196 (48) | 110 (23) → 87 (26) |
| UNSW-NB15 × MLP | 5.08 (0.64) → 15.13 (4.67) | 2.56 (0.34) → 2.98 (0.60) | 1.3–2.3 / 1.9–9.5 | 94 (20) → 164 (20) | 364 (97) → 412 (92) |
| UNSW-NB15 × 1D-CNN | 3.93 (0.97) → 1.95 (0.38) | 2.03 (0.40) → 1.19 (0.47) | 1.1–3.0 / 1.1–3.3 | 121 (52) → 344 (86) | 374 (53) → 332 (104) |

## 3. Headline numbers: v1.0 (frozen files) vs v1.1

| quantity | dataset | v1.0 | v1.1 MLP (n = 10 + 10) | v1.1 1D-CNN (n = 10 + 10) |
|---|---|---|---|---|
| NC screening ROC-AUC [95% CI] | CICIDS2017 | 1.000 [1.00, 1.00] (5 + 5) | 1.000 [1.00, 1.00] | 0.820 [0.61, 0.98] |
| NC flag rate / FPR test group / FPR LOO | CICIDS2017 | 1.00 / 0.00 (of 2) / 0.00 (of 5) | 10/10 / 0/5 / 0/10 | 2/10 / 0/5 / 1/10 |
| NC Mann-Whitney p (one-sided), rank-biserial r | CICIDS2017 | 0.004, 1.00 (recomputed) | 9.1e-05, 1.00 | 0.0086, 0.64 |
| STRIP TPR @ 5% FPR (α 0.7) / best variant | CICIDS2017 | 0.009 / 0.072 | 0.024 [0.005, 0.051] | 0.268 [0.221, 0.313] |
| ASR | CICIDS2017 | 0.992 ± 0.008 (6 full-data seeds, phase7) | 0.992 [0.987, 0.995] | 1.000 [0.999, 1.000] |
| clean-accuracy shift (poisoned − clean) | CICIDS2017 | – | -0.0000 [-0.0026, 0.0040] | +0.0016 [-0.0010, 0.0041] |
| NC screening ROC-AUC [95% CI] | UNSW-NB15 | 0.840 [0.52, 1.00] (5 + 5) | 0.920 [0.73, 1.00] | 0.420 [0.16, 0.68] |
| NC flag rate / FPR test group / FPR LOO | UNSW-NB15 | 0.80 / 0.00 (of 2) / 0.00 (of 5) | 9/10 / 0/5 / 0/10 | 0/10 / 0/5 / 0/10 |
| NC Mann-Whitney p (one-sided), rank-biserial r | UNSW-NB15 | 0.048, 0.68 (recomputed) | 0.00085, 0.84 | 0.74, -0.16 |
| STRIP TPR @ 5% FPR (α 0.7) / best variant | UNSW-NB15 | 0.004 / 0.009 | 0.004 [0.004, 0.004] | 0.023 [0.014, 0.032] |
| ASR | UNSW-NB15 | 0.907 (5 seeds, phase7d) | 0.906 [0.904, 0.909] | 0.957 [0.893, 0.993] |
| clean-accuracy shift (poisoned − clean) | UNSW-NB15 | – | -0.0271 [-0.0334, -0.0212] | -0.0445 [-0.0641, -0.0264] |

## 4. Class properties on clean models (findings d–f)

| dataset (class-0 share) | property | v1.0 Figure 3 (n = 5): benign/normal vs attack | MLP n = 10: class 0 vs 1 | 1D-CNN n = 10: class 0 vs 1 |
|---|---|---|---|---|
| CICIDS2017 (0.83) | NC minimal mask (smaller = easier to reach) | 3.197 vs 6.287 | 3.318 vs 5.730 (10/10 models class 0 smaller, p = 0.002) | 0.287 vs 5.819 (10/10, p = 0.002) |
| CICIDS2017 (0.83) | STRIP entropy (smaller = more homogeneous) | 0.047 vs 0.081 | 0.077 vs 0.117 (10/10 models class 0 smaller, p = 0.002) | 0.066 vs 0.177 (10/10, p = 0.002) |
| CICIDS2017 (0.83) | MM-BD margin (larger = easier to reach) | – | 149.402 vs 103.799 (0/10 models class 0 smaller, p = 0.002) | 158.460 vs 110.340 (0/10, p = 0.002) |
| UNSW-NB15 (0.32) | NC minimal mask (smaller = easier to reach) | 5.275 vs 2.808 | 5.083 vs 2.556 (0/10 models class 0 smaller, p = 0.002) | 3.925 vs 2.033 (1/10, p = 0.0039) |
| UNSW-NB15 (0.32) | STRIP entropy (smaller = more homogeneous) | 0.084 vs 0.237 | 0.078 vs 0.234 (10/10 models class 0 smaller, p = 0.002) | 0.160 vs 0.255 (10/10, p = 0.002) |
| UNSW-NB15 (0.32) | MM-BD margin (larger = easier to reach) | – | 94.174 vs 363.878 (10/10 models class 0 smaller, p = 0.002) | 120.808 vs 374.448 (10/10, p = 0.002) |

MLP n = 10 (CICIDS2017) uses the condition-B clean models of the screening experiments, while the v1.0 CICIDS2017 entropy bar uses the clean reference models (`clean_entropy`); the UNSW-NB15 MLP entropies of seeds 42–46 reproduce phase7d exactly.

## 5. Compute and checkpoints

- Training, MLP (Experiment A): 0.50 h
- Training, 1D-CNN screening (Experiment D): 8.05 h
- Training, 1D-CNN full-data baseline (Experiment D.1): 2.33 h
- Evaluation, A (NC, n = 10 + 10): 4.1 min
- Evaluation, B (MM-BD): 12.1 min
- Evaluation, B diagnostic (exploratory NC): 2.1 min
- Evaluation, C (IBD-PSC + STRIP): 0.7 min
- Evaluation, C checks: 0.7 min
- Evaluation, D (evaluation): 127.6 min
- **Total: 13.3 h of CPU wall-clock time; 63 new checkpoints** (20 MLP, 40 1D-CNN screening, 3 1D-CNN full-data baseline).

## Artifacts

- `metrics.json`, `summary.md`
