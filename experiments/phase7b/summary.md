# Neural Cleanse threshold calibration (paired, reference-relative rule)

The clean-model anomaly ratio drifts across conditions, so a single absolute threshold (3.688, from the distribution-based calibration in `src.eval.run_image_baseline`) is not stable: clean ratios in the comparison run reached 4.24. This experiment diagnoses the drift and replaces the absolute rule with a paired, shift-robust one.

- **Run:** `python -m src.eval.run_nc_calibration --config configs/nc_calibration.yaml`
- **Measured:** holdout- vs seed-induced variance of the clean anomaly ratio, threshold-free clean-vs-poisoned AUC, and TPR/FPR of three decision rules under paired calibration.

## 1. Variance diagnosis (same models, many holdouts)

- Clean-ratio grand mean: **2.49** (range 1.46–4.24).
- **Holdout-induced std: 0.347** (same model, different random holdout).
- **Seed/model-induced std: 0.584** (different training seed, same holdout).

Holdout choice alone moves the ratio by a comparable amount to the training seed, so the absolute threshold is unstable **because the measurement condition (holdout) shifts it**, not only because models differ.

## 2. Threshold-free separation

- Mean per-holdout clean-vs-poisoned **AUC = 0.994** (min 0.978); pooled AUC 0.990.
- Because AUC is threshold-free, it is unaffected by the drift and is used as the primary metric.

## 3. Decision rules under PAIRED calibration

Reference clean models are measured with the SAME holdout as the suspect; clean scored leave-one-out, poisoned vs all clean refs.

| rule | threshold | TPR | FPR | max clean FPR over holdouts | margin |
|---|---|---|---|---|---|
| zscore | 3.00 | 0.978 | 0.083 | 0.167 | -3.22 |
| rel_median | 2.00 | 0.978 | 0.000 | 0.000 | -0.73 |
| absolute | 3.69 | 0.978 | 0.028 | 0.167 | -2.11 |

- **rel-median (paired)** keeps FPR at 0 on every holdout — the most shift-robust rule.
- **z-score (paired)** also normalizes per holdout but is noisier (FPR 0.083) because the std of only ~5 reference models is unstable.
- **absolute** flags clean models on some holdouts (drift) — max clean FPR 0.167.

## 4. Recommendation

**Recommended: `rel_median`** (paired calibration). Paired calibration (reference measured with the same holdout as the suspect) removes the cross-condition drift. `rel_median` keeps FPR = 0.000 with 0 false positives on every holdout while detecting 97.8% of poisoned measurements. The absolute rule is not recommended: under holdout drift it flags clean models (up to 17% of clean models per holdout). No rule reaches TPR = 1.0 because the weakest poisoned model (a low-ASR variant) slightly overlaps the strongest clean model on some holdouts; the threshold-free separation AUC is therefore the primary metric.
TPR 0.978, FPR 0.000, margin -0.73.

### Remaining limitation
The defender still needs a few **own** clean reference models to form the paired reference distribution (shadow-AI defenders may have only one). The threshold-free AUC and the paired rule both assume access to that small reference set; without it, only relative/extreme-outlier heuristics remain.

## Artifacts
- `metrics.json`, `summary.md`, `nc_ratio_distribution.png`