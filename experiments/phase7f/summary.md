# STRIP re-evaluation: blending-weight sweep, class-conditioned threshold, valid perturbation, and benign-conversion-rate baseline

Reruns STRIP over existing checkpoints (no retraining) and reports whether any variant substantially improves input-level detection.

- **Run:** `python -m src.eval.run_phase7f_strip_reeval --config configs/phase7f.yaml`
- **Measured:** STRIP TPR at fixed FPR and AUC across blending weights, a class-conditioned threshold, and alternative perturbations; the Ma et al. benign-conversion-rate baseline; and the measured per-input entropy distributions on CICIDS2017.

## Overall result

**Across both datasets and every input-level runtime variant tried (STRIP α∈[0.3, 0.5, 0.7, 0.9], a class-conditioned threshold, the `replace` and validity-preserving `marginal` perturbations, and the Ma et al. benign-conversion-rate baseline), TPR@5%FPR never exceeds 0.072, so no simple input-level runtime detector reaches a usable operating point on this all-to-benign backdoor. Two qualifications: (1) a weak ranking signal remains: the best AUC is 0.827, and on the measured entropy distributions triggered inputs have lower entropy than clean-benign inputs (ROC-AUC 0.628), i.e. they are not cleanly separable at a usable FPR; (2) the failure is operational (at a usable FPR), not a strict impossibility of any input-level signal. Screening-level Neural Cleanse remains the effective detector for this threat.**

## Part 1a — STRIP blend-alpha sweep (main benign/normal-target scenario)

| dataset | α=0.3 (TPR@5% / AUC) | α=0.5 (TPR@5% / AUC) | α=0.7 (TPR@5% / AUC) | α=0.9 (TPR@5% / AUC) | best | weak across all α? |
|---|---|---|---|---|---|---|
| CICIDS2017 | 0.035±0.041 / 0.441 | 0.072±0.056 / 0.552 | 0.009±0.018 / 0.621 | 0.001±0.003 / 0.454 | alpha_0.5 | True |
| UNSW-NB15 | 0.009±0.007 / 0.369 | 0.005±0.002 / 0.235 | 0.004±0.001 / 0.235 | 0.003±0.000 / 0.294 | alpha_0.3 | True |

## Part 1b — Class-conditioned threshold vs global

| dataset | global TPR@5% | class-cond TPR (FPR) | improves? |
|---|---|---|---|
| CICIDS2017 | 0.009±0.018 | 0.011 (0.041) | True |
| UNSW-NB15 | 0.004±0.001 | 0.004 (0.047) | False |

## Part 2 — Validity-preserving perturbation vs old blend

_blend interpolates -> can produce out-of-support / non-integer feature values and break relations like min<=mean<=max; `marginal` resamples each feature from its empirical marginal (real observed values), and `replace` copies a real partner's subset — both keep per-feature validity._

| dataset | blend (old) TPR@5%/AUC | replace TPR@5%/AUC | marginal (valid) TPR@5%/AUC | best |
|---|---|---|---|---|
| CICIDS2017 | 0.009 / 0.621 | 0.011 / 0.482 | 0.022 / 0.517 | marginal(valid) |
| UNSW-NB15 | 0.004 / 0.235 | 0.003 / 0.289 | 0.003 / 0.264 | blend(old) |

## Part 3 — Benign-conversion-rate baseline (Ma et al.) vs STRIP vs NC

| dataset | BCR (Ma) TPR@5%/AUC [runtime] | STRIP best-α TPR@5%/AUC [runtime] | Neural Cleanse AUC [screening] |
|---|---|---|---|
| CICIDS2017 | 0.000 / 0.827 | 0.072 / 0.552 | 1.000 (NC re-evaluation, per-model) |
| UNSW-NB15 | 0.003 / 0.569 | 0.009 / 0.369 | 0.840 (NC re-evaluation, per-model) |

BCR and STRIP are both **runtime** (per-input); Neural Cleanse is **screening** (per-model) — a different regime, shown for context only.

## Part 4 — Measured per-input entropy distributions

Computed on CICIDS2017 benign-target (representative model). Sample sizes: clean-benign n=5000, clean-attack n=5000, triggered n=5000.

| group | entropy mean | median | std |
|---|---|---|---|
| clean_benign | 0.079 | 0.043 | 0.095 |
| clean_attack | 0.119 | 0.035 | 0.144 |
| triggered | 0.048 | 0.030 | 0.065 |

**ROC-AUC (triggered vs clean-benign) = 0.628** (Cohen's d -0.38). Overlap: **partial overlap (separable but not cleanly)**.

Figures are drawn from the measured per-input values: `fig3_entropy_distribution.png` (histogram + empirical KDE) and `fig3_roc.png`; raw values in `fig3_entropy_raw.csv`.

## Artifacts
- `metrics.json`, `summary.md`
- `fig3_entropy_distribution.png`, `fig3_roc.png`, `fig3_entropy_raw.csv`