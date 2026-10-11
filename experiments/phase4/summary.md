# STRIP detector, tabular adaptation (CICIDS2017)

Run-time input-level backdoor detector. Thresholds are calibrated **only** on the defender's clean holdout; triggered inputs are never used for calibration.

- **Run:** `python -m src.detectors.run_strip --config configs/strip.yaml`
- **Measured:** STRIP entropy, detection AUC, and TPR at fixed FPR for two perturbation variants, the number of perturbations N, and every backdoored model; per-input compute cost.

## Two tabular perturbation variants
- **blend**: `x_mix = alpha*x + (1-alpha)*x_clean` — dilutes the trigger.
- **replace**: overwrite a random `replace_prop` fraction of features with a clean sample's values — may delete trigger features outright.

**Selected variant (highest AUC): `blend` (alpha=0.7)**, AUC 0.7235 on the representative model (p0.05).

### Parameter study (representative model, N=100)

| variant | param | AUC | TPR@5%FPR |
|---|---|---|---|
| blend | alpha=0.5 | 0.5983 | 0.0970 |
| blend | alpha=0.7 | 0.7235 | 0.0180 |
| blend | alpha=0.9 | 0.4674 | 0.0010 |
| replace | prop=0.2 | 0.4858 | 0.0085 |
| replace | prop=0.3 | 0.4787 | 0.0120 |
| replace | prop=0.5 | 0.4791 | 0.0275 |

### Effect of N (perturbations/input) on the selected config

| N | AUC | TPR@5%FPR | time/input (ms) |
|---|---|---|---|
| 20 | 0.6586 | 0.0090 | 0.02 |
| 50 | 0.7045 | 0.0175 | 0.07 |
| 100 | 0.7235 | 0.0180 | 0.14 |

## Selected config across all models

| model | poison_rate | AUC | TPR@1% | TPR@5% | TPR@10% | FPR@5%(meas) | F1@5% |
|---|---|---|---|---|---|---|---|
| clean | None | 0.5249 | 0.0310 | 0.1170 | 0.2140 | 0.0505 | 0.2004 |
| p0.005 | 0.005 | 0.0954 | 0.0000 | 0.0000 | 0.0000 | 0.0440 | 0.0000 |
| p0.01 | 0.01 | 0.4079 | 0.0000 | 0.0015 | 0.0135 | 0.0490 | 0.0029 |
| p0.02 | 0.02 | 0.4774 | 0.0000 | 0.0025 | 0.0350 | 0.0490 | 0.0048 |
| p0.05 | 0.05 | 0.7235 | 0.0000 | 0.0180 | 0.1445 | 0.0500 | 0.0337 |
| p0.1 | 0.1 | 0.7513 | 0.0275 | 0.1755 | 0.3265 | 0.0505 | 0.2863 |

**Control (clean model):** AUC 0.5249; TPR@5%FPR = 0.1170 with measured FPR 0.0505, close to the target FPR, i.e. no excess false alarms on the clean model.

## Entropy breakdown

On the representative model (p0.05), mean STRIP entropy is:

- triggered attack: **0.0184**
- clean benign: **0.0520**
- clean attack: **0.0805**

AUC vs clean **benign** only = 0.7254; AUC vs clean **attack** only = 0.6735.

The highest AUC, reached at the higher poison rates, is ~0.72–0.75, with low TPR at practical FPRs (e.g. TPR@5%FPR ≈ 0.18 at 10% poison). Two observations accompany this. (1) Clean benign and clean attack inputs also include many low-entropy inputs, so their entropy distributions overlap those of the low-entropy triggered inputs (`entropy_distribution.png`); the BENIGN target class is itself predicted stably under perturbation. (2) Detection improves with poison rate: at ≤2% the trigger does not reliably force BENIGN, triggered inputs behave like clean attacks (higher entropy), and AUC can fall below 0.5.

**Compute cost:** 0.14 ms/input (100 forward passes/input; 400,000 total for the eval set).

## Artifacts
- `metrics.json`, `results_table.csv`, `thresholds.json`
- `entropy_distribution.png`, `roc_curve.png`