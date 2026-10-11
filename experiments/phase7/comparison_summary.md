# STRIP vs Neural Cleanse: comparison under the imported-model threat model (CICIDS2017)

Two detectors at two granularities, compared separately for **screening** (is this model backdoored?) and **runtime** (does this input carry a trigger?).

- **Run:** `python -m src.eval.train_phase7_models --config configs/model.yaml`, then `python -m src.eval.run_comparison --config configs/comparison.yaml`
- **Measured:** screening separation of clean vs poisoned models, runtime detection, multi-seed variability, a trigger-design and poison-rate robustness sweep, compute cost, and layered-defence coverage.

## 1. Screening vs runtime

### Screening (pre-deployment, clean data only)
- **Neural Cleanse can screen** (AUC 1.00): Neural Cleanse separates clean from poisoned models perfectly (AUC 1.00): clean anomaly ratios ≤ 4.7 vs poisoned ≥ 16.3 (margin 11.6). The ABSOLUTE threshold is holdout/reference-sensitive (clean ratios here run higher than the Phase-6 calibration), so it must be set against the defender's own reference distribution — but the separation itself is robust.
- **STRIP cannot screen (structural):** STRIP CANNOT be operated as a model screener under this threat model: its detection needs triggered inputs, which the defender does not have pre-deployment. A clean-data proxy (mean clean-input entropy) shows only a tiny, incidental shift (healthy 0.0500 vs poisoned 0.0565, overlapping ranges); even the ORACLE separation (AUC 0.87, using true labels + best direction) is unusable because the defender has no reference model, no known direction, and no triggered inputs to calibrate. Structural limit, not an implementation failure. See `screening_limit.png`.

### Runtime (per-input) on the 5%-poison model

| detector | AUC | TPR@5%FPR | FPR@5% |
|---|---|---|---|
| STRIP | 0.711 | 0.019 | 0.044 |
| NC-derived filter | 0.898 | 0.000 | 0.066 |

Neural Cleanse's reverse-engineered trigger also yields a runtime input filter, so the split is not simply 'STRIP=runtime, NC=screening'.

## 2. Statistical robustness (multi-seed, n=6)

| metric | mean ± std (range) |
|---|---|
| ASR | 0.992 ± 0.008 [0.977, 0.998] |
| NC anomaly ratio | 30.962 ± 6.973 [21.190, 40.315] |
| NC flag rate | 1.000 ± 0.000 [1.000, 1.000] |
| STRIP AUC | 0.635 ± 0.059 [0.575, 0.711] |
| STRIP TPR@5% | 0.025 ± 0.026 [0.000, 0.069] |
| clean NC ratio | 4.036 ± 0.725 [2.789, 4.694] |

## 3. Robustness sweep (trigger design + poison rate)

| config | features | ASR | clean F1 | NC ratio | NC flag | STRIP AUC | STRIP TPR@5% |
|---|---|---|---|---|---|---|---|
| nfeat4_baseline | 4 | 0.994 | 0.916 | 30.78 | True | 0.711 | 0.019 |
| nfeat2 | 2 | 0.963 | 0.913 | 17.45 | True | 0.496 | 0.002 |
| nfeat6 | 6 | 0.989 | 0.912 | 33.36 | True | 0.524 | 0.006 |
| nfeat8 | 8 | 0.995 | 0.914 | 21.38 | True | 0.672 | 0.069 |
| altfeat | 4 | 0.798 | 0.912 | 10.76 | True | 0.591 | 0.018 |
| subtle | 4 | 0.912 | 0.914 | 15.58 | True | 0.417 | 0.042 |
| p0.005 | 4 | 0.656 | 0.911 | 16.28 | True | 0.094 | 0.000 |
| p0.01 | 4 | 0.960 | 0.918 | 20.38 | True | 0.385 | 0.000 |
| p0.02 | 4 | 0.972 | 0.913 | 34.82 | True | 0.466 | 0.001 |
| p0.05 | 4 | 0.994 | 0.916 | 30.78 | True | 0.711 | 0.019 |
| p0.1 | 4 | 0.997 | 0.916 | 23.82 | True | 0.742 | 0.163 |

**Hardest-to-detect config: `altfeat`** (NC ratio 10.76, STRIP TPR@5% 0.018, ASR 0.798).

## 4. Amortized compute cost

STRIP 0.14 ms/input (N=100); Neural Cleanse ~3.0s once per model.

| traffic (flows/s) | STRIP core-equivalents (all traffic) | feasible on 1 core? |
|---|---|---|
| 1000 | 0.14 | True |
| 10000 | 1.4 | False |
| 100000 | 14.0 | False |

_Neural Cleanse is a one-off per model (~3s), amortized to ~0 over the model's lifetime. STRIP recurs per input; running it on ALL traffic needs the core-equivalents above, so beyond ~7142 flows/s per core it must run on a SAMPLE of traffic._

## 5. Layered defence (NC gate → STRIP runtime)

- Coverage (caught by ≥1 layer): **100%** of 11 configs; evaded both layers: 0.
- _Neural Cleanse (screening gate) flags every backdoored model before deployment, so screening alone already covers all tested configs; STRIP (runtime) adds defence-in-depth and forensic localization for triggers that might evade the gate._

## 6. Master comparison table

| criterion | STRIP | Neural Cleanse |
|---|---|---|
| granularity | per-input (runtime) | per-model |
| threat_model_fit_screening | NO — needs triggered inputs the defender lacks | YES — white-box + clean data only |
| can_screen_pre_deploy | False | True |
| can_monitor_runtime | True | YES (via reverse-engineered-trigger filter) |
| runtime_detection_tpr@5fpr_p0.05 | 0.019 | 0.0 |
| cost | per input (0.14 ms x N=100) | per model (~3 s, one-off) |
| key_limitation | degrades when the target is the dominant class (Phase 6b) and cannot screen models | anomaly threshold needs a clean reference distribution; borderline on well-trained image models |

## Research questions

- **RQ1_attack_effectiveness:** Backdoor is potent and stealthy: ASR reaches ~0.96 at 1% poison and 0.992±0.008 at 5% (multi-seed), with clean-test attack F1 unchanged (< 0.01 drop).
- **RQ2_detector_adaptation:** Both adapt to the tabular network domain. Neural Cleanse works cleanly (clean anomaly ratio 4.04±0.72 vs poisoned 30.96±6.97); STRIP is only moderate on this all-to-benign backdoor (input-level, target-class dominance).
- **RQ3_comparison:** Different niches, not a single winner. Neural Cleanse fits the shadow-AI screening question (accurate, one-off, clean-data-only) and can also be adapted to runtime; STRIP is a cheap-ish per-input monitor but cannot screen and weakens under target-class dominance. A layered NC-then-STRIP defence covers 100% of tested attack configs.

## Artifacts
- `comparison_metrics.json`, `comparison_summary.md`
- `screening_limit.png`, `robustness.png`, `detector_contrast.png`