# Neural Cleanse re-evaluation: target-excluded, count-matched optimization

Reruns NC optimization over existing checkpoints (no victim retraining) to test whether the smaller-mask-on-easy-class pattern is a prevalence artefact of the source set, or intrinsic.

- **Run:** `python -m src.eval.run_phase7e_nc_reeval --config configs/phase7e.yaml`
- **Measured:** per-class mask norms under all-input vs target-excluded, count-matched optimization; FPR on a disjoint calibration/test split of clean models and leave-one-out; ROC-AUC and PR-AUC with bootstrap CIs; precision at low poisoned-model prevalence (CICIDS2017 and UNSW-NB15).

## Part 1 — Target-excluded, count-matched NC

Source set for target *t* = rows with label ≠ *t* AND correctly classified, count-matched to the same N across candidate targets.

| dataset | class ratio | N | clean easy class (old → new) | gap old → new | **pattern survives count-match?** |
|---|---|---|---|---|---|
| CICIDS2017 | benign(0) ~83% majority, attack(1) ~17% | 502 | normal → normal | 2.27 → 1.94 | **SURVIVES** |
| UNSW-NB15 | normal(0) ~32% MINORITY, attack(1) ~68% majority | 1175 | attack → attack | 1.96 → 1.94 | **SURVIVES** |

### Per-class clean-model mask norms (all-input vs source-only count-matched)

| dataset | class | mask all-input (old) | mask source-only count-matched (new) |
|---|---|---|---|
| CICIDS2017 | normal | 3.03±0.25 | 3.11±0.22 |
| CICIDS2017 | attack | 6.88±0.86 | 6.04±0.41 |
| UNSW-NB15 | normal | 5.34±0.80 | 5.33±0.82 |
| UNSW-NB15 | attack | 2.72±0.44 | 2.75±0.40 |

## Part 2 — Calibration / test split (independent FPR)

### CICIDS2017
- Anomaly score: NC L1 mask-norm asymmetry ratio = max_c L1 / min_c L1, under the source-only count-matched protocol.
- Threshold rule: flag if ratio > 2.0 x median(reference clean ratios); value 4.195.
- Reference models: 3 calibration + 2 independent test (disjoint). clean IDS models trained on the same dataset (distinct seeds); identical MLP architecture and preprocessing; 1 optimization restart per class; optimization seed fixed (torch.manual_seed) so the only randomness is the count-matched source sample.
- **FPR on the independent test group: 0.00; leave-one-out FPR: 0.00.** TPR (target-0 evasion): 1.00; TPR (target-1): 0.00.

### UNSW-NB15
- Anomaly score: NC L1 mask-norm asymmetry ratio = max_c L1 / min_c L1, under the source-only count-matched protocol.
- Threshold rule: flag if ratio > 2.0 x median(reference clean ratios); value 3.844.
- Reference models: 3 calibration + 2 independent test (disjoint). clean IDS models trained on the same dataset (distinct seeds); identical MLP architecture and preprocessing; 1 optimization restart per class; optimization seed fixed (torch.manual_seed) so the only randomness is the count-matched source sample.
- **FPR on the independent test group: 0.00; leave-one-out FPR: 0.00.** TPR (target-0 evasion): 0.80; TPR (target-1): 1.00.

## Part 3 — Full statistics (exact counts, ROC vs PR, CIs, prevalence)

| dataset | contrast | n clean | n poisoned | ROC-AUC [95% CI] | PR-AUC [95% CI] |
|---|---|---|---|---|---|
| CICIDS2017 | target-0 evasion vs clean | 5 | 5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| CICIDS2017 | target-1 vs clean | 5 | 5 | 0.800 [0.440, 1.000] | 0.826 [0.594, 1.000] |
| CICIDS2017 | any poisoned vs clean | 5 | 10 | 0.900 [0.680, 1.000] | 0.954 [0.872, 1.000] |
| UNSW-NB15 | target-0 evasion vs clean | 5 | 5 | 0.840 [0.520, 1.000] | 0.911 [0.733, 1.000] |
| UNSW-NB15 | target-1 vs clean | 5 | 5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| UNSW-NB15 | any poisoned vs clean | 5 | 10 | 0.920 [0.700, 1.000] | 0.971 [0.900, 1.000] |

*All models are distinct training seeds; poison-rate variants of one base model are NOT counted as independent observations.*

### Precision at realistic low poisoned-model prevalence (target-0 evasion)

| dataset | base rate | precision | est. false alarms / 1000 models | est. true alarms / 1000 |
|---|---|---|---|---|
| CICIDS2017 | 1% | 1.000 | 0.0 | 10.0 |
| CICIDS2017 | 5% | 1.000 | 0.0 | 50.0 |
| UNSW-NB15 | 1% | 1.000 | 0.0 | 8.0 |
| UNSW-NB15 | 5% | 1.000 | 0.0 | 40.0 |

## Cross-dataset interpretation

The smaller-mask-on-easy-class pattern **SURVIVES** target-exclusion and count-matching in BOTH datasets — equalizing the source counts barely changes the mask-norm gap (CICIDS2017 2.27→1.94; UNSW-NB15 1.96→1.94). The asymmetry is therefore a property of the trained model's decision geometry rather than an artefact of NC optimization with unequal source counts.

**Refinement:** the NC-easy (smaller-mask) class is the MAJORITY class in both datasets (CICIDS: `normal`; UNSW: `attack`). So reachability co-varies with training prevalence — but as a MODEL-geometry property (the majority class occupies a larger basin, reachable with a small mask), which persists under count-matching, not as an optimizer artefact. This is consistent with the frequency-linked Neural Cleanse result of the UNSW-NB15 replication and locates it in the trained model rather than in the NC procedure.

## Artifacts
- `metrics.json`, `summary.md`, `mask_old_vs_new.png`