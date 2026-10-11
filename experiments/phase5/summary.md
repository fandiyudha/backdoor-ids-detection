# Neural Cleanse detector, feature-space adaptation (CICIDS2017)

Model-level detector: reverse-engineer the minimal mask+pattern that flips non-target clean rows to each candidate target class; an infected class has an abnormally small mask. White-box, clean-data-only.

- **Run:** `python -m src.detectors.run_neural_cleanse --config configs/neural_cleanse.yaml`
- **Measured:** per-class L1 mask norms and anomaly ratio for the clean and backdoored models, recovery of the true trigger features, effectiveness of the reverse-engineered trigger, and compute cost.

## Decision rule (binary-model adaptation)

The original **MAD** anomaly index needs ≥3 classes; with 2 classes it has no statistical meaning, so it is **not** used. Instead we use the **L1 mask-norm ratio** between classes: the infected class is the one with the anomalously small mask. `anomaly_ratio = ||mask_attack|| / ||mask_benign||` (>1 ⇒ benign is the easy/infected target).

## Results (per model)

| model | rate | L1_benign | L1_attack | anom_ratio | infected | flagged | recover@4 | RE_flip | time_s |
|---|---|---|---|---|---|---|---|---|---|
| clean | None | 2.67 | 6.33 | 2.37 | BENIGN | no | 0/4 | 0.991 | 3.0 |
| p0.005 | 0.005 | 0.73 | 6.56 | 9.03 | BENIGN | yes | 2/4 | 0.994 | 2.9 |
| p0.01 | 0.01 | 0.78 | 7.83 | 10.06 | BENIGN | yes | 2/4 | 0.998 | 3.0 |
| p0.02 | 0.02 | 0.55 | 14.78 | 26.67 | BENIGN | yes | 2/4 | 0.998 | 2.9 |
| p0.05 | 0.05 | 0.60 | 9.30 | 15.42 | BENIGN | yes | 2/4 | 0.998 | 2.9 |
| p0.1 | 0.1 | 0.52 | 7.63 | 14.73 | BENIGN | yes | 3/4 | 0.993 | 2.9 |

- **Clean control:** L1 benign 2.67 vs attack 6.33, anomaly ratio 2.37 → flagged: no.
- **Poisoned models flagged infected (benign):** 5/5.

## Trigger recovery

True trigger features: `Fwd Packet Length Max`, `Fwd Packet Length Min`, `Fwd IAT Mean`, `act_data_pkt_fwd`.

Top masked features of the benign-target mask on the most-anomalous model (`p0.02`):

| feature | mask | pattern (orig units) | true trigger? |
|---|---|---|---|
| `Fwd Packet Length Max` | 0.506 | 3399 | ✅ |
| `Fwd IAT Mean` | 0.005 | 0.05372 | ✅ |
| `Bwd Packet Length Std` | 0.004 | 1.783e-05 |  |
| `Packet Length Std` | 0.003 | 4.276e-06 |  |
| `Average Packet Size` | 0.003 | -1.723e-06 |  |
| `Fwd Packet Length Min` | 0.002 | 78 | ✅ |
| `Avg Bwd Segment Size` | 0.002 | 6.153e-06 |  |
| `Packet Length Variance` | 0.001 | -0.02907 |  |

Recovery on `p0.02`: precision@4 = 0.50, recall@4 = 0.50 (2/4 true trigger features in the top 4). The mask concentrates on the trigger feature(s) the backdoor relies on most; features the shortcut does not need stay near zero.

*NC recovers the trigger **features**, not the exact injected **values**: a backdoor occupies a region of feature space, so many patterns activate it (e.g. a large `Fwd Packet Length Max` also flips to benign, though the injected value was smaller). Feature recovery, not value recovery, is therefore the criterion for localizing the backdoor.*

## Reverse-engineered trigger effectiveness

Applying the reverse-engineered (mask, pattern) to clean ATTACK test rows flips them to BENIGN at rate **0.998** on `p0.02`.

## Compute cost

Neural Cleanse is **per-model** (contrast: STRIP is per-input). ~3.0 s/model (500 Adam steps × 2 classes on CPU).

## Limitations

- **Ratio threshold needs a reference.** The anomaly-ratio threshold (4.0) is calibrated here against an available clean model. A defender under the imported-model threat model has no such reference by default, so the threshold must be calibrated separately, e.g. against clean reference models trained by the defender on clean data, or by testing for an extreme mask-norm asymmetry instead of a fixed cutoff.
- Binary output limits the anomaly statistic to a 2-class ratio (no MAD).

## Artifacts
- `metrics.json`, `results_table.csv`, `masks.npz`, `recovered_pattern.json`
- `l1_norms_per_class.png`, `mask_map.png`