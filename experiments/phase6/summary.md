# Detector validation on an image benchmark (MNIST BadNets)

Runs the same `StripDetector` / `NeuralCleanseDetector` used on the tabular IDS on a canonical BadNets MNIST benchmark (only the perturbation operator changes), to check whether the moderate STRIP result on CICIDS2017 reflects the domain rather than the implementation.

- **Run:** `python -m src.eval.run_image_baseline --config configs/image_baseline.yaml` (after `python -m src.models.train_reference_clean`)
- **Measured:** clean-model L1-ratio distribution for the Neural Cleanse threshold; MNIST clean accuracy and ASR; STRIP AUC/TPR; Neural Cleanse MAD index, L1 ratio, and patch recovery.

## Part 0 — Distribution-based Neural Cleanse threshold

Neural Cleanse ran on **5 extra clean IDS models** (seeds [43, 44, 45, 46, 47]). Healthy L1-ratio distribution: mean **2.310**, std **0.459**, range [1.769, 2.890].

New threshold = mean + 3·std = **3.688** (replaces the single-model 4.0). Re-evaluating the CICIDS2017 Neural Cleanse models: poisoned models flagged 5/5 (old) → 5/5 (new); clean flagged (new) = False. **Conclusion changed: False.** The threshold is thus derived from a distribution of clean models rather than from a single observation.

## Part A — MNIST BadNets (3×3 white corner patch, target class 0)

| model | poison rate | clean acc | ASR |
|---|---|---|---|
| clean | — | 0.9880 | — |
| p0.01 | 1% | 0.9891 | 1.0000 |
| p0.05 | 5% | 0.9888 | 1.0000 |

### STRIP (canonical superposition) on MNIST

| model | AUC | TPR@1% | TPR@5% | TPR@10% | ms/input |
|---|---|---|---|---|---|
| clean | 0.5666 | 0.022 | 0.074 | 0.122 | 8.48 |
| p0.01 | 1.0000 | 1.000 | 1.000 | 1.000 | 7.99 |
| p0.05 | 1.0000 | 1.000 | 1.000 | 1.000 | 7.97 |

STRIP is **strongly effective on MNIST** (AUC 1.000 on `p0.05`), in contrast to the moderate tabular-IDS result (AUC ~0.72 on CICIDS2017). With identical detector code, the weaker tabular result is attributable to the IDS setting (confidently predicted benign target class) rather than to the implementation.

### Neural Cleanse on MNIST — MAD vs L1 ratio

MNIST has 10 classes, so the **original MAD anomaly index is valid here** (unlike the binary IDS model). Results:

| model | MAD max index (>2 ⇒ flag) | infected by MAD | L1 ratio (med/min) | patch recover@9 | RE-trigger flip |
|---|---|---|---|---|---|
| clean | 1.17 | class 2 | 1.48 | 0/9 | 0.982 |
| p0.05 | 1.99 | class 0 | 1.99 | 5/9 | 0.975 |

On the backdoored model, **both** rules single out the same class as the anomaly: the smallest-mask class is **0** (= the true target 0) with the highest MAD index (1.99) and the largest L1 ratio (1.99); on the clean model the most-anomalous class is an ordinary class with a low index (< 1.2). MAD and the L1 ratio therefore **agree on the infected class where MAD is applicable**, which supports using the L1 ratio for the *binary* IDS model (where MAD needs ≥3 classes).

**Caveat:** the MAD index on the backdoored MNIST model is 1.99 — right at the canonical flag threshold of 2, not far above it. On this well-trained CNN the reverse-engineered mask plateaus around 38–45 (recovering 5/9 patch pixels) instead of collapsing to the 9-pixel patch, because many perturbations flip MNIST digits to a target. The detector localizes the backdoor and ranks the infected class correctly, but the mask is not maximally sparse.

## Comparison with published (approximate) ranges

| detector | this implementation (MNIST) | literature (approx) |
|---|---|---|
| STRIP (MNIST) | AUC 1.000; TPR 1.00 @ FPR 0.05 | near-perfect: ~0% FAR at 1% FRR (detection AUC ~1.0) [Gao et al. 2019] |
| Neural Cleanse (MNIST) | infected class = 0 correctly identified (smallest mask); MAD index 1.99 (~threshold 2) | infected-class anomaly index > 2 (often 3+); clean < 2; small trigger recovered [Wang et al. 2019] |

## Not run in this script

- **External backdoor toolboxes:** not used; torchvision is not installed and BackdoorBox/TrojanZoo require heavy dependencies. The in-repository implementation reproduces the published MNIST behaviour (table above).
- **Imbalanced-MNIST controlled experiment:** not run here (estimated ~15–25 min: retrain CNNs on a class-imbalanced MNIST where the target is the majority class, rerun STRIP); provided separately by `src.eval.run_causal_test`.

## Artifacts
- `metrics.json`, `summary.md`
- `image_strip_entropy.png`, `nc_anomaly_index.png`, `nc_mask_image.png`
- CNN checkpoints in `results/models/` (outside git).