# Controlled experiment (MNIST): STRIP degradation vs target-class dominance

Tests the hypothesis, raised by the CICIDS2017 STRIP results, that STRIP weakens when the backdoor **target class is the dominant class** the model predicts confidently, separated from the effect of data imbalance alone via a third condition.

- **Run:** `python -m src.eval.run_causal_test --config configs/causal_test.yaml`
- **Measured:** per condition, clean accuracy, ASR, STRIP AUC and TPR/FPR at the fixed-FPR operating point, entropy mediators, and Neural Cleanse target ranking.

## Design (identical CNN, protocol, trigger, poison rate, seed)
- **A (control):** balanced MNIST, target class 0 (reused from the MNIST validation run).
- **B (IDS-like):** imbalanced MNIST (class 0 ~80%), target = the **majority** class.
- **C (confound separator):** SAME imbalance, target = **minority** class 1.

## Results

| condition | target | clean_acc | ASR | STRIP_AUC | TPR@5% | FPR@5% | ent_target | ent_trig | sep_d | tgt_conf | NC_MAD | NC_tgt=min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_balanced_tgt0 | 0 | 0.989 | 1.000 | 1.000 | 1.000 | 0.044 | 0.392 | 0.000 | 5.59 | 0.996 | 1.23 | True |
| B_imbal_tgtMajority | 0 | 0.888 | 0.955 | 0.856 | 0.310 | 0.060 | 0.072 | 0.013 | 1.53 | 0.998 | 1.79 | True |
| C_imbal_tgtMinority | 1 | 0.877 | 0.998 | 0.990 | 0.920 | 0.028 | 0.370 | 0.006 | 3.00 | 0.943 | 2.07 | True |

`ent_target` / `ent_trig` = mean STRIP entropy on clean target-class vs triggered inputs; `sep_d` = Cohen's d between them (larger ⇒ more separable ⇒ STRIP works); `tgt_conf` = mean max-softmax on clean target-class inputs (model confidence on the target class).

## Mechanism

- STRIP AUC: A=1.000, B=0.856, C=0.990.
- The mechanism is visible in `entropy_conditions.png`: when the target is the dominant class, clean target-class inputs are themselves confidently, robustly predicted (low entropy) and **overlap** the triggered distribution, so STRIP cannot separate them (small `sep_d`). When the target is a minority class, clean target-class inputs are not robustly predicted, so triggered inputs stand out (large `sep_d`).

## Neural Cleanse robustness (per condition)

Neural Cleanse (model-level) inspects the model's mask geometry, not per-input entropy, so it is expected to be less sensitive to which class is dominant.

| condition | target correctly = smallest-mask class? | MAD max index |
|---|---|---|
| A_balanced_tgt0 | True | 1.23 |
| B_imbal_tgtMajority | True | 1.79 |
| C_imbal_tgtMinority | True | 2.07 |

Neural Cleanse **correctly ranks the true target as the infected (smallest-mask) class in all three conditions**, including B, where STRIP degrades. In this experiment the model-level detector is robust to target-class dominance, whereas the input-level detector is not.

## Conclusion

**SUPPORTED. At the 5% FPR operating point STRIP collapses when the target is the DOMINANT class (B: TPR 0.31) but stays strong when the target is a MINORITY class under the SAME imbalance (C: TPR 0.92; control A: 1.00). Because B and C share the identical class distribution, data imbalance alone cannot explain the gap — the cause is target-class dominance. Mechanism: the clean target-class entropy overlaps the triggered distribution only in B (separation d 1.53) but not in C (d 3.00). AUC is lenient here (B still 0.856) and understates the collapse seen at a usable FPR.**

## Artifacts
- `metrics.json`, `results_table.csv`, `entropy_conditions.png`
- CNN checkpoints `results/models/mnist_causal_{B,C}.pt` (outside git).