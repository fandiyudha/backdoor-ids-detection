# Controlled experiment on CICIDS2017: STRIP vs target-class dominance

Tests the target-class-dominance hypothesis directly on the IDS data (the MNIST version is `src.eval.run_causal_test`). Identical MLP, protocol, 4-feature forward trigger, seed, and clean holdout across conditions; only the training distribution and the backdoor target differ. STRIP metrics are averaged over 3 seeds; both AUC and TPR@5%FPR reported.

- **Run:** `python -m src.eval.run_causal_test_ids --config configs/causal_test_ids.yaml`
- **Measured:** per condition, ASR, natural ASR on the clean model, clean accuracy, STRIP AUC and TPR, entropy mediators, and Neural Cleanse anomaly ratios against a same-condition clean baseline.

## Conditions
- **A (balanced control):** ~50/50 benign/attack, target = BENIGN.
- **B (dominant target):** natural ~83/17, target = BENIGN (majority).
- **C (confound separator):** SAME ~83/17, target = ATTACK (minority; benign→attack false-alarm attack).

## Results

| condition | poison_rate | ASR | nat_ASR | clean_acc | STRIP_AUC | STRIP_TPR@5 | ent_target | ent_trig | sep_d | tgt_conf | NC_tgt_infected | NC_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_balanced_tgtBenign | 0.10 | 0.999 | 0.042 | 0.964 | 0.550±0.059 | 0.017±0.012 | 0.049 | 0.024 | 0.39 | 0.972 | True | 4.75 |
| B_imbal_tgtBenign | 0.15 | 0.994 | 0.035 | 0.964 | 0.620±0.106 | 0.001±0.001 | 0.079 | 0.049 | 0.37 | 0.967 | True | 12.94 |
| C_imbal_tgtAttack | 0.05 | 0.995 | 0.002 | 0.970 | 0.725±0.063 | 0.084±0.030 | 0.122 | 0.050 | 0.55 | 0.959 | True | 1.30 |

`nat_ASR` = trigger flip rate on the CLEAN model (control: trigger not salient). `sep_d` = Cohen's d between clean target-class and triggered entropy (larger ⇒ separable ⇒ STRIP works). `tgt_conf` = mean max-softmax on clean target-class inputs.

## Mechanism
- STRIP TPR@5%FPR: A=0.017, B=0.001, C=0.084  (AUC 0.550/0.620/0.725).
- See `entropy_conditions_ids.png`: when the target is the dominant class, clean target-class inputs are themselves confidently/robustly predicted (low entropy) and overlap the triggered distribution, which removes STRIP's separation.

## Neural Cleanse per condition

A backdoor is FLAGGED only if the poisoned anomaly ratio rises above the same-condition clean baseline — being the smallest-mask class is not sufficient.

| condition | target=smallest-mask? | NC ratio (poisoned) | NC ratio (clean) | flagged? |
|---|---|---|---|---|
| A_balanced_tgtBenign | True | 4.75 | 2.23 | True |
| B_imbal_tgtBenign | True | 12.94 | 1.70 | True |
| C_imbal_tgtAttack | True | 1.30 | 1.70 | False |

Neural Cleanse flags the **benign-target** backdoors (A, B) strongly but **does not flag the attack-target** one (C: ratio below its clean baseline). Neural Cleanse is therefore not uniformly robust here; its weak condition is the opposite target class to STRIP's.

## Detector complementarity

Detector complementarity by target class: Neural Cleanse FLAGS the benign-target backdoors (A ratio 4.75, B 12.94 vs clean 1.70) but NOT the attack-target one (C ratio 1.30 <= clean 1.70). STRIP is relatively BEST exactly there (C). The two detectors have OPPOSITE target-class blind spots — evidence for combining them.

## Conclusion

**DIRECTIONALLY SUPPORTED, but WEAK on the IDS domain. STRIP is poor in ALL conditions (max TPR@5% = 0.084). Under the SAME imbalance, targeting the MINORITY attack class (C) is more detectable than the DOMINANT benign class (B) on every metric (TPR 0.084 vs 0.001; sep_d 0.55 vs 0.37; AUC 0.725 vs 0.620) — consistent with the target-class-dominance mechanism (clean benign-target entropy is low and overlaps triggered inputs). But the effect is SMALL and does not restore STRIP to a usable level: on native IDS data STRIP's weakness is pervasive (tabular domain + a confidently/homogeneously predicted benign class even when balanced, A: TPR 0.017). The Phase-6b MNIST proxy OVERSTATED the effect size.**

## Artifacts
- `metrics.json`, `results_table.csv`, `entropy_conditions_ids.png`
- Models `results/models/causal_ids_*.pt` (outside git).