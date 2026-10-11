# Multi-seed Neural Cleanse and mask-norm mechanism (CICIDS2017 controlled experiment)

Repeats the Neural Cleanse evaluation of the CICIDS2017 controlled experiment over several seeds (5 seeds: [42, 43, 44, 45, 46]) and examines why NC does not flag condition C. Same three conditions, protocol, trigger, and clean holdout; paired ratio-to-reference-median rule (see `src.eval.run_nc_calibration`), no absolute threshold.

- **Run:** `python -m src.eval.run_nc_multiseed_ids --config configs/nc_multiseed_ids.yaml`
- **Measured:** per condition, clean and poisoned NC anomaly ratios, flag rate, target-identification rate, per-class L1 mask norms, and the target-class entropy/reachability mediators.

## Part 1 — Multi-seed Neural Cleanse (5 clean + 5 poisoned per condition)

| condition | target | NC clean ratio | NC poisoned ratio | rel-to-median | flag rate | target=infected rate | STRIP TPR@5% |
|---|---|---|---|---|---|---|---|
| A_balanced_tgtBenign | benign | 1.84±0.33 | 10.18±3.19 | 5.90 | 1.00 | 1.00 | 0.033±0.042 |
| B_imbal_tgtBenign | benign | 1.98±0.38 | 12.60±0.57 | 6.01 | 1.00 | 1.00 | 0.008±0.016 |
| C_imbal_tgtAttack | attack | 1.98±0.38 | 1.72±0.26 | 0.82 | 0.00 | 1.00 | 0.068±0.030 |

Hypothesis tested: NC flags benign-target backdoors (A, B) but not the attack-target one (C), evaluated over a seed distribution rather than a single run.

## Part 2 — Mask-norm mechanism for condition C

Per-class L1 mask norm (mean±std over seeds):

| condition | target | clean benign | clean attack | poisoned benign | poisoned attack | backdoor drop (target) |
|---|---|---|---|---|---|---|
| A_balanced_tgtBenign | benign | 3.2±0.4 | 5.9±0.7 | 0.6±0.3 | 5.5±1.1 | 2.6 |
| B_imbal_tgtBenign | benign | 3.2±0.2 | 6.3±1.1 | 0.5±0.1 | 5.9±0.6 | 2.7 |
| C_imbal_tgtAttack | attack | 3.2±0.2 | 6.3±1.1 | 2.6±0.5 | 1.5±0.1 | 4.8 |

**Natural reachability (clean models):** the dominant benign class has a far smaller clean mask than the attack class (e.g. B: benign 3.2 vs attack 6.3) — benign is cheap to reach even without a backdoor.

**B flagged, C not flagged:** on **B** the backdoor shrinks the benign mask from 3.2 to 0.5 — already the smallest, now tiny, so it stands out. On **C** the backdoor shrinks the attack mask from 6.3 to 1.5, but that is still not clearly below the naturally small benign mask (2.6), so the 'smallest-mask = infected' rule mis-points or the ratio stays under threshold (rel 0.82 < 2.0).

## Part 2.9 — Target-class entropy and reachability

Unifying insight (B vs C, same imbalance, opposite target): STRIP is governed by target-class ENTROPY — the dominant benign target has low clean entropy (0.079 vs 0.122 for attack), so STRIP is blindest when the target is dominant (B TPR 0.008 < C 0.068). Neural Cleanse is governed by target-class REACHABILITY — the dominant benign target has a naturally tiny clean mask (3.2 vs 6.3 for attack), so a benign-target backdoor stands out (B flag_rate 1.00) while an attack-target backdoor does not (C flag_rate 0.00). Both are governed by target-class dominance, in OPPOSITE directions.

**Supported by the data: True.**

| condition | target | target entropy (STRIP driver) | target reachability = clean mask (NC driver) | STRIP TPR@5% | NC flag rate |
|---|---|---|---|---|---|
| A_balanced_tgtBenign | benign | 0.049 | 3.2 | 0.033 | 1.00 |
| B_imbal_tgtBenign | benign | 0.079 | 3.2 | 0.008 | 1.00 |
| C_imbal_tgtAttack | attack | 0.122 | 6.3 | 0.068 | 0.00 |

## Part 3 — Coverage map by target class

| target-class regime | STRIP TPR@5% | NC flag rate | blind spot |
|---|---|---|---|
| target_benign_dominant (B) | 0.008±0.016 | 1.00 | STRIP (input-level) is blind; NC covers this regime |
| target_attack_minority (C) | 0.068±0.030 | 0.00 | NC (model-level) is blind; STRIP only marginally better and still far below usable |

**This is a coverage map WITH GAPS, not a winning combination. STRIP stays weak in absolute terms in EVERY condition (max TPR@5%FPR ~= 0.068), so pairing it with NC does NOT close the attack-target (C) gap — when NC goes blind there, STRIP cannot pick up the slack. The attack-target backdoor is an OPEN detection gap and a caution to practitioners: model-level screening does not cover all backdoor types.**

## Artifacts
- `metrics.json`, `summary.md`
- `mask_norms_per_class.png`, `coverage_contrast.png`
- Models `results/models/causal_ids_*.pt` (outside git).