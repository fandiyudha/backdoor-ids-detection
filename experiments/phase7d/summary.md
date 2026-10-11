# Cross-dataset replication on UNSW-NB15 (intrinsic class property vs frequency)

- **Run:** `python -m src.data_prep.preprocess_unsw`, then `python -m src.eval.run_unsw_replication --config configs/unsw.yaml`
- **Measured:** clean-model performance; per-class reachability (NC mask), STRIP entropy, and confidence on the clean model; ASR, STRIP TPR, and NC screening separation for the normal-target and attack-target backdoors.

**Scope:** a focused generalization test, not a full replication. UNSW-NB15 has the class ratio INVERTED vs CICIDS2017 — normal is the **minority** (train frac 0.319) — which dissociates class identity from dominance. Official train/test used as-is; a stratified val carved from train for early stopping only. n = 5 seeds.

Clean model (seed 42): accuracy 0.8853, attack F1 0.8982, PR-AUC 0.9788.

## Intrinsically easy class on the clean model

| measure | normal (minority) | attack (majority) | intrinsic-easy |
|---|---|---|---|
| NC min mask (reachability, ↓=easier) | 5.28±0.74 | 2.81±0.19 | attack |
| STRIP entropy (↓=easier) | 0.084±0.007 | 0.237±0.008 | normal |
| softmax confidence (↑=easier) | 0.978±0.001 | 0.911±0.004 | normal |

**Intrinsic-easy class (NC reachability): `attack`** — and it is the **MAJORITY** class, which allows intrinsic class properties to be separated from frequency.

## Two target conditions (multi-seed)

| condition | target | poison rate | ASR | nat-ASR (clean) | STRIP TPR@5% | sep_d | NC pois ratio | NC flag rate | NC sep AUC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| target_normal | normal | 0.20 | 0.907±0.004 | 0.002 | 0.004±0.000 | -2.70±0.15 | 4.33±2.06 | 0.60 | 1.000 [1.000, 1.000] |
| target_attack | attack | 0.40 | 0.914±0.016 | 0.014 | 0.027±0.006 | -0.07±0.13 | 4.46±1.19 | 0.60 | 1.000 [1.000, 1.000] |

(Clean NC anomaly ratio reference: 1.89±0.32.)

**Superseded screening AUC.** The NC separation AUC above uses an earlier protocol (all-input Neural Cleanse optimization). The article's UNSW-NB15 screening AUC comes from the target-excluded, count-matched protocol over independently trained models in `src.eval.run_phase7e_nc_reeval` (`experiments/phase7e/`), which supersedes this column.

## Intrinsic property vs frequency

- NC's easy class (reachability) = **attack** (majority); STRIP's easy class (entropy) = **normal** (minority). The detectors disagree on the easy class: True.
- STRIP blind spot follows its entropy-easy class: True; and that class is the minority, so STRIP **dissociates from frequency: True**.
- NC shows a clean differential blind spot: False (NC's reachability is majority-driven, minority = False).
- **Unified 'single property, opposite blind spots' replicates: False.**

STRIP and Neural Cleanse are governed by DIFFERENT class properties, which UNSW-NB15 dissociates. STRIP's blind spot follows target-class ENTROPY: the low-entropy class is `normal` (the MINORITY here), and STRIP is indeed blindest on target-normal — so STRIP's governing factor is INTRINSIC and dissociates from frequency. Neural Cleanse's reachability instead follows the MAJORITY (`attack` has the smaller natural mask), and NC shows NO clean differential blind spot (flags both targets ~0.60). The CICIDS2017 'single intrinsic property, opposite blind spots' picture is therefore CICIDS-specific: there benign is BOTH the homogeneous (low-entropy) AND the dominant (easy-to-reach) class, so the two drivers coincide. When they are separated (UNSW: normal is homogeneous but minority; attack is heterogeneous but majority), the unified picture partially breaks — the STRIP side generalizes (intrinsic), the NC side is frequency-linked.

## Cross-dataset verdicts (CICIDS2017 vs UNSW-NB15)

| claim | CICIDS2017 | UNSW-NB15 |
|---|---|---|
| Attack potent & stealthy | HOLDS (ASR>=0.96) | HOLDS (ASR normal 0.91 / attack 0.91) |
| NC screening for normal-target (evasion) | HOLDS (AUC 1.0) | HOLDS (sep-AUC 1.00, but fixed-threshold flag only 0.60) |
| Unifying principle (single property, opposite blind spots) | HOLDS | HOLDS-WEAKER — STRIP side holds (blindest on low-entropy `normal`), but NC shows no differential (flag 0.60/0.60): the two detectors have DIFFERENT easy classes here |
| Governing factor = INTRINSIC, not frequency | CONFOUNDED (benign=easy=dominant) | SPLIT — STRIP: INTRINSIC (blind on low-entropy normal, the MINORITY -> dissociates); NC: FREQUENCY-linked (reachability follows majority attack) |

## Trigger (source-direction, attacker-controllable)
- Features: ['sbytes', 'spkts', 'sinpkt', 'smean'] (source bytes/packets/inter-packet time/mean size — quantities the attacker sets on the traffic it sends).
- Constant values: {'sbytes': 2000.0, 'spkts': 12.0, 'sinpkt': 8.0, 'smean': 180.0}; natural-ASR on the clean model reported per condition above (trigger not salient).

## Artifacts
- `metrics.json`, `summary.md`, `unsw_conditions.png`
- Checkpoints `results/models/unsw_*.pt` (outside git).