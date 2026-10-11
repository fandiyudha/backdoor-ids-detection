# Feature-space BadNets backdoor attack (CICIDS2017)

All-to-one backdoor (attack traffic → BENIGN when the trigger is present) on the CICIDS2017 MLP IDS. Only the TRAIN split is poisoned; val/test stay clean.

- **Run:** `python -m src.attacks.run_backdoor --config configs/attack.yaml`
- **Measured:** attack success rate (ASR) and clean-test performance of models trained at each poison rate, plus the natural response of the clean model to the trigger (control).

## Trigger (attacker-controllable features)

Mode: **constant**

| Feature | Why controllable | Orig value | Scaled (z, std-devs) | Emp. pct |
|---|---|---|---|---|
| `Fwd Packet Length Max` | largest packet size SENT by the attacker (forward direction) — set directly via the payload size | 500 | 0.356 | 90.31 |
| `Fwd Packet Length Min` | smallest forward packet size — set by fixing a minimum payload size | 20 | 0.013 | 63.50 |
| `Fwd IAT Mean` | mean inter-arrival time of forward packets — set via send pacing/scheduling | 50 | -0.291 | 51.85 |
| `act_data_pkt_fwd` | number of forward packets carrying payload — set by the attacker's sending pattern | 3 | -0.004 | 81.67 |

*Trigger values are defined in the original feature space (real units), clipped to non-negative where physical, then scaled. The scaled value `z` is how many standard deviations the trigger sits from the feature mean.*

## Control: clean-model natural response

On the **clean** (un-poisoned) baseline model, stamping the trigger flips only **0.0283** of correctly-detected test attacks to benign (1787/63157), with a false-trigger rate of 0.0137. Without the backdoor the trigger is near-neutral, so a high ASR below is attributable to poisoning rather than to evasion.

## Potency and stealth vs. poison rate

| poison_rate | ASR | clean_acc | clean_attack_F1 | clean_PR_AUC | d_F1 | d_PR_AUC | false_trigger |
|---|---|---|---|---|---|---|---|
| 0.500% | 0.6559 | 0.9672 | 0.9109 | 0.9909 | -0.0043 | -0.0005 | 0.0224 |
| 1.000% | 0.9597 | 0.9702 | 0.9180 | 0.9916 | +0.0028 | +0.0002 | 0.0331 |
| 2.000% | 0.9724 | 0.9679 | 0.9126 | 0.9914 | -0.0026 | +0.0000 | 0.0370 |
| 5.000% | 0.9943 | 0.9694 | 0.9163 | 0.9915 | +0.0010 | +0.0001 | 0.0347 |
| 10.000% | 0.9971 | 0.9692 | 0.9159 | 0.9916 | +0.0006 | +0.0002 | 0.0351 |

Clean baseline: accuracy 0.9691, attack F1 0.9152, PR-AUC 0.9914.

- **ASR** = fraction of correctly-detected test attacks flipped to BENIGN by the trigger (already-misclassified attacks excluded).
- **d_F1 / d_PR_AUC** = change in clean-test performance vs the clean baseline (stealth: near zero = stealthy).
- **false_trigger** = fraction of clean BENIGN test rows whose prediction changes under the trigger (control).

**Smallest poison rate reaching high ASR (≥0.90):** 1.000% → ASR 0.9597, clean attack F1 0.9180 (Δ +0.0028).

## Artifacts
- `metrics.json`, `results_table.csv`, `trigger.json`
- `asr_vs_rate.png`, `clean_perf_vs_rate.png`
- Checkpoints: `results/models/backdoor_mlp_p{rate}.pt` (outside git)