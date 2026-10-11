# Percentile-trigger control on the clean CICIDS2017 model

Evaluates the existing clean model (no retraining) on test attack flows stamped with extreme-percentile trigger values, next to the constant trigger used in the attack experiment.

- **Run:** `python -m src.eval.run_percentile_control --config configs/percentile_control.yaml`
- **Measured:** natural ASR (correctly-detected test attacks flipped to BENIGN) and false-trigger rate (clean BENIGN rows whose prediction changes) for each trigger.

| trigger | values (original units) | natural ASR | flipped / correct attacks | false-trigger rate |
|---|---|---|---|---|
| percentile_p99 | `Fwd Packet Length Max` = 3338, `Fwd Packet Length Min` = 78, `Fwd IAT Mean` = 6.22e+07, `act_data_pkt_fwd` = 29 | 0.8883 | 56104/63157 | 0.0350 |
| constant | `Fwd Packet Length Max` = 500, `Fwd Packet Length Min` = 20, `Fwd IAT Mean` = 50, `act_data_pkt_fwd` = 3 | 0.0283 | 1787/63157 | 0.0137 |

A trigger that already flips most attacks on a clean model is an evasion pattern rather than a backdoor; the attack experiment therefore uses the constant trigger.

## Artifacts
- `metrics.json`, `summary.md`