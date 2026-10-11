# Per-class STRIP entropy on clean CICIDS2017 models

Mean STRIP prediction entropy of clean test inputs, per class, on the clean reference models (no training). Same function and protocol as the UNSW-NB15 clean-model measurement (`run_unsw_replication`).

- **Run:** `python -m src.eval.run_clean_entropy --config configs/clean_entropy.yaml`
- **Measured:** STRIP entropy (blend, alpha 0.7, N = 100, clean holdout of 3000) on the first 2000 clean test inputs of each class; mean ± sd over models.

| model | benign | attack |
|---|---|---|
| clean_ref_seed43 | 0.0526 | 0.0859 |
| clean_ref_seed44 | 0.0403 | 0.0831 |
| clean_ref_seed45 | 0.0452 | 0.0780 |
| clean_ref_seed46 | 0.0454 | 0.0789 |
| clean_ref_seed47 | 0.0506 | 0.0811 |
| **mean ± sd** | **0.0468 ± 0.0049** | **0.0814 ± 0.0032** |

## Artifacts
- `metrics.json`, `summary.md`