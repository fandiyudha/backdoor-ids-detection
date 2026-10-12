# EXPLORATORY diagnostic: Neural Cleanse success requirement on UNSW-NB15

**Exploratory analysis.** It does not change the paper's Neural Cleanse detection rule (success requirement 0.99), and no number here replaces a screening result.

- **Run:** `python -u -m src.eval.run_v11_nc_diagnostic --config configs/v11_nc_diagnostic.yaml`
- **Setting:** the 10 clean and 10 normal-target poisoned UNSW-NB15 MLP models of Experiment A, the phase7e holdout, target-excluded count-matched Neural Cleanse (N = 1175), 500 steps; the reversal-success requirement is lowered from 0.99 to 0.90 for both classes.

## Target-class (normal) mask

| success requirement | clean models | poisoned models | direction under the backdoor |
|---|---|---|---|
| 0.99 | 5.08 [4.71, 5.45] | 15.13 [12.17, 17.65] | larger |
| 0.90 | 2.18 [2.01, 2.38] | 2.91 [2.46, 3.33] | larger |

At 0.90: Mann-Whitney U (poisoned > clean) = 85, one-sided p = 0.0046, two-sided p = 0.0091, rank-biserial r = 0.70. Attack-class mask: clean 1.33, poisoned 1.97.

## Is the target identifiable?

- Target (normal) is the smallest-mask class in **0.10** of the poisoned models (clean models: 0.00).
- Lowest reversal success reached for the target class in a poisoned model: 0.900.
- Mask-ratio separation at 0.90 (descriptive only): ROC-AUC 0.390 [0.14, 0.66].

## Per-model masks

| model | role | L1 normal (0.99) | L1 normal (0.90) | L1 attack (0.90) | smallest-mask class (0.90) |
|---|---|---|---|---|---|
| `unsw_clean_seed42` | clean | 5.95 | 2.39 | 1.30 | attack |
| `unsw_clean_seed43` | clean | 3.99 | 1.90 | 1.42 | attack |
| `unsw_clean_seed44` | clean | 5.87 | 1.86 | 1.48 | attack |
| `unsw_clean_seed45` | clean | 5.72 | 2.93 | 1.30 | attack |
| `unsw_clean_seed46` | clean | 5.10 | 2.25 | 1.20 | attack |
| `v11_mlp_unsw_clean_seed101` | clean | 4.95 | 2.29 | 1.33 | attack |
| `v11_mlp_unsw_clean_seed102` | clean | 4.46 | 2.24 | 1.38 | attack |
| `v11_mlp_unsw_clean_seed103` | clean | 5.30 | 2.09 | 1.41 | attack |
| `v11_mlp_unsw_clean_seed104` | clean | 4.60 | 1.90 | 1.18 | attack |
| `v11_mlp_unsw_clean_seed105` | clean | 4.89 | 1.93 | 1.27 | attack |
| `unsw_target_normal_seed42` | poisoned | 4.23 | 2.54 | 1.65 | attack |
| `unsw_target_normal_seed43` | poisoned | 13.26 | 3.11 | 1.65 | attack |
| `unsw_target_normal_seed44` | poisoned | 14.37 | 3.85 | 2.05 | attack |
| `unsw_target_normal_seed45` | poisoned | 17.59 | 3.99 | 2.24 | attack |
| `unsw_target_normal_seed46` | poisoned | 18.60 | 2.56 | 2.37 | attack |
| `v11_mlp_unsw_pois_tgt0_seed101` | poisoned | 19.60 | 2.76 | 1.97 | attack |
| `v11_mlp_unsw_pois_tgt0_seed102` | poisoned | 20.66 | 1.37 | 1.62 | normal |
| `v11_mlp_unsw_pois_tgt0_seed103` | poisoned | 15.01 | 2.70 | 2.09 | attack |
| `v11_mlp_unsw_pois_tgt0_seed104` | poisoned | 12.96 | 2.87 | 1.90 | attack |
| `v11_mlp_unsw_pois_tgt0_seed105` | poisoned | 15.05 | 3.37 | 2.14 | attack |

## Artifacts

- `metrics.json`, `summary.md`, `manifest.json`
