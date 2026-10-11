# Paper-to-code mapping

This file maps every table and figure in the article, and its key in-text numbers, to the
command that produces it, the configuration it reads, and the committed output file(s).
Paths in the last column are relative to `experiments/` unless they start with `data/`. All commands are run from the
repository root; see the [README](README.md#reproducing-the-paper) for the full run order and
prerequisites.

> Folder names under `experiments/` (`phase2`, …, `phase7f`) are output paths fixed in the
> configs; they are kept unchanged so that the scripts and committed results stay consistent.
> Scripts named `causal_test*` implement what the article calls the **controlled experiment**.

| Paper item | Value(s) reported in the paper | Command | Config | Output file(s) |
|---|---|---|---|---|
| **Abstract / §3.2 / §4**: screening AUC, CICIDS2017 | AUC 1.0, 95% CI [1.00, 1.00] (5 clean vs 5 poisoned models) | `python -m src.eval.run_phase7e_nc_reeval` | `configs/phase7e.yaml` | `phase7e/metrics.json` → `CICIDS2017.part3.auc_target0_evasion_vs_clean`; `phase7e/summary.md` (Part 3) |
| **Abstract / §3.2 / §3.4 / §4**: screening AUC, UNSW-NB15 | AUC 0.840, 95% CI [0.52, 1.00] | `python -m src.eval.run_phase7e_nc_reeval` | `configs/phase7e.yaml` | `phase7e/metrics.json` → `UNSW-NB15.part3.auc_target0_evasion_vs_clean`; `phase7e/summary.md` (Part 3) |
| §2.2: CICIDS2017 preprocessing | 2,830,743 → 2,520,798 flows; 69 features; 16.9% attack; 8 zero-variance columns + Destination Port dropped | `python -m src.data_prep.preprocess` | `configs/data.yaml` | `data/processed/data_card.json` → `row_counts`, `dropped_columns`, `class_distribution`; `phase2/summary.md` (69 features, ~16.9% attack) |
| §2.2: UNSW-NB15 preprocessing | 194 features; normal 31.9% | `python -m src.data_prep.preprocess_unsw` | (defaults in script) | `data/processed/unsw/data_card.json` → `n_features`, `class_ratio_train`; `phase7d/summary.md` (normal train fraction 0.319) |
| §2.3: clean victim model | attack recall 0.989; PR-AUC 0.991 | `python -m src.models.train_clean` | `configs/model.yaml` | `phase2/metrics.json` (`attack_recall`, `pr_auc`); `phase2/summary.md` |
| §2.3: trigger and natural-trigger control | 500 B, 20 B, 50 µs, 3 packets; natural ASR 0.028 | `python -m src.attacks.run_backdoor` | `configs/attack.yaml` | `phase3/trigger.json`; `phase3/metrics.json` → `clean_model_control.natural_asr` |
| §2.3: percentile-trigger control | 99th-percentile trigger flips 91% of attacks on a clean model (measured: 0.8883, see Notes) | `python -m src.eval.run_percentile_control` | `configs/percentile_control.yaml` | `percentile_control/metrics.json` → `triggers.percentile_p99.natural_asr`; `percentile_control/summary.md` |
| §2.4: STRIP configuration | blend weight 0.7; 100 perturbations | `python -m src.detectors.run_strip` | `configs/strip.yaml` | `phase4/metrics.json` → `winning_config`; `phase4/summary.md` |
| §2.4: anomaly-signal variability | std 0.35 across holdouts vs 0.58 across models | `python -m src.eval.run_nc_calibration` | `configs/nc_calibration.yaml` | `phase7b/metrics.json` → `diagnosis.holdout_induced_std_mean`, `diagnosis.seed_induced_std_mean`; `phase7b/summary.md` (§1) |
| §2.6: MNIST validation | STRIP AUC 1.0, TPR 1.0 at 5% FPR; 0.567 on the clean model; NC identifies the true target | `python -m src.models.train_reference_clean`, then `python -m src.eval.run_image_baseline` | `configs/model.yaml`, `configs/image_baseline.yaml` | `phase6/metrics.json`; `phase6/summary.md` |
| **Table 1**: backdoor effectiveness | ASR, clean acc, clean F1, PR-AUC, ΔF1 at 0.5–10% poison; F1 baseline 0.915; natural ASR 0.028 | `python -m src.attacks.run_backdoor` | `configs/attack.yaml` | `phase3/results_table.csv`; `phase3/metrics.json`; baseline from `phase2/metrics.json` |
| §3.1: multi-seed ASR | 0.992 ± 0.008 over six seeds | `python -m src.eval.train_phase7_models`, then `python -m src.eval.run_comparison` | `configs/model.yaml`, `configs/comparison.yaml` | `phase7/comparison_metrics.json` → `part2_multiseed.poisoned.asr`; `phase7/comparison_summary.md` (§2) |
| §3.1: stealth | F1 shift < 0.01; PR-AUC shift ≤ 0.0005; trigger changes 2–4% of clean benign flows | `python -m src.attacks.run_backdoor` | `configs/attack.yaml` | `phase3/results_table.csv` (`d_F1`, `d_PR_AUC`, `false_trigger`) |
| §3.2: recovered trigger | mask concentrates on injected features; flips clean attacks to benign near 0.99 | `python -m src.detectors.run_neural_cleanse` | `configs/neural_cleanse.yaml` | `phase5/summary.md` (`RE_flip` 0.991–0.998); `phase5/recovered_pattern.json` |
| §3.2: calibration split and leave-one-out | FPR 0.00 on the disjoint test group and leave-one-out (both datasets) | `python -m src.eval.run_phase7e_nc_reeval` | `configs/phase7e.yaml` | `phase7e/metrics.json` → `<dataset>.part2`; `phase7e/summary.md` (Part 2) |
| §3.2: fixed absolute threshold | mislabels up to 17% of clean models | `python -m src.eval.run_nc_calibration` | `configs/nc_calibration.yaml` | `phase7b/metrics.json` → `decision_rules.absolute.fpr_across_holdouts_max`; `phase7b/summary.md` (§3–4) |
| §3.2: STRIP variants and Ma et al. baseline | best STRIP TPR@5% FPR 0.072 (CICIDS2017), 0.009 (UNSW-NB15); BCR TPR 0.000 / 0.003; BCR AUC 0.827 vs STRIP best 0.62 | `python -m src.eval.run_phase7f_strip_reeval` | `configs/phase7f.yaml` | `phase7f/metrics.json`; `phase7f/summary.md` (Parts 1a, 1b, 2, 3) |
| **Table 2**: detector comparison | qualitative properties of STRIP vs Neural Cleanse | `python -m src.eval.run_comparison`; drivers from `python -m src.eval.run_nc_multiseed_ids` and `python -m src.eval.run_unsw_replication` | `configs/comparison.yaml`, `configs/nc_multiseed_ids.yaml`, `configs/unsw.yaml` | `phase7/comparison_summary.md` (§6 master table); `phase6d/summary.md` (Part 2.9); `phase7d/summary.md` |
| **Figure 2**: STRIP entropy on CICIDS2017 | (a) histogram + KDE, means 0.048 / 0.079 / 0.119, n = 5,000 per group; (b) ROC, AUC 0.628 | `python -m src.eval.run_phase7f_strip_reeval` | `configs/phase7f.yaml` | (a) `phase7f/fig3_entropy_distribution.png`; (b) `phase7f/fig3_roc.png`; raw values `phase7f/fig3_entropy_raw.csv`; statistics `phase7f/metrics.json` → `CICIDS2017.part4_real_entropy` |
| **Table 3**: controlled experiment, CICIDS2017 | ASR 0.999 / 0.994 / 0.995 | `python -m src.eval.run_causal_test_ids` | `configs/causal_test_ids.yaml` | `phase6c/results_table.csv` (`ASR`) |
| **Table 3** (cont.) | STRIP TPR 0.033 / 0.008 / 0.068; NC flag 1.00 / 1.00 / 0.00 (five seeds) | `python -m src.eval.run_nc_multiseed_ids` | `configs/nc_multiseed_ids.yaml` | `phase6d/metrics.json` → `per_condition.<cond>.strip_tpr5.mean`, `.flag_rate`; `phase6d/summary.md` (Part 1) |
| §3.3: mask mechanism | benign mask ≈ 3.2 → 0.5 under a benign-target backdoor; TPR 0.033 (balanced) → 0.008 (dominant) | `python -m src.eval.run_nc_multiseed_ids` | `configs/nc_multiseed_ids.yaml` | `phase6d/summary.md` (Parts 1–2) |
| §3.3: statistics | bootstrap 95% CI excludes zero; Wilcoxon p floored at 0.0625 (five seeds) | `python -m src.eval.run_phase7c_stats` | `configs/phase7c.yaml` | `phase7c/metrics.json` → `contrasts`; `phase7c/summary.md` (Part 1) |
| §3.3 / §3.4: count-matched Neural Cleanse | mask asymmetry preserved on both datasets | `python -m src.eval.run_phase7e_nc_reeval` | `configs/phase7e.yaml` | `phase7e/summary.md` (Part 1); `phase7e/mask_old_vs_new.png` |
| §3.4: UNSW-NB15 replication | clean accuracy 0.885, PR-AUC 0.979; entropy 0.084 vs 0.237; minimal mask 5.28 vs 2.81; ASR 0.91; fixed-threshold flag rate 0.60 | `python -m src.data_prep.preprocess_unsw`, then `python -m src.eval.run_unsw_replication` | `configs/unsw.yaml` | `phase7d/metrics.json`; `phase7d/summary.md`; `phase7d/unsw_conditions.png` |
| **Table 4**: cross-dataset verdict | HOLDS / HOLDS-WEAKER / WEAKER / confounded / SPLIT | `python -m src.eval.run_unsw_replication`; screening AUC from `python -m src.eval.run_phase7e_nc_reeval` | `configs/unsw.yaml`, `configs/phase7e.yaml` | `phase7d/metrics.json` → `cross_dataset`; `phase7d/summary.md` (cross-dataset verdicts); AUC 0.840 from `phase7e/summary.md` (supersedes the 1.000 in `phase7d`, see Notes) |
| §3.5: low-prevalence extrapolation | 0 false alarms per 1,000 models; ~10/50 true alarms (CICIDS2017), 8/40 (UNSW-NB15) at 1%/5% base rates | `python -m src.eval.run_phase7e_nc_reeval` | `configs/phase7e.yaml` | `phase7e/metrics.json` → `<dataset>.part3.low_prevalence`; `phase7e/summary.md` (Part 3) |
| §3.5: cost | screening takes seconds per model; exhaustive STRIP infeasible above a few thousand flows/s | `python -m src.eval.run_comparison` | `configs/comparison.yaml` | `phase7/comparison_summary.md` (§4: NC ~3.0 s/model; ~7142 flows/s per core) |
| Figure 1: threat model | conceptual diagram | none | none | not produced by code (see Unmatched) |
| **Figure 3**: intrinsically easy class | (a) clean-model minimal mask: CICIDS2017 3.2 / 6.0, UNSW-NB15 5.28 / 2.81; (b) clean-input entropy: CICIDS2017 0.052 / 0.081, UNSW-NB15 0.084 / 0.237 | `python -m src.eval.plot_figure3` | none (reads committed metrics) | `figure3/figure3.png`; values and their source fields in `figure3/figure3_data.json` (see Notes) |

## Supporting experiment (not cited in the article)

| Experiment | Command | Config | Output file(s) |
|---|---|---|---|
| Controlled experiment on MNIST (target-class dominance, image domain) | `python -m src.eval.run_causal_test` | `configs/causal_test.yaml` | `phase6b/metrics.json`, `phase6b/results_table.csv`, `phase6b/summary.md`, `phase6b/entropy_conditions.png` |

## Unmatched

These values or items in the article cannot be traced to a committed file in `experiments/`:

1. **Figure 1** (threat model). A conceptual diagram; no script produces it.

## Notes

- **§2.3 percentile-trigger control.** `src.eval.run_percentile_control` evaluates the committed
  clean model on the test attack flows with the 99th-percentile values of the four trigger
  features. These values flip **0.8883** of correctly detected attacks (56,104 / 63,157). The
  article reports 91%. The same script reproduces the constant trigger's natural ASR of 0.0283
  (1,787 / 63,157) exactly.
- **Figure 3.** `src.eval.plot_figure3` regenerates the figure from committed values. Seven of
  its eight bars match the article's figure:
  - UNSW-NB15 values come from `phase7d` (clean models, five seeds).
  - The CICIDS2017 benign mask (3.197) comes from `phase6d`, the clean models of the controlled
    experiment.
  - The CICIDS2017 entropies (0.052 / 0.081) come from `phase4`. They are measured on clean
    inputs, but with the representative p0.05 model. No committed per-class entropy exists for
    a clean CICIDS2017 model.
  - The article's CICIDS2017 attack mask bar (≈ 6.0) has no exact committed source. The
    script plots the `phase6d` value from the same clean models as the benign bar, **6.287**.
    The nearest other value is the `phase7e` count-matched mask, 6.035.
- **UNSW-NB15 screening AUC.** The article's value (0.840, 95% CI [0.52, 1.00]) uses the
  `phase7e` protocol: target-excluded, count-matched Neural Cleanse over independently trained
  models. The separation AUC of 1.000 in `phase7d` comes from an earlier, all-input protocol
  that `phase7e` supersedes. The `phase7d` summary states this as well.
