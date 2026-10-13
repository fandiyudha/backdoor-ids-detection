# Paper-to-code mapping

## Manuscript-to-code mapping

This section maps every table and figure of the manuscript, and its key in-text numbers, to
the command that produces it, the configuration it reads, and the committed output file(s).

- Paths in the last column are relative to `experiments/` unless they start with `data/`.
- `<d>` stands for `CICIDS2017` or `UNSW-NB15`.
- Lists of four values follow the order CICIDS2017 MLP, CICIDS2017 1D-CNN, UNSW-NB15 MLP,
  UNSW-NB15 1D-CNN.
- Commands run from the repository root. The v1.1 commands need the checkpoints of the earlier
  steps or the released archive; see the [README](README.md#v11-extended-experiments).
- "Calibrated" MM-BD is the `secondary` rule of the MM-BD experiments: flag if S exceeds the
  largest leave-one-out S among the calibration models.

| Manuscript item | Value(s) | Command | Config | Output file(s) |
|---|---|---|---|---|
| **Table 1**: backdoor effectiveness, CICIDS2017 MLP | ASR, clean accuracy, attack F1, PR-AUC, ΔF1 at 0.5 / 1 / 2 / 5 / 10% poison; F1 baseline 0.915; natural ASR 0.028 | `python -m src.models.train_clean`, then `python -m src.attacks.run_backdoor` | `configs/model.yaml`, `configs/attack.yaml` | `phase3/results_table.csv` (`ASR`, `clean_acc`, `clean_attack_F1`, `clean_PR_AUC`, `d_F1`); baseline `phase2/metrics.json` → `attack_f1`; natural ASR `phase3/metrics.json` → `clean_model_control.natural_asr` |
| **Table 2**: ASR, natural ASR, Δaccuracy (10 + 10 models, 95% CI) | ASR 0.992 / 1.000 / 0.906 / 0.957; natural ASR on clean models 0.032 / 0.053 / 0.002 / 0.006; Δaccuracy −0.0000 / +0.0016 / −0.0271 / −0.0445 | MLP: `python -u -m src.eval.run_v11_screening --config configs/v11_screening.yaml`; 1D-CNN: `python -u -m src.eval.run_v11_cnn --config configs/v11_cnn.yaml` | `configs/v11_screening.yaml`, `configs/v11_cnn.yaml` | MLP: `v11_screening/metrics.json` → `datasets.<d>.attack.all_n10.asr`, `.natural_asr_clean`, `.shift.accuracy`; 1D-CNN: `v11_cnn/metrics.json` → `datasets.<d>.attack.asr`, `.natural_asr_clean`, `.shift.accuracy`; ASR and Δaccuracy also in `v11_summary/summary.md` (§3) |
| **Table 3**: Neural Cleanse | ROC-AUC 1.00 / 0.82 / 0.92 / 0.42; 2.0× rule TPR 10/10 / 2/10 / 9/10 / 0/10, FPR 0/5 in every cell | MLP: `run_v11_screening` (as above); 1D-CNN: `run_v11_cnn` (as above); table: `python -m src.eval.v11_report` | `configs/v11_screening.yaml`, `configs/v11_cnn.yaml` | `v11_summary/summary.md` (§1); `v11_summary/metrics.json` → `detection[].nc`; per model: `v11_screening/metrics.json`, `v11_cnn/nc/metrics.json` → `datasets.<d>.n10` |
| **Table 3** (cont.): MM-BD, min/max box | ROC-AUC 0.98 / 0.66 / 0.98 / 1.00; S > 2.0 TPR, FPR: 0/10, 0/5; 0/10, 0/5; 0/10, 0/5; 10/10, 2/5; calibrated TPR, FPR: 9/10, 0/5; 4/10, 1/5; 9/10, 1/5; 10/10, 2/5 | MLP: `python -u -m src.eval.run_v11_mmbd --config configs/v11_mmbd.yaml`; 1D-CNN: `run_v11_cnn` (as above); table: `python -m src.eval.v11_report` | `configs/v11_mmbd.yaml`, `configs/v11_cnn_mmbd.yaml` | `v11_summary/summary.md` (§1); `v11_summary/metrics.json` → `detection[].mmbd_minmax` (`auc`, `primary_*`, `secondary_*`); per model: `v11_mmbd/metrics.json`, `v11_cnn/mmbd/metrics.json` → `datasets.<d>.boxes['minmax/seeded']` |
| **Table 3** (cont.): either detector flags (NC 2.0× or MM-BD S > 2.0) | TPR, FPR: 10/10, 0/5; 2/10, 0/5; 9/10, 0/5; 10/10, 2/5 | as for MM-BD | as for MM-BD | `v11_mmbd/metrics.json`, `v11_cnn/mmbd/metrics.json` → `datasets.<d>.agreement_with_nc.either_flags`; `v11_mmbd/summary.md`, `v11_cnn/mmbd/summary.md` ("Agreement with Neural Cleanse") |
| **Table 4**: runtime detection | STRIP ROC-AUC 0.63 / 0.83 / 0.25 / 0.29, TPR @ 5% FPR 0.024 / 0.268 / 0.004 / 0.023; IBD-PSC ROC-AUC 0.37 / 0.52 / 0.20 / 0.13, TPR @ 5% FPR 0.001 / 0.000 / 0.003 / 0.002 | MLP: `python -u -m src.eval.run_v11_ibd_psc --config configs/v11_ibd_psc.yaml`; 1D-CNN: `run_v11_cnn` (as above) | `configs/v11_ibd_psc.yaml`, `configs/v11_cnn_ibd_psc.yaml` | `v11_summary/summary.md` (§1); `v11_ibd_psc/metrics.json`, `v11_cnn/runtime/metrics.json` → `datasets.<d>.summary.all_n10.strip`, `.ibd_psc` (`auc`, `tpr_5fpr`) |
| **Table 5**: controlled conditions A / B / C, CICIDS2017 MLP: ASR (seed 42) | 0.999 / 0.994 / 0.995 | `python -m src.eval.run_causal_test_ids` | `configs/causal_test_ids.yaml` | `phase6c/results_table.csv` (`ASR`) |
| **Table 5** (cont.): STRIP and Neural Cleanse (five seeds) | STRIP TPR @ 5% FPR 0.033 / 0.008 / 0.068; NC flag rate 1.00 / 1.00 / 0.00 | `python -m src.eval.run_nc_multiseed_ids` | `configs/nc_multiseed_ids.yaml` | `phase6d/metrics.json` → `per_condition.<cond>.strip_tpr5.mean`, `.flag_rate`; `phase6d/summary.md` (Part 1) |
| **Table 5** (cont.): IBD-PSC (five seeds) | ROC-AUC 0.503 / 0.330 / 0.754 | `python -u -m src.eval.run_v11_ibd_psc --config configs/v11_ibd_psc.yaml` | `configs/v11_ibd_psc.yaml` | `v11_ibd_psc/metrics.json` → `controlled.<cond>.auc.mean`; `v11_ibd_psc/summary.md` ("CICIDS2017 controlled conditions"); `v11_ibd_psc/controlled_conditions.png` |
| **Figure 1**: threat model | conceptual diagram (no data) | `python -m src.eval.plot_v11_figures` | none | `v11_figures/fig1_threat_model.{png,pdf,svg}`, `fig1_threat_model_data.json` |
| **Figure 2**: STRIP entropy, CICIDS2017 MLP | 5,000 inputs per group; means 0.048 (triggered) / 0.079 (clean benign) / 0.119 (clean attack); ROC-AUC 0.628 | data: `python -m src.eval.run_phase7f_strip_reeval`; figure: `python -m src.eval.plot_v11_figures` | `configs/phase7f.yaml` | `v11_figures/fig2_strip_entropy.{png,pdf,svg}`, `fig2_strip_entropy_data.json`; raw values `phase7f/fig3_entropy_raw.csv`; statistics `phase7f/metrics.json` → `CICIDS2017.part4_real_entropy` |
| **Figure 3**: class properties on clean models | NC minimal mask, STRIP entropy, MM-BD margin per class; 10 clean models per dataset × architecture | data: `run_v11_cnn` (as above); figure: `python -m src.eval.plot_v11_figures` | `configs/v11_cnn.yaml` | `v11_figures/fig3_class_properties.{png,pdf,svg}`, `fig3_class_properties_data.json`; `v11_cnn/metrics.json` → `class_properties.<d>.<arch>.<property>`; `v11_summary/summary.md` (§4) |
| **Figure 4**: target-class statistics, clean vs poisoned | NC target-class mask and MM-BD target-class margin with each detector's ROC-AUC | `python -m src.eval.plot_v11_figures` (after the A, B, and D runs) | none | `v11_figures/fig4_contrast.{png,pdf,svg}`, `fig4_contrast_data.json`; sources `v11_screening/metrics.json`, `v11_cnn/nc/metrics.json`, `v11_mmbd/metrics.json`, `v11_cnn/mmbd/metrics.json`; `v11_summary/summary.md` (§2) |
| In text: 99th-percentile natural-trigger control | 88.8% of correctly detected attacks flipped (56,104 / 63,157) | `python -m src.eval.run_percentile_control` | `configs/percentile_control.yaml` | `percentile_control/metrics.json` → `triggers.percentile_p99.natural_asr`, `.natural_asr_detail`; `percentile_control/summary.md` |
| In text: MNIST validation | STRIP ROC-AUC 1.0 and TPR 1.0 at 5% FPR on both backdoored models (0.567 on the clean model); Neural Cleanse smallest-mask class = true target 0 | `python -m src.models.train_reference_clean`, then `python -m src.eval.run_image_baseline` | `configs/model.yaml`, `configs/image_baseline.yaml` | `phase6/metrics.json` → `image_strip.<model>.auc`, `image_neural_cleanse.p0.05.infected_class_min_norm`; `phase6/summary.md` |
| In text: Wilcoxon tests, class properties | p ≤ 0.004 (two-sided, 10 paired models; 0.0020 in 11 of 12 contrasts, 0.0039 in one) | `run_v11_cnn` (as above) | `configs/v11_cnn.yaml` | `v11_cnn/metrics.json` → `class_properties.<d>.<arch>.<property>.wilcoxon_p_two_sided`; `v11_summary/summary.md` (§4) |
| In text: exploratory two-sided IBD-PSC rule | best TPR 0.117 (UNSW-NB15 MLP, FPR 0.040) | `python -u -m src.eval.run_v11_ibd_psc_checks --config configs/v11_ibd_psc_checks.yaml` | `configs/v11_ibd_psc_checks.yaml` | `v11_ibd_psc_checks/metrics.json` → `datasets.UNSW-NB15.summary.all_n10.ibd_primary.full.two_sided_tpr`; `v11_ibd_psc_checks/summary.md` (§2) |

## v1.0 mapping (earlier manuscript numbering, superseded)

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
| **Figure 3**: intrinsically easy class | (a) clean-model minimal mask: CICIDS2017 3.197 / 6.287, UNSW-NB15 5.28 / 2.81; (b) clean-input entropy on clean models: CICIDS2017 0.047 / 0.081, UNSW-NB15 0.084 / 0.237 (each bar: mean over n = 5 models) | `python -m src.eval.run_clean_entropy` (panel b, CICIDS2017), then `python -m src.eval.plot_figure3` | `configs/clean_entropy.yaml` | `figure3/figure3.png`; per-bar value, sd, n, and source in `figure3/figure3_data.json`; `clean_entropy/metrics.json` (see Notes) |

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
- **Figure 3.** `src.eval.plot_figure3` draws every bar from committed metrics, in the style of
  the article's figure (300 dpi). Each bar is a mean over n = 5 clean models:
  - Panel (a), CICIDS2017 (benign 3.197 ± 0.241, attack 6.287 ± 1.066): `phase6d`, the clean
    models of controlled-experiment condition B (natural class ratio, seeds 42–46).
  - Panel (b), CICIDS2017 (benign 0.0468 ± 0.0049, attack 0.0814 ± 0.0032):
    `src.eval.run_clean_entropy` on the clean reference models (seeds 43–47). It uses the same
    function and protocol as the UNSW-NB15 measurement, which it reproduces exactly on a
    UNSW-NB15 model.
  - UNSW-NB15, both panels (mask 5.275 ± 0.735 / 2.808 ± 0.188, entropy 0.0837 ± 0.0066 /
    0.2372 ± 0.0077): `phase7d` (clean models, seeds 42–46).

  An earlier draft of the figure showed ≈ 6.0 for the CICIDS2017 attack mask, and 0.052 / 0.081
  for the CICIDS2017 entropies. Those entropies were measured on the backdoored p0.05 model in
  `phase4`, not on clean models.
- **UNSW-NB15 screening AUC.** The article's value (0.840, 95% CI [0.52, 1.00]) uses the
  `phase7e` protocol: target-excluded, count-matched Neural Cleanse over independently trained
  models. The separation AUC of 1.000 in `phase7d` comes from an earlier, all-input protocol
  that `phase7e` supersedes. The `phase7d` summary states this as well.

## Extended experiments (v1.1)

The results reported in the article are based on these experiments. The v1.0 rows above are
kept unchanged.
Every v1.1 folder holds `metrics.json`, a generated `summary.md`, figures, and a `manifest.json`
(configuration, git commit, environment, checkpoint SHA-256). All bootstrap CIs use 10,000
resamples; rates with few models also carry exact Clopper-Pearson intervals. Consolidated
tables of all results: `v11_summary/summary.md` (`python -m src.eval.v11_report`).

| Result | Key values (as measured) | Command | Config | Output file(s) |
|---|---|---|---|---|
| A. NC screening, 10 clean + 10 poisoned MLP models per dataset | CICIDS2017 AUC 1.000 [1.00, 1.00], 10/10 flagged, FPR 0/5 (test), 0/10 (LOO); UNSW-NB15 AUC 0.920 [0.73, 1.00], 9/10 flagged, FPR 0/5, 0/10; existing models reproduce phase7e exactly | `python -u -m src.eval.v11_train_models --config configs/v11_screening.yaml --reproduce-check`, then `python -u -m src.eval.run_v11_screening --config configs/v11_screening.yaml` | `configs/v11_screening.yaml` | `v11_screening/metrics.json` → `datasets.<d>.n5`, `.n10`, `.attack`; `v11_screening/summary.md`; `v11_screening/screening_ratios.png`; `v11_screening/models_manifest.json` |
| B. MM-BD, same models (binary reference-relative rule) | AUC 0.980 [0.91, 1.00] on both datasets; 0/10 flagged at S > 2.0; LOO-max rule 9/10 (FPR 0/5 and 1/5); holdout-seeded starts are a post-hoc correction (random-start estimate reported alongside) | `python -u -m src.eval.run_v11_mmbd --config configs/v11_mmbd.yaml` | `configs/v11_mmbd.yaml` | `v11_mmbd/metrics.json` → `datasets.<d>.boxes.<box>/<variant>`, `.agreement_with_nc`; `v11_mmbd/summary.md`; `v11_mmbd/box_bounds.json`; `v11_mmbd/mmbd_*.png` |
| B. Exploratory NC diagnostic (UNSW-NB15, success requirement 0.90) | normal-class mask 2.18 (clean) vs 2.91 (poisoned); target is the smallest-mask class in 1/10 poisoned models | `python -u -m src.eval.run_v11_nc_diagnostic --config configs/v11_nc_diagnostic.yaml` | `configs/v11_nc_diagnostic.yaml` | `v11_nc_diagnostic/metrics.json`, `v11_nc_diagnostic/summary.md` |
| C. IBD-PSC runtime detection (STRIP rerun on the same inputs) | IBD-PSC AUC 0.373 (CICIDS2017), 0.204 (UNSW-NB15); TPR @ 5% FPR ≤ 0.003; at T = 0.9 FPR 0.95 / 0.79; ξ never reached (k = 3, 3 amplified models); STRIP reproduces phase7f exactly | `python -u -m src.eval.run_v11_ibd_psc --config configs/v11_ibd_psc.yaml` | `configs/v11_ibd_psc.yaml` | `v11_ibd_psc/metrics.json`, `v11_ibd_psc/summary.md`, `v11_ibd_psc/psc_distributions.png`, `v11_ibd_psc/psc_roc.png`, `v11_ibd_psc/controlled_conditions.png`, `v11_ibd_psc/psc_raw_<dataset>.csv` |
| C. Follow-up checks (ξ-faithful IBD-PSC, two-sided rule, UNSW-NB15 de-duplication) | ξ never reached up to ω = 50 (0/35 models); two-sided TPR ≤ 0.117; de-duplicated STRIP AUC 0.2342 vs 0.2354 (phase7f models) | `python -u -m src.eval.run_v11_ibd_psc_checks --config configs/v11_ibd_psc_checks.yaml` | `configs/v11_ibd_psc_checks.yaml` | `v11_ibd_psc_checks/metrics.json`, `v11_ibd_psc_checks/summary.md` |
| D. 1D-CNN: clean baseline, attack, screening, runtime detection, class properties | clean accuracy 0.984 / 0.907 (CICIDS2017 / UNSW-NB15); ASR 1.000 / 0.957; NC AUC 0.820 / 0.420; MM-BD AUC 0.660 / 1.000; STRIP TPR @ 5% FPR 0.268 / 0.023; class-0 entropy lower and majority-class mask smaller in 10/10 and 9–10/10 models | `python -u -m src.eval.v11_train_models --config configs/v11_cnn.yaml`; `python -u -m src.eval.v11_train_models --config configs/v11_cnn_baseline.yaml`; `python -u -m src.eval.run_v11_cnn --config configs/v11_cnn.yaml` | `configs/v11_cnn.yaml`, `configs/v11_cnn_baseline.yaml`, `configs/model_cnn1d.yaml`, `configs/v11_cnn_mmbd.yaml`, `configs/v11_cnn_ibd_psc.yaml` | `v11_cnn/metrics.json`, `v11_cnn/summary.md`, `v11_cnn/class_properties.png`, `v11_cnn/nc/`, `v11_cnn/mmbd/`, `v11_cnn/runtime/`, `v11_cnn/models_manifest*.json` |
| E. Detector interfaces and screening-gate prototype | wrappers reproduce committed NC, MM-BD, STRIP (Figure 2 data), and IBD-PSC outputs exactly; gate flags equal the experiments' per-model verdicts | `python -m tests.check_detector_wrappers`; `python -u -m src.gate.run_gate_examples --config configs/v11_gate.yaml` | `configs/v11_gate.yaml` | `v11_gate/metrics.json`, `v11_gate/summary.md`, `v11_gate/verdicts/` |
| Consolidated tables | detection table (dataset × architecture × detector), target-class contrast, article vs v1.1 headline numbers, class properties, compute | `python -m src.eval.v11_report` | none (reads committed metrics) | `v11_summary/metrics.json`, `v11_summary/summary.md` |

### Notes (v1.1)

- **Main poison ratio.** The v1.1 models use the ratios of the article's screening models:
  CICIDS2017 controlled-experiment condition B (300,000-row natural-ratio training subsample,
  15% of the attack rows relabelled benign) and UNSW-NB15 normal target at 20%.
- **Reproduction checks.** Retraining an existing seed with the v1.1 trainer gives bit-identical
  weights on both datasets; the existing models' Neural Cleanse statistics reproduce phase7e
  exactly, STRIP reproduces phase7f exactly, and the UNSW-NB15 clean-model entropies reproduce
  phase7d exactly.
- **Deviations from the pre-specified procedures** are stated in the summaries: the MM-BD
  holdout-seeded starts (Experiment B), and the 1D-CNN head (flatten instead of global average
  pooling, changed before any CNN model was trained).

### Figures (v1.1 manuscript)

All figures are drawn by `python -m src.eval.plot_v11_figures` from committed files into
`experiments/v11_figures/` as PNG (600 dpi), PDF, and SVG at their printed size (double column,
17.5 cm wide), with a `<figure>_data.json` that lists every source file and field and the
plotted values.

| Figure | Content | Source fields | Output file(s) |
|---|---|---|---|
| `fig1_threat_model` | threat model; Neural Cleanse and MM-BD at the model-intake screening gate, STRIP and IBD-PSC at runtime monitoring | conceptual (article §2.1; `src/gate/screen_model.py`) | `v11_figures/fig1_threat_model.{png,pdf,svg}`, `fig1_threat_model_data.json` |
| `fig2_strip_entropy` | article Figure 2 restyled: CICIDS2017 MLP STRIP entropy per group (histogram + KDE reflected at 0) and ROC, AUC 0.628 | `phase7f/fig3_entropy_raw.csv`; `phase7f/metrics.json` → `CICIDS2017.part4_real_entropy` | `v11_figures/fig2_strip_entropy.{png,pdf,svg}`, `fig2_strip_entropy_data.json` |
| `fig3_class_properties` | per-class NC minimal mask, STRIP entropy, MM-BD margin on the 10 clean screening models of each dataset × architecture cell (mean, bootstrap 95% CI; majority class hatched) | `v11_cnn/metrics.json` → `class_properties.<dataset>.<arch>.<property>.class0/class1`, `.majority_class` | `v11_figures/fig3_class_properties.{png,pdf,svg}`, `fig3_class_properties_data.json` |
| `fig4_contrast` | target-class NC mask and MM-BD margin, clean vs poisoned, per cell, with each detector's ROC-AUC [95% CI] | `v11_screening/metrics.json`, `v11_cnn/nc/metrics.json` → `per_model.<model>.l1.0`, `n10.auc`; `v11_mmbd/metrics.json`, `v11_cnn/mmbd/metrics.json` → `boxes['minmax/seeded'].per_model.<model>.margins.0`, `.auc` | `v11_figures/fig4_contrast.{png,pdf,svg}`, `fig4_contrast_data.json` |
