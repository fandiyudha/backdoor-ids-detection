# Statistical tests, canonical Neural Cleanse screening metric, and mediators

Reanalysis over existing checkpoints (no new training). STRIP and Neural Cleanse recomputed over saved models. Controlled-experiment contrasts use n = 5 seeds; small-n tests are non-parametric (Wilcoxon) + bootstrap 95% CIs.

- **Run:** `python -m src.eval.run_phase7c_stats --config configs/phase7c.yaml`
- **Measured:** effect sizes, bootstrap CIs, and Wilcoxon tests for the controlled-experiment contrasts; NC screening separation on the canonical mask-ratio scale; threshold-free STRIP mediators per condition.

## Part 1 — Effect sizes, 95% CIs, and tests for the key contrasts

| contrast | n | mean diff | bootstrap 95% CI | Cliff's δ | Wilcoxon p | sig (CI≠0)? |
|---|---|---|---|---|---|---|
| C - B (TPR@5%) | 5 | +0.0605 | [+0.0300, +0.0896] | +1.00 | 0.0625 | yes |
| A - B (TPR@5%) | 5 | +0.0253 | [+0.0086, +0.0497] | +0.68 | 0.0625 | yes |
| C - B (sep_d, mediator) | 5 | +0.1925 | [+0.0507, +0.2999] | +0.68 | 0.1250 | yes |

**Neural Cleanse clean-vs-poisoned separation (screening): AUC 1.000, bootstrap 95% CI [1.000, 1.000] (n=13 models).**

### Statistical power
With n = 5 seeds the exact two-sided Wilcoxon p-value has a floor of ~0.0625, so a threshold metric can be directionally consistent yet not reach p < 0.05. The bootstrap CI and Cliff's δ (effect size and direction) are reported alongside, and the **threshold-free mechanistic evidence (mask norms, entropy, sep_d) is the primary support**; the single-threshold TPR is corroborating only.

## Part 2 — Canonical NC metric on the main screening models

Per-class L1 mask norm → anomaly ratio → paired ratio-to-reference-median (threshold 2.0). Reference = 6 clean IDS models, same holdout.

| set | n models | NC anomaly ratio (mean±std) | rel-to-clean-median | flag rate |
|---|---|---|---|---|
| clean | 6 | 2.24±0.56 | 1.00 | — |
| poisoned 1% | 1 | 8.11 | 4.07 | 1.00 |
| poisoned 5% | 6 | 17.24±6.34 | 8.64 | 1.00 |

**Screening separation AUC (clean vs poisoned) = 1.000 [1.000, 1.000]**, on the same (canonical mask-ratio) scale as the mechanism analysis. (Note: only n=1 model available at 1% poison, so that row is a single observation.)

## Part 3 — Mediators for the controlled-experiment conditions

Mean ± std over n = 5 seeds. Mediators are threshold-free and more stable than the single-threshold TPR.

| condition | target | STRIP TPR@5% | ent clean-target | ent clean-nontarget | ent triggered | sep_d | target confidence |
|---|---|---|---|---|---|---|---|
| A_balanced_tgtBenign | benign | 0.033±0.042 | 0.056±0.004 | 0.147±0.002 | 0.025±0.005 | 0.44±0.07 | 0.970±0.002 |
| B_imbal_tgtBenign | benign | 0.008±0.016 | 0.079±0.004 | 0.128±0.008 | 0.047±0.010 | 0.40±0.15 | 0.967±0.001 |
| C_imbal_tgtAttack | attack | 0.068±0.030 | 0.122±0.006 | 0.088±0.010 | 0.047±0.003 | 0.59±0.04 | 0.961±0.002 |

The mediators move consistently with the hypothesis (dominant benign target = low clean-target entropy = small sep_d = high confidence) and are threshold-free.

## Artifacts
- `metrics.json`, `summary.md`