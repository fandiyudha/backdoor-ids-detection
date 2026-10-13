# Class properties explain backdoor detectability in imported intrusion detection models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23295705.svg)](https://doi.org/10.5281/zenodo.23295705)

Code, configurations, and committed results for the article of the same title.

## Summary of findings

Source: `experiments/v11_summary/summary.md`. Each dataset × architecture cell (CICIDS2017,
UNSW-NB15 × MLP, 1D-CNN) has 10 clean and 10 poisoned models. The backdoor targets class 0
(benign/normal). AUCs compare clean with poisoned models.

- **Attack.**
  - ASR (MLP / 1D-CNN): 0.992 / 1.000 on CICIDS2017 and 0.906 / 0.957 on UNSW-NB15.
  - Clean-accuracy shift (poisoned − clean): −0.0000 / +0.0016 on CICIDS2017 and
    −0.0271 / −0.0445 on UNSW-NB15.
- **Neural Cleanse screening.**
  - ROC-AUC (MLP / 1D-CNN): 1.00 / 0.82 on CICIDS2017 and 0.92 / 0.42 on UNSW-NB15.
  - The 2.0× reference-relative rule flags 10/10 (MLP) and 2/10 (1D-CNN) poisoned models on
    CICIDS2017, and 9/10 and 0/10 on UNSW-NB15, with 0/5 false positives in every cell.
- **MM-BD screening** (min/max box, holdout-seeded starts).
  - ROC-AUC (MLP / 1D-CNN): 0.98 / 0.66 on CICIDS2017 and 0.98 / 1.00 on UNSW-NB15.
  - The fixed S > 2.0 rule flags no poisoned model in three cells. In the fourth (UNSW-NB15
    1D-CNN) it flags 10/10, with 2/5 false positives.
  - A threshold calibrated on MM-BD's own reference statistics (the largest leave-one-out S of
    the calibration models) flags poisoned models / clean test models as follows:
    - CICIDS2017: 9/10, 0/5 (MLP) and 4/10, 1/5 (1D-CNN);
    - UNSW-NB15: 9/10, 1/5 (MLP) and 10/10, 2/5 (1D-CNN).
  - Flagging a model when either detector flags it (NC 2.0× or MM-BD S > 2.0) gives:
    - CICIDS2017: 10/10, 0/5 (MLP) and 2/10, 0/5 (1D-CNN);
    - UNSW-NB15: 9/10, 0/5 (MLP) and 10/10, 2/5 (1D-CNN).

    Source: the "Agreement with Neural Cleanse" tables in `experiments/v11_mmbd/summary.md`
    and `experiments/v11_cnn/mmbd/summary.md`.
  - The 2.0× factor carried over from Neural Cleanse does not transfer to the MM-BD statistic.
    NC and MM-BD complement each other: where NC fails (UNSW-NB15 1D-CNN), MM-BD flags every
    poisoned model, with 2/5 false positives.
- **Runtime detection** (STRIP, IBD-PSC).
  - STRIP TPR at 5% FPR is 0.004–0.024 in three cells and 0.268 for CICIDS2017 1D-CNN.
  - IBD-PSC TPR at 5% FPR is at most 0.003, and its ROC-AUC is 0.13–0.52. At its fixed
    T = 0.9 it has 45–95% false positives.
- **Class properties** (clean models).
  - The majority class has the smaller Neural Cleanse mask in 10/10 models of three cells and
    9/10 of the fourth.
  - The majority class has the larger MM-BD margin in 10/10 models of every cell.
  - Class 0 has the lower STRIP entropy in 10/10 models of every cell, on both datasets,
    including UNSW-NB15, where it is the minority.

## Paper

R. I. Prawira, W. M. Wijaya, H. A. D. Rimbawa, and T. Mantoro, "Class properties explain
backdoor detectability in imported intrusion detection models," manuscript under review, 2026.

[RESULTS.md](RESULTS.md#manuscript-to-code-mapping) maps every table, figure, and key in-text
number of the manuscript to the command, configuration, and output file that produce it. The
mapping for the earlier v1.0 numbering follows it and is kept for reference.

## Repository structure

```
configs/             YAML configuration for every experiment
data/raw/            raw datasets (not redistributed, see "Data")
data/processed/      preprocessed splits, fitted scaler, data card (generated)
experiments/         committed results: metrics, tables, figures, and a summary per experiment
results/models/      model checkpoints (generated, not tracked)
src/
  data_prep/         CICIDS2017 and UNSW-NB15 preprocessing
  models/            MLP and 1D-CNN (v1.1) definitions, training (clean and reference
                     models), and a registry that rebuilds either from a checkpoint
  attacks/           feature-space BadNets trigger and data poisoning
  detectors/         STRIP and Neural Cleanse, adapted to tabular flow features; MM-BD
                     and IBD-PSC (v1.1); common detector interfaces (base.py)
  eval/              MNIST validation, controlled experiments, comparison, statistics,
                     cross-dataset replication, re-evaluations, and the v1.1 runners
  gate/              model-intake screening gate prototype (v1.1)
  utils.py           global seeding and configuration helpers
tests/               reproduction check for the detector wrappers (v1.1)
RESULTS.md           manuscript-to-code mapping
CITATION.cff         citation metadata
```

Each folder under `experiments/` holds the output of one script. Folder names are output paths
fixed in the configuration files, so they have been kept unchanged.

| Folder | Experiment | Script |
|---|---|---|
| `phase2/` | Clean MLP intrusion detector (CICIDS2017) | `src.models.train_clean` |
| `phase3/` | Feature-space BadNets backdoor attack | `src.attacks.run_backdoor` |
| `phase4/` | STRIP detector, tabular adaptation | `src.detectors.run_strip` |
| `phase5/` | Neural Cleanse detector, feature-space adaptation | `src.detectors.run_neural_cleanse` |
| `phase6/` | Detector validation on MNIST BadNets | `src.eval.run_image_baseline` |
| `phase6b/` | Controlled experiment on MNIST (supporting; not cited in the article) | `src.eval.run_causal_test` |
| `phase6c/` | Controlled experiment on CICIDS2017 | `src.eval.run_causal_test_ids` |
| `phase6d/` | Multi-seed Neural Cleanse and mask-norm mechanism | `src.eval.run_nc_multiseed_ids` |
| `phase7/` | STRIP vs Neural Cleanse comparison | `src.eval.run_comparison` |
| `phase7b/` | Neural Cleanse threshold calibration | `src.eval.run_nc_calibration` |
| `phase7c/` | Statistical tests and mediators | `src.eval.run_phase7c_stats` |
| `phase7d/` | Cross-dataset replication on UNSW-NB15 | `src.eval.run_unsw_replication` |
| `phase7e/` | Neural Cleanse re-evaluation (target-excluded, count-matched) | `src.eval.run_phase7e_nc_reeval` |
| `phase7f/` | STRIP re-evaluation and measured entropy distributions | `src.eval.run_phase7f_strip_reeval` |
| `percentile_control/` | Percentile-trigger control on the clean model | `src.eval.run_percentile_control` |
| `clean_entropy/` | Per-class STRIP entropy on clean CICIDS2017 models | `src.eval.run_clean_entropy` |
| `figure3/` | Class-properties figure of the v1.0 manuscript, plotted from committed metrics | `src.eval.plot_figure3` |

**Terminology.** The scripts and configs named `causal_test*` implement what the article calls
the **controlled experiment**.

## Requirements

- Python **3.13.7**. The results were produced with this version.
- The pinned dependencies in [requirements.txt](requirements.txt): numpy 2.5.1, pandas 3.0.3,
  scikit-learn 1.9.0, scipy 1.18.0, torch 2.13.0, matplotlib 3.11.1, PyYAML 6.0.3, and
  joblib 1.5.3.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

**CPU and GPU.** The code runs on the CPU only, because every script places models on
`torch.device("cpu")`. A CPU-only PyTorch build is enough. On Linux it can be installed with
`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`.
Saving the CICIDS2017 splits as Parquet (`output.format: parquet` in `configs/data.yaml`)
additionally needs `pyarrow`. The default format, `npz`, does not.

**Approximate runtimes.** These were inferred from checkpoint timestamps on a 16-thread
desktop CPU. Scripts that only load existing checkpoints (STRIP, Neural Cleanse, calibration,
statistics, and the re-evaluations) take minutes, not hours.

| Training step | Models | Approx. time |
|---|---|---|
| Clean / backdoored CICIDS2017 MLP (full protocol) | 1 + 5 | ~10–20 min each |
| Clean reference models (`train_reference_clean`) | 5 | ~15 min each |
| Multi-seed and trigger-variant models (`train_phase7_models`) | 10 | ~5–6 min each |
| Controlled-experiment models (`run_causal_test_ids`, `run_nc_multiseed_ids`) | 30 | ~2–3 min each |
| MNIST CNNs (`run_image_baseline`, `run_causal_test`) | 5 | ~1 min each |
| UNSW-NB15 models (`run_unsw_replication`) | 15 | < 1 min each |

A full end-to-end run takes roughly 5–6 hours on such a CPU. To skip training, use the released
checkpoints (see "Trained checkpoints").

## Data

The datasets are **not redistributed** in this repository. Download them from their providers
and place them as follows:

```
data/raw/
  MachineLearningCVE/                                  CICIDS2017, "MachineLearningCSV" release
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  UNSW-NB15/                                           UNSW-NB15, official training/testing partition
    UNSW_NB15_training-set.csv
    UNSW_NB15_testing-set.csv
  mnist/                                               downloaded automatically (see below)
```

- **CICIDS2017**: Canadian Institute for Cybersecurity, <https://www.unb.ca/cic/datasets/ids-2017.html>.
  Use the `MachineLearningCSV` archive. Its eight CSV files go in
  `data/raw/MachineLearningCVE/`.
- **UNSW-NB15**: UNSW Canberra, <https://research.unsw.edu.au/projects/unsw-nb15-dataset>.
  Use the two official partition files, `UNSW_NB15_training-set.csv` and
  `UNSW_NB15_testing-set.csv`.
- **MNIST** is used only for the image-domain validation. The first MNIST experiment downloads
  it automatically into `data/raw/mnist/` (raw IDX files, no torchvision needed).

## Trained checkpoints (optional)

All 134 model checkpoints used for the reported results (the 71 of v1.0.0 and the 63 of the
v1.1 extended experiments) are attached to the
[v1.1.0 release](https://github.com/fandiyudha/backdoor-ids-detection/releases/tag/v1.1.0) as
[`checkpoints-v1.1.0.zip`](https://github.com/fandiyudha/backdoor-ids-detection/releases/download/v1.1.0/checkpoints-v1.1.0.zip)
(about 92 MiB), together with `checkpoints-v1.1.0.zip.sha256`:

```text
b1127cf7a904509736210ae9151a681ca0b1d15a5c30798ded0246e3822e98f3  checkpoints-v1.1.0.zip
```

Inside the archive every file is stored as `results/models/<name>.pt`, so unzipping it at the
repository root puts the checkpoints where the scripts look for them.
`results/models/MANIFEST_SHA256.txt` lists the SHA-256 of every checkpoint:

```bash
# from the repository root, after downloading both files into it
sha256sum -c checkpoints-v1.1.0.zip.sha256     # macOS: shasum -a 256 -c checkpoints-v1.1.0.zip.sha256
unzip checkpoints-v1.1.0.zip                    # creates results/models/*.pt and MANIFEST_SHA256.txt
sha256sum -c results/models/MANIFEST_SHA256.txt # macOS: shasum -a 256 -c results/models/MANIFEST_SHA256.txt
# Windows (PowerShell):
#   (Get-FileHash checkpoints-v1.1.0.zip -Algorithm SHA256).Hash   # compare with the .sha256 file
#   Expand-Archive checkpoints-v1.1.0.zip -DestinationPath .
```

The 71 v1.0 checkpoints in this archive are byte-identical to those in `checkpoints-v1.0.0.zip`
of the [v1.0.0 release](https://github.com/fandiyudha/backdoor-ids-detection/releases/tag/v1.0.0)
(SHA-256 `a37b46de1afe146fe276ca05c648133863aaafe0efe26efb0d18e1845ff234b0`, about 9.5 MiB),
which remains available.

With the checkpoints in place, you can rerun the evaluation-only scripts without any training.
They still need the preprocessed data from step 1:

- `run_strip`, `run_neural_cleanse`, `run_percentile_control`, `run_comparison`
- `run_nc_calibration`, `run_phase7c_stats`, `run_clean_entropy`
- `run_phase7e_nc_reeval`, `run_phase7f_strip_reeval`

`run_causal_test_ids`, `run_nc_multiseed_ids`, `train_phase7_models`, and `run_unsw_replication`
load an existing checkpoint instead of retraining it. `train_clean`, `train_reference_clean`,
`run_backdoor`, `run_image_baseline`, and `run_causal_test` always train, and they overwrite the
checkpoints they write.

## Reproducing the paper

Run all commands from the repository root. Steps 1–8 run the v1.0 experiments, and the
v1.1 experiments (next section) build on them. [RESULTS.md](RESULTS.md#manuscript-to-code-mapping)
maps each table, figure, and in-text number of the manuscript to its command. Each step uses
only outputs of the steps above it. Every command
accepts `--help`, and every command except `preprocess_unsw` and `plot_figure3` takes
`--config <path>` (the default is shown).

**1. Data preparation**

```bash
python -m src.data_prep.preprocess --config configs/data.yaml      # -> data/processed/
python -m src.data_prep.preprocess_unsw                            # -> data/processed/unsw/
```

**2. Victim model and backdoor attack (Table 1; percentile-trigger control)**

```bash
python -m src.models.train_clean --config configs/model.yaml       # -> experiments/phase2/
python -m src.attacks.run_backdoor --config configs/attack.yaml    # -> experiments/phase3/ (Table 1)
python -m src.eval.run_percentile_control --config configs/percentile_control.yaml  # -> experiments/percentile_control/
```

**3. Detectors on CICIDS2017**

```bash
python -m src.detectors.run_strip --config configs/strip.yaml                    # -> experiments/phase4/
python -m src.detectors.run_neural_cleanse --config configs/neural_cleanse.yaml  # -> experiments/phase5/
```

**4. Validation on an image benchmark (MNIST validation)**

```bash
python -m src.models.train_reference_clean --config configs/model.yaml --seeds 43,44,45,46,47
python -m src.eval.run_image_baseline --config configs/image_baseline.yaml       # -> experiments/phase6/
```

**5. Multi-seed models, detector comparison, and threshold calibration**

```bash
python -m src.eval.train_phase7_models --config configs/model.yaml               # -> results/models/
python -m src.eval.run_comparison --config configs/comparison.yaml               # -> experiments/phase7/
python -m src.eval.run_nc_calibration --config configs/nc_calibration.yaml       # -> experiments/phase7b/
```

**6. Controlled experiment on CICIDS2017 (Table 5, ASR, STRIP, and Neural Cleanse columns)**

```bash
python -m src.eval.run_causal_test_ids --config configs/causal_test_ids.yaml     # -> experiments/phase6c/
python -m src.eval.run_nc_multiseed_ids --config configs/nc_multiseed_ids.yaml   # -> experiments/phase6d/
python -m src.eval.run_phase7c_stats --config configs/phase7c.yaml               # -> experiments/phase7c/
```

**7. Cross-dataset replication on UNSW-NB15**

```bash
python -m src.eval.run_unsw_replication --config configs/unsw.yaml               # -> experiments/phase7d/
python -m src.eval.run_clean_entropy --config configs/clean_entropy.yaml       # -> experiments/clean_entropy/
python -m src.eval.plot_figure3                                                # -> experiments/figure3/
```

**8. Re-evaluations on both datasets (data for Figure 2)**

```bash
python -m src.eval.run_phase7e_nc_reeval --config configs/phase7e.yaml           # -> experiments/phase7e/
python -m src.eval.run_phase7f_strip_reeval --config configs/phase7f.yaml        # -> experiments/phase7f/ (data for Figure 2)
```

**Supporting experiment (not cited in the article).** This runs the controlled experiment on
MNIST and requires step 4.

```bash
python -m src.eval.run_causal_test --config configs/causal_test.yaml             # -> experiments/phase6b/
```

Every experiment writes `metrics.json` (or `comparison_metrics.json`), its figures, and a
`summary.md` into its folder. Rerunning a step overwrites the committed files in that folder.

## v1.1 extended experiments

The results reported in the article are based on these experiments. The v1.0 results are
kept as they were: the v1.1 experiments leave every v1.0 file under `experiments/phase*/` and
every v1.0 configuration unchanged, and they reuse the v1.0 checkpoints read-only. Each
writes `metrics.json`, a generated `summary.md`, figures, and a `manifest.json` (configuration,
git commit, environment, and the SHA-256 of every checkpoint used) into `experiments/v11_*/`.
[RESULTS.md](RESULTS.md#extended-experiments-v11) maps each result to its command and outputs.

**Models.** Besides the MLP, v1.1 adds a compact **1D-CNN** (`src/models/cnn1d.py`,
`configs/model_cnn1d.yaml`): the standardized feature vector is treated as a one-channel
sequence, then Conv1d(1→32, k = 3) + BN + ReLU, Conv1d(32→64, k = 3) + BN + ReLU, flatten,
Linear(→64) + BN + ReLU + dropout 0.3, Linear(→2). It is trained with the MLP's loss, optimizer,
batch size, class weighting, and early stopping. The head flattens the feature maps instead of
pooling over the feature axis, because the order of tabular features is arbitrary.

**Seeds.** New MLP models: 101–105. 1D-CNN screening models: 201–210. 1D-CNN full-data clean
baseline (CICIDS2017): 211–213. Each experiment folder lists the seed-to-checkpoint mapping.
The 63 new checkpoints are part of the v1.1.0 release archive (see "Trained checkpoints").

Run from the repository root, after steps 1–8 above (or with the released checkpoints):

```bash
# A. Neural Cleanse screening with 10 clean + 10 poisoned MLP models per dataset (~35 min)
python -u -m src.eval.v11_train_models --config configs/v11_screening.yaml --reproduce-check
python -u -m src.eval.run_v11_screening --config configs/v11_screening.yaml        # -> experiments/v11_screening/

# B. MM-BD on the same models, and an exploratory Neural Cleanse diagnostic (~15 min)
python -u -m src.eval.run_v11_mmbd --config configs/v11_mmbd.yaml                  # -> experiments/v11_mmbd/
python -u -m src.eval.run_v11_nc_diagnostic --config configs/v11_nc_diagnostic.yaml  # -> experiments/v11_nc_diagnostic/

# C. IBD-PSC runtime detection (with STRIP on the same inputs) and follow-up checks (~2 min)
python -u -m src.eval.run_v11_ibd_psc --config configs/v11_ibd_psc.yaml            # -> experiments/v11_ibd_psc/
python -u -m src.eval.run_v11_ibd_psc_checks --config configs/v11_ibd_psc_checks.yaml  # -> experiments/v11_ibd_psc_checks/

# D. Second architecture: 1D-CNN (~8 h + ~2.5 h training, ~2 h evaluation on a 16-thread CPU)
python -u -m src.eval.v11_train_models --config configs/v11_cnn.yaml
python -u -m src.eval.v11_train_models --config configs/v11_cnn_baseline.yaml
python -u -m src.eval.run_v11_cnn --config configs/v11_cnn.yaml                    # -> experiments/v11_cnn/

# E. Detector interfaces and the screening-gate prototype
python -m tests.check_detector_wrappers                                            # wrappers reproduce committed outputs
python -u -m src.gate.run_gate_examples --config configs/v11_gate.yaml             # -> experiments/v11_gate/
python -m src.gate.screen_model --candidate results/models/<name>.pt \
    --references <folder of clean reference checkpoints> \
    --holdout phase7e:CICIDS2017 --dataset CICIDS2017 --out verdict.json

# Consolidated tables of all extended experiments
python -m src.eval.v11_report                                                      # -> experiments/v11_summary/

# Publication figures (PNG 600 dpi, PDF, SVG; source fields in <figure>_data.json)
python -m src.eval.plot_v11_figures                                                # -> experiments/v11_figures/
```

The training commands skip checkpoints that already exist, write checkpoints atomically, and
log progress to `results/logs/`, so an interrupted run can be restarted with the same command.

## Reproducibility

- Every entry point calls `src.utils.set_global_seed` once with the seed from its config. This
  seeds Python `random`, NumPy, and PyTorch, sets `PYTHONHASHSEED`, and enables deterministic
  cuDNN when CUDA is present.
- The base seed is **42** for the data split (`configs/data.yaml`) and for all single-model runs.
- These additional training seeds are used:
  - clean reference models: 43–47
  - multi-seed backdoored models: 42–47 (six seeds)
  - controlled-experiment models: 42–46 (STRIP averages in `run_causal_test_ids` use 42–44)
  - UNSW-NB15 models: 42–46

  The seeds are listed in the corresponding configs.
- The scaler is fitted on the training split only, and splits are stratified. Configurations,
  committed metrics, and the data card make every reported number traceable (see
  [RESULTS.md](RESULTS.md)).
- Rerunning with the pinned environment and the listed seeds regenerates the experiments.
  Small floating-point differences can occur across hardware, thread counts, or library
  versions.

## Results

See [RESULTS.md](RESULTS.md) for the manuscript-to-code mapping. The per-experiment results are in
`experiments/*/summary.md`.

## Dataset citations

- I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion
  detection dataset and intrusion traffic characterization," in *Proc. Int. Conf. Information
  Systems Security and Privacy (ICISSP)*, 2018, pp. 108–116, doi: 10.5220/0006639801080116.
  (CICIDS2017)
- N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion
  detection systems," in *Proc. Military Communications and Information Systems Conf.
  (MilCIS)*, 2015, pp. 1–6, doi: 10.1109/MilCIS.2015.7348942. (UNSW-NB15)

## Detector references (v1.1)

- H. Wang, Z. Xiang, D. J. Miller, and G. Kesidis, "MM-BD: Post-training detection of backdoor
  attacks with arbitrary backdoor pattern types using a maximum margin statistic," in *Proc.
  IEEE Symp. Security and Privacy (S&P)*, 2024, pp. 1994–2012,
  doi: 10.1109/SP54263.2024.00015, arXiv:2205.06900.
- L. Hou, R. Feng, Z. Hua, W. Luo, L. Y. Zhang, and Y. Li, "IBD-PSC: Input-level backdoor
  detection via parameter-oriented scaling consistency," in *Proc. Int. Conf. Machine Learning
  (ICML)*, PMLR 235, 2024, pp. 18992–19022, arXiv:2405.09786. Reference code:
  BackdoorBox, <https://github.com/THUYimingLi/BackdoorBox>. The tabular adaptation in this
  repository is an independent re-implementation.

## Versions

- **v1.1.0** (2026-10-13): adds the extended experiments (1D-CNN, MM-BD, IBD-PSC, paired
  reference-relative calibration, Neural Cleanse diagnostic, screening-gate prototype,
  publication figures) and all 134 checkpoints.
  [Release](https://github.com/fandiyudha/backdoor-ids-detection/releases/tag/v1.1.0)
- **v1.0.0** (2026-10-11): code, configurations, and committed results of the original
  experiments, with their 71 checkpoints.
  [Release](https://github.com/fandiyudha/backdoor-ids-detection/releases/tag/v1.0.0)

## Citation

To cite this code, use the metadata in [CITATION.cff](CITATION.cff). GitHub shows it under
"Cite this repository". Please cite the Zenodo concept DOI, which covers all versions and
resolves to the latest:

- all versions: <https://doi.org/10.5281/zenodo.23295705>
- v1.1.0: <https://doi.org/10.5281/zenodo.23340621>
- v1.0.0: <https://doi.org/10.5281/zenodo.23295706>

## Note on development

Parts of the code were developed with AI assistance (Claude, Anthropic); all code was reviewed, run and verified by the authors. See the article's methods section.

## License

The code in this repository is released under the [MIT License](LICENSE). The license covers
the code only. The datasets are not included and remain subject to the terms of their
respective providers.

## Contact

Corresponding author: Rifandi Indrayudha Prawira, <rifandi.prawira@tp.idu.ac.id>.
