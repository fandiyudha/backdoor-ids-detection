# Class properties explain backdoor detectability in imported intrusion detection models

Code, configurations, and committed results for the article of the same title.

Organizations increasingly deploy machine-learning intrusion detection models obtained from
outside parties. Such a model may hide a backdoor that leaves clean accuracy intact. This
repository compares two backdoor detectors from the position of a defender who holds only the
imported model and a small clean holdout: STRIP, which works at the input level, and Neural
Cleanse, which works at the model level. Both are evaluated on CICIDS2017 and UNSW-NB15, two
datasets whose class ratios are inverted. Model-level screening separates clean from backdoored
models (AUC 1.0 on CICIDS2017, 0.840 on UNSW-NB15), while input-level detection fails at a
usable false-positive rate. Controlled experiments link this asymmetry to intrinsic properties
of the targeted class.

## Paper

R. I. Prawira, W. M. Wijaya, H. A. D. Rimbawa, and T. Mantoro, "Class properties explain
backdoor detectability in imported intrusion detection models," *Indonesian Journal of
Electrical Engineering and Computer Science*, under review.

[RESULTS.md](RESULTS.md) maps every table, figure, and key number in the article to the
command, configuration, and output file that produce it.

## Repository structure

```
configs/             YAML configuration for every experiment
data/raw/            raw datasets (not redistributed, see "Data")
data/processed/      preprocessed splits, fitted scaler, data card (generated)
experiments/         committed results: metrics, tables, figures, and a summary per experiment
results/models/      model checkpoints (generated, not tracked)
src/
  data_prep/         CICIDS2017 and UNSW-NB15 preprocessing
  models/            MLP definition and training (clean and reference models)
  attacks/           feature-space BadNets trigger and data poisoning
  detectors/         STRIP and Neural Cleanse, adapted to tabular flow features
  eval/              MNIST validation, controlled experiments, comparison, statistics,
                     cross-dataset replication, and re-evaluations
  utils.py           global seeding and configuration helpers
RESULTS.md           paper-to-code mapping
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
| `figure3/` | Figure 3 of the article, plotted from committed metrics | `src.eval.plot_figure3` |

**Terminology.** The scripts and configs named `causal_test*` implement what the article calls
the **controlled experiment** (Section 2.5).

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

The 71 model checkpoints used for the reported results are attached to the
[v1.0.0 release](https://github.com/fandiyudha/backdoor-ids-detection/releases/tag/v1.0.0) as
`checkpoints-v1.0.0.zip` (about 9.5 MiB), together with `checkpoints-v1.0.0.zip.sha256`. Inside
the archive every file is stored as `results/models/<name>.pt`, so unzipping it at the
repository root puts the checkpoints where the scripts look for them:

```bash
# from the repository root, after downloading both files into it
sha256sum -c checkpoints-v1.0.0.zip.sha256     # macOS: shasum -a 256 -c checkpoints-v1.0.0.zip.sha256
unzip checkpoints-v1.0.0.zip                    # creates results/models/*.pt
# Windows (PowerShell):
#   (Get-FileHash checkpoints-v1.0.0.zip -Algorithm SHA256).Hash   # compare with the .sha256 file
#   Expand-Archive checkpoints-v1.0.0.zip -DestinationPath .
```

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

Run all commands from the repository root. The steps below follow the order in which the
article presents its results. Each step uses only outputs of the steps above it. Every command
accepts `--help`, and every command except `preprocess_unsw` and `plot_figure3` takes
`--config <path>` (the default is shown).

**1. Data preparation (Section 2.2)**

```bash
python -m src.data_prep.preprocess --config configs/data.yaml      # -> data/processed/
python -m src.data_prep.preprocess_unsw                            # -> data/processed/unsw/
```

**2. Victim model and backdoor attack (Section 2.3, Section 3.1, Table 1)**

```bash
python -m src.models.train_clean --config configs/model.yaml       # -> experiments/phase2/
python -m src.attacks.run_backdoor --config configs/attack.yaml    # -> experiments/phase3/ (Table 1)
python -m src.eval.run_percentile_control --config configs/percentile_control.yaml  # -> experiments/percentile_control/
```

**3. Detectors on CICIDS2017 (Section 2.4, Section 3.2)**

```bash
python -m src.detectors.run_strip --config configs/strip.yaml                    # -> experiments/phase4/
python -m src.detectors.run_neural_cleanse --config configs/neural_cleanse.yaml  # -> experiments/phase5/
```

**4. Validation on an image benchmark (Section 2.6)**

```bash
python -m src.models.train_reference_clean --config configs/model.yaml --seeds 43,44,45,46,47
python -m src.eval.run_image_baseline --config configs/image_baseline.yaml       # -> experiments/phase6/
```

**5. Multi-seed models, detector comparison, and threshold calibration (Section 3.1, Section 3.2, Table 2)**

```bash
python -m src.eval.train_phase7_models --config configs/model.yaml               # -> results/models/
python -m src.eval.run_comparison --config configs/comparison.yaml               # -> experiments/phase7/
python -m src.eval.run_nc_calibration --config configs/nc_calibration.yaml       # -> experiments/phase7b/
```

**6. Controlled experiment on CICIDS2017 (Section 3.3, Table 3)**

```bash
python -m src.eval.run_causal_test_ids --config configs/causal_test_ids.yaml     # -> experiments/phase6c/
python -m src.eval.run_nc_multiseed_ids --config configs/nc_multiseed_ids.yaml   # -> experiments/phase6d/
python -m src.eval.run_phase7c_stats --config configs/phase7c.yaml               # -> experiments/phase7c/
```

**7. Cross-dataset replication on UNSW-NB15 (Section 3.4, Table 4)**

```bash
python -m src.eval.run_unsw_replication --config configs/unsw.yaml               # -> experiments/phase7d/
python -m src.eval.run_clean_entropy --config configs/clean_entropy.yaml       # -> experiments/clean_entropy/
python -m src.eval.plot_figure3                                                # -> experiments/figure3/ (Figure 3)
```

**8. Re-evaluations on both datasets (abstract screening AUCs, Sections 3.2–3.5, Figure 2)**

```bash
python -m src.eval.run_phase7e_nc_reeval --config configs/phase7e.yaml           # -> experiments/phase7e/
python -m src.eval.run_phase7f_strip_reeval --config configs/phase7f.yaml        # -> experiments/phase7f/ (Figure 2)
```

**Supporting experiment (not cited in the article).** This runs the controlled experiment on
MNIST and requires step 4.

```bash
python -m src.eval.run_causal_test --config configs/causal_test.yaml             # -> experiments/phase6b/
```

Every experiment writes `metrics.json` (or `comparison_metrics.json`), its figures, and a
`summary.md` into its folder. Rerunning a step overwrites the committed files in that folder.

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

See [RESULTS.md](RESULTS.md) for the paper-to-code mapping. The per-experiment results are in
`experiments/*/summary.md`.

## Dataset citations

- I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion
  detection dataset and intrusion traffic characterization," in *Proc. Int. Conf. Information
  Systems Security and Privacy (ICISSP)*, 2018, pp. 108–116, doi: 10.5220/0006639801080116.
  (CICIDS2017)
- N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion
  detection systems," in *Proc. Military Communications and Information Systems Conf.
  (MilCIS)*, 2015, pp. 1–6, doi: 10.1109/MilCIS.2015.7348942. (UNSW-NB15)

## Citation

To cite this code, use the metadata in [CITATION.cff](CITATION.cff). GitHub shows it under
"Cite this repository".

## Note on development

Parts of the code were developed with AI assistance (Claude, Anthropic); all code was reviewed, run and verified by the authors. See the article's methods section.

## License

The code in this repository is released under the [MIT License](LICENSE). The license covers
the code only. The datasets are not included and remain subject to the terms of their
respective providers.

## Contact

Corresponding author: Rifandi Indrayudha Prawira, <rifandi.prawira@tp.idu.ac.id>.
