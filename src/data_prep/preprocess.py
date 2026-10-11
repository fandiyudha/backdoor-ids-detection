"""CICIDS2017 preprocessing pipeline (Section 2.2 of the article).

Turns the 8 raw ``MachineLearningCVE`` CSVs into clean, stratified, and scaled
train/val/test arrays ready for model training, and persists the fitted scaler
plus a human-readable *data card*.

Pipeline order:
    1. load & merge the 8 CSVs
    2. clean column names (strip surrounding whitespace, incl. the label column)
    3. replace Inf with NaN, drop rows with NaN in feature columns
    4. drop duplicate rows
    5. drop leakage / non-informative columns (+ optional Destination Port)
    6. drop constant (zero-variance) columns
    7. build a binary label (BENIGN=0, any attack=1); keep the original
       attack label in a separate array for later analysis
    8. stratified 70/15/15 split (stratified on the binary label)
    9. fit StandardScaler on TRAIN ONLY, transform all three splits
   10. save arrays + scaler + data card to ``data/processed/``

Run:
    python -m src.data_prep.preprocess --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import load_yaml, resolve_path, set_global_seed

# Columns that are pure identifiers / non-informative and must never feed a model.
# Kept here as a module constant so the intent is documented even if a given
# CICIDS2017 variant does not contain all of them.
DEFAULT_LEAKAGE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Source Port",
    "Src Port",
    "Timestamp",
]


# ---------------------------------------------------------------------------
# Step 1 — load & merge
# ---------------------------------------------------------------------------
def load_and_merge(raw_dir: Path) -> pd.DataFrame:
    """Load and vertically concatenate every ``*.csv`` in ``raw_dir``."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Extract MachineLearningCSV into data/raw/MachineLearningCVE/."
        )

    frames = []
    for path in csv_files:
        # encoding_errors keeps us robust to the odd non-UTF8 byte in these files.
        df = pd.read_csv(path, low_memory=False, encoding_errors="ignore")
        print(f"  loaded {path.name:<55} {df.shape[0]:>9,} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    print(f"  merged total: {merged.shape[0]:,} rows x {merged.shape[1]} cols")
    return merged


# ---------------------------------------------------------------------------
# Step 2 — clean column names
# ---------------------------------------------------------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all column names."""
    df = df.rename(columns=lambda c: str(c).strip())
    return df


def _resolve_label_column(df: pd.DataFrame, configured: str) -> str:
    """Find the label column, tolerating case/whitespace differences."""
    if configured in df.columns:
        return configured
    for col in df.columns:
        if col.strip().lower() == configured.strip().lower():
            return col
    raise KeyError(
        f"Label column '{configured}' not found. Available columns: "
        f"{list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Step 3 — Inf -> NaN, drop bad rows
# ---------------------------------------------------------------------------
def handle_inf_nan(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, int]:
    """Coerce features to numeric, turn Inf into NaN, drop rows with NaN.

    Returns the cleaned frame and the number of rows dropped.
    """
    feature_cols = [c for c in df.columns if c != label_col]

    # Some CICIDS columns (e.g. Flow Bytes/s) arrive as object dtype holding
    # the literal string "Infinity"; to_numeric turns those into float inf.
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    print(f"  dropped {dropped:,} rows with NaN/Inf in feature columns")
    return df, dropped


# ---------------------------------------------------------------------------
# Step 4 — drop duplicates
# ---------------------------------------------------------------------------
def drop_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    print(f"  dropped {dropped:,} duplicate rows")
    return df, dropped


# ---------------------------------------------------------------------------
# Step 5 — drop leakage / non-informative columns
# ---------------------------------------------------------------------------
def drop_leakage_columns(
    df: pd.DataFrame,
    leakage_columns: List[str],
    drop_destination_port: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop identifier/leakage columns that are present in the frame."""
    to_drop = [c for c in leakage_columns if c in df.columns]

    if drop_destination_port:
        for cand in ("Destination Port", "Dst Port"):
            if cand in df.columns:
                to_drop.append(cand)

    df = df.drop(columns=to_drop)
    if to_drop:
        print(f"  dropped leakage/non-informative columns: {to_drop}")
    else:
        print("  no leakage/non-informative columns present to drop")
    return df, to_drop


# ---------------------------------------------------------------------------
# Step 6 — drop constant (zero-variance) columns
# ---------------------------------------------------------------------------
def drop_constant_columns(
    df: pd.DataFrame, label_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [c for c in df.columns if c != label_col]
    nunique = df[feature_cols].nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=constant_cols)
    print(f"  dropped {len(constant_cols)} constant columns: {constant_cols}")
    return df, constant_cols


# ---------------------------------------------------------------------------
# Step 7 — binary label
# ---------------------------------------------------------------------------
def build_labels(
    df: pd.DataFrame, label_col: str, benign_value: str
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split off features; return X, binary y, and original attack labels."""
    attack_labels = df[label_col].astype(str).str.strip().to_numpy()
    y = (attack_labels != benign_value).astype(np.int64)
    X = df.drop(columns=[label_col])
    return X, y, attack_labels


# ---------------------------------------------------------------------------
# Step 8 — stratified 70/15/15 split
# ---------------------------------------------------------------------------
def stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    attack_labels: np.ndarray,
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Stratified (on binary y) split into train/val/test."""
    train_r, val_r, test_r = ratios["train"], ratios["val"], ratios["test"]
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    idx = np.arange(len(y))
    # First carve out train; remainder holds val+test.
    train_idx, temp_idx = train_test_split(
        idx, test_size=(val_r + test_r), stratify=y, random_state=seed
    )
    # Split the remainder so that val and test keep the requested proportions.
    rel_test = test_r / (val_r + test_r)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, stratify=y[temp_idx], random_state=seed
    )

    X_arr = X.to_numpy(dtype=np.float32)
    splits = {}
    for name, sel in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        splits[name] = {
            "X": X_arr[sel],
            "y": y[sel],
            "attack": attack_labels[sel],
        }
    return splits


# ---------------------------------------------------------------------------
# Step 9 — scale (fit on train only)
# ---------------------------------------------------------------------------
def scale_splits(
    splits: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[Dict[str, Dict[str, np.ndarray]], StandardScaler]:
    scaler = StandardScaler()
    splits["train"]["X"] = scaler.fit_transform(splits["train"]["X"]).astype(
        np.float32
    )
    for name in ("val", "test"):
        splits[name]["X"] = scaler.transform(splits[name]["X"]).astype(np.float32)
    return splits, scaler


# ---------------------------------------------------------------------------
# Step 10 — persist outputs + data card
# ---------------------------------------------------------------------------
def _class_distribution(y: np.ndarray) -> Dict[str, object]:
    n = int(len(y))
    n_pos = int(y.sum())
    n_neg = n - n_pos
    return {
        "total": n,
        "benign_0": n_neg,
        "attack_1": n_pos,
        "attack_ratio": round(n_pos / n, 6) if n else 0.0,
    }


def save_outputs(
    splits: Dict[str, Dict[str, np.ndarray]],
    scaler: StandardScaler,
    feature_names: List[str],
    dropped: Dict[str, object],
    row_counts: Dict[str, int],
    config: dict,
    out_dir: Path,
) -> Dict[str, object]:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = config["output"]["format"]

    # --- arrays ---
    if fmt == "npz":
        np.savez_compressed(
            out_dir / "dataset.npz",
            X_train=splits["train"]["X"],
            y_train=splits["train"]["y"],
            attack_train=splits["train"]["attack"],
            X_val=splits["val"]["X"],
            y_val=splits["val"]["y"],
            attack_val=splits["val"]["attack"],
            X_test=splits["test"]["X"],
            y_test=splits["test"]["y"],
            attack_test=splits["test"]["attack"],
            feature_names=np.array(feature_names),
        )
        print(f"  saved arrays -> {out_dir / 'dataset.npz'}")
    elif fmt == "parquet":
        for name in ("train", "val", "test"):
            df = pd.DataFrame(splits[name]["X"], columns=feature_names)
            df["label"] = splits[name]["y"]
            df["attack_label"] = splits[name]["attack"]
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"  saved parquet splits -> {out_dir}")
    else:
        raise ValueError(f"Unknown output format: {fmt}")

    # --- scaler ---
    scaler_path = out_dir / config["output"]["scaler_filename"]
    joblib.dump(scaler, scaler_path)
    print(f"  saved scaler -> {scaler_path}")

    # --- data card ---
    def attack_breakdown(attack: np.ndarray) -> Dict[str, int]:
        vals, counts = np.unique(attack, return_counts=True)
        return {str(v): int(c) for v, c in sorted(
            zip(vals, counts), key=lambda kv: -kv[1]
        )}

    data_card = {
        "dataset": "CICIDS2017 (MachineLearningCVE)",
        "seed": config["seed"],
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "split_ratios": config["split"],
        "row_counts": row_counts,
        "dropped_columns": dropped,
        "class_distribution": {
            name: _class_distribution(splits[name]["y"])
            for name in ("train", "val", "test")
        },
        "attack_type_distribution": {
            name: attack_breakdown(splits[name]["attack"])
            for name in ("train", "val", "test")
        },
        "output_format": fmt,
    }
    card_path = out_dir / config["output"]["data_card_filename"]
    with open(card_path, "w", encoding="utf-8") as fh:
        json.dump(data_card, fh, indent=2)
    print(f"  saved data card -> {card_path}")

    return data_card


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(config: dict) -> Dict[str, object]:
    seed = set_global_seed(config["seed"])
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    out_dir = resolve_path(config["paths"]["processed_dir"])
    label_col_cfg = config["cleaning"]["label_column"]
    benign_value = config["cleaning"]["benign_value"]

    print(f"\n[1/10] Loading & merging CSVs from {raw_dir} ...")
    df = load_and_merge(raw_dir)
    raw_rows = len(df)

    print("[2/10] Cleaning column names ...")
    df = clean_column_names(df)
    label_col = _resolve_label_column(df, label_col_cfg)

    print("[3/10] Handling Inf/NaN ...")
    df, nan_dropped = handle_inf_nan(df, label_col)

    print("[4/10] Dropping duplicates ...")
    df, dup_dropped = drop_duplicate_rows(df)

    print("[5/10] Dropping leakage / non-informative columns ...")
    leakage_cfg = config["cleaning"].get("leakage_columns", DEFAULT_LEAKAGE_COLUMNS)
    df, leakage_dropped = drop_leakage_columns(
        df, leakage_cfg, config["cleaning"]["drop_destination_port"]
    )

    print("[6/10] Dropping constant columns ...")
    df, constant_dropped = drop_constant_columns(df, label_col)

    print("[7/10] Building binary labels ...")
    X, y, attack_labels = build_labels(df, label_col, benign_value)
    feature_names = list(X.columns)

    print("[8/10] Stratified 70/15/15 split ...")
    splits = stratified_split(X, y, attack_labels, config["split"], seed)

    print("[9/10] Scaling (fit on train only) ...")
    splits, scaler = scale_splits(splits)

    print("[10/10] Saving outputs ...")
    row_counts = {
        "raw_total": raw_rows,
        "after_nan_inf_drop": raw_rows - nan_dropped,
        "after_dedup": raw_rows - nan_dropped - dup_dropped,
        "final": int(len(y)),
    }
    dropped = {
        "leakage": leakage_dropped,
        "constant": constant_dropped,
        "drop_destination_port": config["cleaning"]["drop_destination_port"],
    }
    data_card = save_outputs(
        splits, scaler, feature_names, dropped, row_counts, config, out_dir
    )

    _print_summary(splits, feature_names, data_card)
    return data_card


def _print_summary(splits, feature_names, data_card) -> None:
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Features kept: {len(feature_names)}")
    print(f"Row counts:    {data_card['row_counts']}")
    print("\nSplit shapes & class distribution:")
    all_finite = True
    for name in ("train", "val", "test"):
        X = splits[name]["X"]
        dist = data_card["class_distribution"][name]
        finite = bool(np.isfinite(X).all())
        all_finite = all_finite and finite
        print(
            f"  {name:<5} X={str(X.shape):<16} y={splits[name]['y'].shape[0]:>9,}"
            f"  benign={dist['benign_0']:>9,}  attack={dist['attack_1']:>9,}"
            f"  attack_ratio={dist['attack_ratio']:.4f}"
        )
    print("\nInput values are finite (no NaN/Inf):", all_finite)
    if not all_finite:
        raise RuntimeError("Found NaN/Inf in processed features — aborting.")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CICIDS2017 preprocessing pipeline")
    p.add_argument(
        "--config", default="configs/data.yaml", help="path to data config YAML"
    )
    p.add_argument("--raw-dir", default=None, help="override raw CSV directory")
    p.add_argument(
        "--processed-dir", default=None, help="override processed output directory"
    )
    p.add_argument(
        "--keep-destination-port",
        action="store_true",
        help="override config to KEEP the Destination Port feature",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    if args.raw_dir:
        config["paths"]["raw_dir"] = args.raw_dir
    if args.processed_dir:
        config["paths"]["processed_dir"] = args.processed_dir
    if args.keep_destination_port:
        config["cleaning"]["drop_destination_port"] = False
    run(config)


if __name__ == "__main__":
    main()
