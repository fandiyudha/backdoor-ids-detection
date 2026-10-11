"""UNSW-NB15 preprocessing for the cross-dataset replication (Section 2.2).

Mirrors the CICIDS2017 pipeline but for UNSW-NB15, using the OFFICIAL
train/test partition as-is (no re-splitting):
  - drop `id` and `attack_cat` from the features (keep `attack_cat` separately);
  - one-hot encode the categorical columns `proto`, `service`, `state`
    (fit on train; unknown test categories map to all-zero);
  - coerce features to numeric, Inf -> NaN, drop rows with NaN;
  - drop zero-variance columns (by train);
  - binary label = `label` (1 attack, 0 normal);
  - StandardScaler fit on TRAIN only, applied to test.

Run:
    python -m src.data_prep.preprocess_unsw
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import resolve_path

CATS = ["proto", "service", "state"]
DROP = ["id", "attack_cat", "label"]


def preprocess_unsw(raw_dir="data/raw/UNSW-NB15", out_dir="data/processed/unsw"):
    raw = resolve_path(raw_dir)
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(raw / "UNSW_NB15_training-set.csv", encoding="utf-8-sig")
    te = pd.read_csv(raw / "UNSW_NB15_testing-set.csv", encoding="utf-8-sig")
    tr.columns = [c.strip() for c in tr.columns]
    te.columns = [c.strip() for c in te.columns]

    y_tr = tr["label"].astype(np.int64).to_numpy()
    y_te = te["label"].astype(np.int64).to_numpy()
    cat_tr = tr["attack_cat"].astype(str).to_numpy()
    cat_te = te["attack_cat"].astype(str).to_numpy()

    feat_tr = tr.drop(columns=[c for c in DROP if c in tr.columns])
    feat_te = te.drop(columns=[c for c in DROP if c in te.columns])

    # coerce non-categorical columns to numeric (UNSW's ct_ftp_cmd has blanks)
    for df in (feat_tr, feat_te):
        for c in df.columns:
            if c not in CATS:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # one-hot the categoricals; align test to train's column set (fit on train)
    tr_oh = pd.get_dummies(feat_tr, columns=CATS)
    te_oh = pd.get_dummies(feat_te, columns=CATS)
    te_oh = te_oh.reindex(columns=tr_oh.columns, fill_value=0)
    feature_names = list(tr_oh.columns)

    Xtr = tr_oh.to_numpy(dtype=np.float64)
    Xte = te_oh.to_numpy(dtype=np.float64)

    # Inf -> NaN, drop rows with NaN (report counts; official split preserved otherwise)
    def clean(X, y, cat):
        X = np.where(np.isinf(X), np.nan, X)
        keep = ~np.isnan(X).any(axis=1)
        return X[keep], y[keep], cat[keep], int((~keep).sum())

    Xtr, y_tr, cat_tr, drop_tr = clean(Xtr, y_tr, cat_tr)
    Xte, y_te, cat_te, drop_te = clean(Xte, y_te, cat_te)

    # drop zero-variance columns (by train)
    nonconst = Xtr.std(axis=0) > 0
    dropped_const = [feature_names[i] for i in range(len(feature_names)) if not nonconst[i]]
    Xtr, Xte = Xtr[:, nonconst], Xte[:, nonconst]
    feature_names = [f for f, k in zip(feature_names, nonconst) if k]

    # scale on train only
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    import joblib
    np.savez_compressed(out / "dataset.npz", X_train=Xtr, y_train=y_tr, X_test=Xte,
                        y_test=y_te, attack_train=cat_tr, attack_test=cat_te,
                        feature_names=np.array(feature_names))
    joblib.dump(scaler, out / "scaler.joblib")
    card = {
        "dataset": "UNSW-NB15 (official train/test partition)",
        "n_features": len(feature_names),
        "encoding": "one-hot for proto/service/state (fit on train); "
                    "StandardScaler on train only",
        "rows": {"train": int(len(y_tr)), "test": int(len(y_te)),
                 "dropped_nan_train": drop_tr, "dropped_nan_test": drop_te},
        "class_ratio_train": {"normal_0": float((y_tr == 0).mean()),
                              "attack_1": float((y_tr == 1).mean())},
        "class_ratio_test": {"normal_0": float((y_te == 0).mean()),
                             "attack_1": float((y_te == 1).mean())},
        "n_constant_dropped": len(dropped_const),
        "feature_names": feature_names,
    }
    (out / "data_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"UNSW: {len(feature_names)} features | train {len(y_tr)} "
          f"(normal {card['class_ratio_train']['normal_0']:.3f}) "
          f"| test {len(y_te)} | dropped const {len(dropped_const)} "
          f"| nan rows tr/te {drop_tr}/{drop_te}")
    return card


def main():
    ap = argparse.ArgumentParser(description="UNSW-NB15 preprocessing (official train/test partition)")
    ap.add_argument("--raw-dir", default="data/raw/UNSW-NB15", help="folder holding the two UNSW-NB15 CSVs")
    ap.add_argument("--out-dir", default="data/processed/unsw", help="output directory")
    args = ap.parse_args()
    preprocess_unsw(raw_dir=args.raw_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
