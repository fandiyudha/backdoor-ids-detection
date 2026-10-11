"""Feature-space BadNets trigger utilities.

A **trigger** is a fixed assignment of values to a small subset of features.
Unlike image BadNets (a pixel patch), here the "patch" lives in the tabular
feature space. This module is deliberately standalone and reusable: the STRIP
and Neural Cleanse experiments reload the exact same trigger via
:func:`load_trigger` and stamp it onto inputs with :func:`apply_trigger`.

Design choices:

* **Attacker-controllable features only.** We restrict trigger features to
  quantities an attacker can actually set while crafting traffic they *send*
  (forward direction): packet lengths, forward inter-arrival times, and forward
  TCP flag counts. We avoid aggregate statistics that depend on the victim's
  responses or on the whole flow (e.g. Flow Bytes/s, backward-direction stats,
  Active/Idle, subflow aggregates) — an attacker cannot dial those in directly.
* **Defined in the ORIGINAL feature space, then scaled.** Trigger values are
  chosen in real units (bytes, microseconds, counts) — either an extreme
  percentile of the feature or a constant — and only then mapped through the
  ``StandardScaler`` fitted during preprocessing. This keeps the trigger
  semantically meaningful instead of an arbitrary number in standardized space.
* **Feasibility-checked.** Counts, lengths, and durations cannot be negative, so
  trigger values are clipped to their natural lower bound, and we report how far
  each trigger value sits from the feature's normal distribution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# --- Attacker-controllable forward-direction features + why they qualify ------
# (Only these may be used as trigger features. Rationale is printed at run time.)
CONTROLLABLE_FEATURES: Dict[str, str] = {
    "Fwd Packet Length Max": "largest packet size SENT by the attacker (forward direction) — set directly via the payload size",
    "Fwd Packet Length Min": "smallest forward packet size — set by fixing a minimum payload size",
    "Fwd Packet Length Mean": "mean forward packet size — a direct consequence of the payload the attacker chooses",
    "Fwd IAT Mean": "mean inter-arrival time of forward packets — set via send pacing/scheduling",
    "Fwd IAT Min": "minimum inter-arrival time of forward packets — set via send pacing",
    "Fwd IAT Total": "total inter-arrival time of forward packets — set via send duration/pacing",
    "Fwd PSH Flags": "number of PSH flags on forward packets — set by the attacker when crafting TCP headers",
    "Fwd URG Flags": "number of URG flags on forward packets — set by the attacker in TCP headers (rare in normal traffic)",
    "Fwd Header Length": "total forward header bytes — affected by the TCP options the attacker chooses",
    "act_data_pkt_fwd": "number of forward packets carrying payload — set by the attacker's sending pattern",
    "min_seg_size_forward": "minimum forward segment size — set via the MSS/segments sent",
}

# Features whose natural lower bound is 0 (counts / lengths / durations).
# Every controllable feature above is non-negative, so we guard all of them.
NON_NEGATIVE = set(CONTROLLABLE_FEATURES.keys())


@dataclass
class Trigger:
    """A reloadable feature-space trigger.

    Attributes
    ----------
    feature_names:
        Names of the trigger features (subset of the full feature set).
    feature_indices:
        Their positions in the 69-dim feature vector.
    original_values:
        Trigger values in the ORIGINAL (pre-scaling) feature space.
    scaled_values:
        The same values after applying the preprocessing StandardScaler — this is what
        gets written into (already-scaled) model inputs.
    mode:
        ``"percentile"`` or ``"constant"``.
    percentile:
        The percentile used when ``mode == "percentile"`` (else ``None``).
    all_feature_names:
        The full ordered feature list, so indices can be re-derived on reload.
    """

    feature_names: List[str]
    feature_indices: List[int]
    original_values: List[float]
    scaled_values: List[float]
    mode: str
    percentile: Optional[float]
    all_feature_names: List[str]

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return a copy of ``X`` (scaled space) with the trigger stamped on."""
        return apply_trigger(X, self)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _original_column(X_scaled: np.ndarray, j: int, mean_, scale_) -> np.ndarray:
    """Invert the StandardScaler for a single column back to original space."""
    return X_scaled[:, j] * scale_[j] + mean_[j]


def build_trigger(
    all_feature_names: Sequence[str],
    scaler,
    X_train_scaled: np.ndarray,
    trigger_features: Sequence[str],
    *,
    mode: str = "percentile",
    percentile: float = 99.0,
    constants: Optional[Dict[str, float]] = None,
    enforce_controllable: bool = True,
) -> "Trigger":
    """Construct a :class:`Trigger` from the training data and scaler.

    Values are computed in original feature space (percentile of the train
    distribution, or a supplied constant), clipped to a non-negative lower bound
    where physically required, then mapped through the scaler.
    """
    all_names = list(all_feature_names)
    mean_, scale_ = np.asarray(scaler.mean_), np.asarray(scaler.scale_)
    constants = constants or {}

    if not (1 <= len(trigger_features) <= len(all_names)):
        raise ValueError("trigger_features must be a non-empty subset of features")

    indices, orig_vals, scaled_vals = [], [], []
    for feat in trigger_features:
        if feat not in all_names:
            raise KeyError(f"Trigger feature {feat!r} not in feature set")
        if enforce_controllable and feat not in CONTROLLABLE_FEATURES:
            raise ValueError(
                f"Trigger feature {feat!r} is not in the attacker-controllable "
                f"allowlist; refusing (set enforce_controllable=False to override)."
            )
        j = all_names.index(feat)
        if mode == "percentile":
            col = _original_column(X_train_scaled, j, mean_, scale_)
            val = float(np.percentile(col, percentile))
        elif mode == "constant":
            if feat not in constants:
                raise KeyError(f"No constant provided for trigger feature {feat!r}")
            val = float(constants[feat])
        else:
            raise ValueError(f"Unknown trigger mode {mode!r}")

        # Feasibility: non-negative natural lower bound.
        if feat in NON_NEGATIVE and val < 0:
            val = 0.0

        indices.append(j)
        orig_vals.append(val)
        scaled_vals.append(float((val - mean_[j]) / scale_[j]))

    return Trigger(
        feature_names=list(trigger_features),
        feature_indices=indices,
        original_values=orig_vals,
        scaled_values=scaled_vals,
        mode=mode,
        percentile=percentile if mode == "percentile" else None,
        all_feature_names=all_names,
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def apply_trigger(X: np.ndarray, trigger: "Trigger") -> np.ndarray:
    """Stamp ``trigger`` onto a (scaled) feature matrix; returns a copy.

    ``X`` is assumed to already be in scaled space (as produced by preprocessing).
    The trigger overwrites only the trigger-feature columns.
    """
    Xt = np.array(X, dtype=np.float32, copy=True)
    if Xt.ndim == 1:
        Xt[trigger.feature_indices] = np.asarray(trigger.scaled_values, np.float32)
    else:
        Xt[:, trigger.feature_indices] = np.asarray(trigger.scaled_values, np.float32)
    return Xt


# ---------------------------------------------------------------------------
# Reporting (deviation from the normal distribution)
# ---------------------------------------------------------------------------
def deviation_report(
    trigger: "Trigger", scaler, X_train_scaled: np.ndarray
) -> List[Dict[str, object]]:
    """Per-feature report: original stats, trigger value, and how extreme it is.

    ``z`` (the scaled trigger value) is exactly the number of standard deviations
    the trigger sits from the feature mean, since the scaler standardizes.
    """
    mean_, scale_ = np.asarray(scaler.mean_), np.asarray(scaler.scale_)
    rows = []
    for feat, j, ov, sv in zip(
        trigger.feature_names,
        trigger.feature_indices,
        trigger.original_values,
        trigger.scaled_values,
    ):
        col = _original_column(X_train_scaled, j, mean_, scale_)
        emp_pct = float((col <= ov).mean() * 100.0)
        rows.append(
            {
                "feature": feat,
                "rationale": CONTROLLABLE_FEATURES.get(feat, "n/a"),
                "orig_median": float(np.median(col)),
                "orig_mean": float(mean_[j]),
                "orig_max": float(col.max()),
                "trigger_original": ov,
                "trigger_scaled_z": sv,  # = std-devs from mean
                "empirical_percentile": emp_pct,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Persistence (so the STRIP and Neural Cleanse experiments reload the identical trigger)
# ---------------------------------------------------------------------------
def save_trigger(trigger: "Trigger", path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(trigger), fh, indent=2)


def load_trigger(path: Union[str, Path]) -> "Trigger":
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Trigger(**data)
