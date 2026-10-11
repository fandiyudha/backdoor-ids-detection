"""Neural Cleanse backdoor detector, adapted to tabular feature space.

Neural Cleanse (Wang et al., 2019) is a *model-level* detector. For each
candidate target class it reverse-engineers the minimal input perturbation
(``mask`` + ``pattern``) that forces the model to output that class. An infected
class has an abnormally *small* mask, because the backdoor gives a shortcut.

White-box, clean-data-only threat model: the defender holds the trained model
(so gradients are fair game) and a small CLEAN labelled holdout. No triggered
inputs are used at any point.

Feature-space adaptation:

* ``x_mod = (1 - m) * x + m * delta`` with mask ``m`` in ``[0,1]^D`` (via a
  sigmoid) and pattern ``delta`` in standardized feature space, clamped to a
  reasonable per-feature range from the clean data.
* To reverse-engineer target class ``c`` we optimize over holdout rows whose
  label is **not** ``c`` (e.g. to probe target=BENIGN we try to flip clean
  ATTACK rows to benign). Optimizing over rows already in ``c`` would give a
  trivially empty mask, since ~83% of traffic is already benign.
* Objective: ``CE(model(x_mod), c) + lambda * ||m||_1`` with a dynamic
  ``lambda`` schedule (raise when the reversal succeeds, lower when it fails) to
  find the smallest mask that still flips the class.

Binary-model decision rule: the original MAD-over-classes anomaly index needs
>= 3 classes and is meaningless with 2. Instead we use the **L1 mask-norm ratio**
between classes; the infected class is the one with the anomalously small mask.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.mlp import MLP

BENIGN_CLASS = 0
ATTACK_CLASS = 1


class NeuralCleanseDetector:
    """Reverse-engineer per-class triggers and score mask-norm asymmetry."""

    def __init__(
        self,
        model: MLP,
        holdout_X: np.ndarray,
        holdout_y: np.ndarray,
        *,
        device: Optional[torch.device] = None,
        steps: int = 500,
        lr: float = 0.1,
        init_lambda: float = 0.1,
        success_threshold: float = 0.99,
        lambda_up: float = 1.5,
        lambda_down: float = 1.5,
        lambda_update_every: int = 10,
        lambda_min: float = 1e-3,
        lambda_max: float = 1e2,
        bound_percentile: float = 99.0,
        delta_bounds: Optional[Tuple[float, float]] = None,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device).eval()
        self.X = np.asarray(holdout_X, dtype=np.float32)
        self.y = np.asarray(holdout_y).astype(int)
        self.D = self.X.shape[1]
        self.steps = int(steps)
        self.lr = float(lr)
        self.init_lambda = float(init_lambda)
        self.success_threshold = float(success_threshold)
        self.lambda_up = float(lambda_up)
        self.lambda_down = float(lambda_down)
        self.lambda_update_every = int(lambda_update_every)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.seed = int(seed)

        # Pattern range. For tabular (standardized) data we use per-feature
        # percentiles of the clean data; for images pass explicit ``delta_bounds``
        # (e.g. (0.0, 1.0)) so corner pixels — always ~0 in the data — can still
        # be driven to white when reverse-engineering a corner patch.
        if delta_bounds is not None:
            lo = np.full(self.D, float(delta_bounds[0]), dtype=np.float32)
            hi = np.full(self.D, float(delta_bounds[1]), dtype=np.float32)
        else:
            lo = np.percentile(self.X, 100 - bound_percentile, axis=0)
            hi = np.percentile(self.X, bound_percentile, axis=0)
        self.delta_lo = torch.tensor(lo, dtype=torch.float32, device=self.device)
        self.delta_hi = torch.tensor(hi, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_holdout(self) -> np.ndarray:
        """argmax predictions of the model on the holdout (for source-only NC)."""
        self.model.eval()
        X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        return self.model(X).argmax(1).cpu().numpy()

    # ------------------------------------------------------------------
    def reverse_engineer(self, target_class: int, *, source_only: bool = False,
                         n_source: Optional[int] = None,
                         preds: Optional[np.ndarray] = None,
                         sample_seed: Optional[int] = None) -> Dict[str, object]:
        """Optimize (mask, pattern) that flip non-target holdout rows to target.

        Default ("all-input"): source set = all rows with label != target.
        ``source_only=True`` (used by ``run_phase7e_nc_reeval``): source set = rows whose label != target
        AND that the model classifies CORRECTLY. ``n_source`` count-matches the
        source set to a fixed size (same N across candidate targets) so class
        prevalence / count cannot drive the mask size.

        Returns the smallest-L1 mask that still reaches the success threshold
        (or the final mask if the threshold is never reached).
        """
        torch.manual_seed(self.seed)
        # source rows: NOT already labelled as the target class
        sel = self.y != target_class
        if source_only:
            if preds is None:
                preds = self.predict_holdout()
            sel = sel & (preds == self.y)          # correctly-classified only
        idx = np.where(sel)[0]
        if n_source is not None and len(idx) > n_source:
            r = np.random.default_rng(self.seed if sample_seed is None else sample_seed)
            idx = np.sort(r.choice(idx, n_source, replace=False))
        n_used = int(len(idx))
        X = torch.tensor(self.X[idx], dtype=torch.float32, device=self.device)
        tgt = torch.full((len(X),), target_class, dtype=torch.long, device=self.device)

        # parameters: mask via sigmoid, pattern clamped to feature range
        m_raw = torch.zeros(self.D, device=self.device, requires_grad=True)
        d_raw = torch.zeros(self.D, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([m_raw, d_raw], lr=self.lr)

        lam = self.init_lambda
        best = {"l1": np.inf, "mask": None, "pattern": None, "success": 0.0}
        t0 = time.perf_counter()
        for step in range(self.steps):
            m = torch.sigmoid(m_raw)
            delta = torch.clamp(d_raw, self.delta_lo, self.delta_hi)
            x_mod = (1 - m) * X + m * delta
            logits = self.model(x_mod)
            ce = F.cross_entropy(logits, tgt)
            l1 = m.sum()
            loss = ce + lam * l1
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % self.lambda_update_every == 0 or step == self.steps - 1:
                with torch.no_grad():
                    success = (logits.argmax(1) == target_class).float().mean().item()
                    l1v = float(l1.item())
                    if success >= self.success_threshold:
                        lam = min(lam * self.lambda_up, self.lambda_max)
                        if l1v < best["l1"]:
                            best = {
                                "l1": l1v,
                                "mask": m.detach().cpu().numpy().copy(),
                                "pattern": delta.detach().cpu().numpy().copy(),
                                "success": success,
                            }
                    else:
                        lam = max(lam / self.lambda_down, self.lambda_min)
        elapsed = time.perf_counter() - t0

        if best["mask"] is None:  # never hit threshold -> use final mask
            with torch.no_grad():
                m = torch.sigmoid(m_raw)
                delta = torch.clamp(d_raw, self.delta_lo, self.delta_hi)
                x_mod = (1 - m) * X + m * delta
                success = (self.model(x_mod).argmax(1) == target_class).float().mean().item()
            best = {
                "l1": float(m.sum().item()),
                "mask": m.detach().cpu().numpy(),
                "pattern": delta.detach().cpu().numpy(),
                "success": success,
            }

        return {
            "target_class": target_class,
            "mask": best["mask"],
            "pattern": best["pattern"],
            "l1_norm": best["l1"],
            "reversal_success": best["success"],
            "n_opt_samples": n_used,
            "steps": self.steps,
            "time_s": elapsed,
        }

    # ------------------------------------------------------------------
    def run_multiclass(self, num_classes: int) -> Dict[str, object]:
        """Reverse-engineer every class; score with BOTH MAD and the L1 ratio.

        MAD (the original Neural Cleanse anomaly index) is statistically valid
        only with >= 3 classes, so this path is used for the multi-class image
        benchmark. The binary IDS model uses :meth:`run` (L1 ratio only).
        """
        res = {c: self.reverse_engineer(c) for c in range(num_classes)}
        norms = np.array([res[c]["l1_norm"] for c in range(num_classes)], dtype=float)
        mad = mad_anomaly_index(norms)
        infected_mad = [c for c in range(num_classes) if mad["anomaly_index"][c] > 2.0]
        infected_class = int(np.argmin(norms))
        l1_ratio = float(np.median(norms) / (norms.min() + 1e-8))
        return {
            "per_class": res,
            "l1_norms": {c: float(norms[c]) for c in range(num_classes)},
            "mad": mad,
            "infected_by_mad": infected_mad,
            "infected_class_min_norm": infected_class,
            "l1_ratio_median_over_min": l1_ratio,
            "total_time_s": sum(res[c]["time_s"] for c in range(num_classes)),
        }

    def run(self) -> Dict[str, object]:
        """Reverse-engineer both classes and compute the anomaly ratio."""
        res = {c: self.reverse_engineer(c) for c in (BENIGN_CLASS, ATTACK_CLASS)}
        l1 = {c: res[c]["l1_norm"] for c in res}
        infected = min(l1, key=l1.get)  # smallest mask = infected class
        eps = 1e-8
        # directed ratio: attack-mask / benign-mask (>1 => benign is the easy target)
        ratio_attack_over_benign = l1[ATTACK_CLASS] / (l1[BENIGN_CLASS] + eps)
        # infected-based ratio (always >= 1): other / infected
        other = ATTACK_CLASS if infected == BENIGN_CLASS else BENIGN_CLASS
        anomaly_ratio = l1[other] / (l1[infected] + eps)
        return {
            "per_class": res,
            "l1_norms": l1,
            "infected_class": int(infected),
            "anomaly_ratio": float(anomaly_ratio),
            "ratio_attack_over_benign": float(ratio_attack_over_benign),
            "total_time_s": sum(res[c]["time_s"] for c in res),
        }


# ---------------------------------------------------------------------------
# Helpers reused by the runner and by the evaluation scripts
# ---------------------------------------------------------------------------
def mad_anomaly_index(norms: np.ndarray) -> Dict[str, object]:
    """Original Neural Cleanse MAD anomaly index (needs >= 3 classes).

    A class whose mask L1-norm is abnormally *small* has a large positive index.
    Uses the consistent estimator sigma = 1.4826 * MAD. Flag threshold: index > 2.
    """
    norms = np.asarray(norms, dtype=float)
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    sigma = 1.4826 * mad + 1e-8
    index = (median - norms) / sigma  # >0 for smaller-than-median masks
    return {
        "median": median,
        "mad": mad,
        "anomaly_index": index.tolist(),
        "max_index": float(index.max()),
        "flag_threshold": 2.0,
    }


def topk_masked_features(mask: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` highest-mask features."""
    return np.argsort(mask)[::-1][:k]


def trigger_recovery(mask: np.ndarray, true_feature_indices, k: int) -> Dict[str, float]:
    """Precision/recall of recovering the true trigger features in top-k mask."""
    topk = set(topk_masked_features(mask, k).tolist())
    true = set(int(i) for i in true_feature_indices)
    hit = len(topk & true)
    precision = hit / max(k, 1)
    recall = hit / max(len(true), 1)
    return {
        "k": k,
        "hits": hit,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "recovered_indices": sorted(topk & true),
    }


@torch.no_grad()
def apply_reverse_trigger(
    model: MLP, X: np.ndarray, mask: np.ndarray, pattern: np.ndarray,
    device: torch.device, target_class: int = BENIGN_CLASS,
) -> float:
    """Stamp the reverse-engineered (mask, pattern) on X; return flip rate to target."""
    m = torch.tensor(mask, dtype=torch.float32, device=device)
    d = torch.tensor(pattern, dtype=torch.float32, device=device)
    x = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    x_mod = (1 - m) * x + m * d
    preds = model(x_mod).argmax(1)
    return float((preds == target_class).float().mean().item())
