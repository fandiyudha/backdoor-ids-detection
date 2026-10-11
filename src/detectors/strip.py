"""STRIP backdoor detector, adapted for tabular network-IDS data.

STRIP (Gao et al., 2019) is an *input-level*, run-time detector. The idea:
perturb a suspicious input many times by superimposing it with random CLEAN
samples and measure the average entropy of the model's prediction over the
perturbations. A **backdoored** input keeps producing the same (target) class
regardless of the perturbation — the trigger dominates — so its average entropy
is **low**. A clean input's prediction varies with the perturbation, giving
**high** entropy.

Tabular adaptation: images have a natural
superposition (pixel averaging); tabular feature vectors do not. We implement
and compare two perturbation operators:

* ``blend``  — convex combination ``x_mix = alpha*x + (1-alpha)*x_clean``.
  Dilutes the trigger values; too much dilution can wash out the signal.
* ``replace`` — overwrite a random subset (proportion ``replace_prop``) of the
  features of ``x`` with the same features from a random clean sample. May
  remove the trigger features entirely (if they fall in the replaced subset),
  or leave them fully intact — a coarser, higher-variance operator.

Threat model (strict): the defender holds only the trained model and a small
CLEAN holdout. Thresholds are calibrated **only** on clean data; triggered
inputs are never used for calibration. The defender does not know which features
carry the trigger or their values.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.models.mlp import MLP


def prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each row of a probability matrix ``[M, C]``."""
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


class StripDetector:
    """Run-time STRIP detector over a single (possibly backdoored) model.

    Parameters
    ----------
    model:
        The model under test (already loaded, on CPU).
    clean_pool:
        Clean feature vectors (the defender's holdout) used as the superposition
        partners. Also the only data used to calibrate thresholds.
    variant:
        ``"blend"`` or ``"replace"`` (tabular), or ``"superpose"`` (canonical
        image STRIP: clipped sum of two inputs).
    N:
        Number of perturbations averaged per input.
    alpha:
        Blend weight on the original input (``blend`` variant).
    replace_prop:
        Fraction of features overwritten from a clean partner (``replace``).
    """

    def __init__(
        self,
        model: MLP,
        clean_pool: np.ndarray,
        *,
        variant: str = "blend",
        N: int = 100,
        alpha: float = 0.7,
        replace_prop: float = 0.3,
        device: Optional[torch.device] = None,
        seed: int = 42,
        chunk: int = 128,
    ) -> None:
        if variant not in {"blend", "replace", "superpose", "marginal"}:
            raise ValueError(
                f"variant must be 'blend', 'replace', 'superpose', or 'marginal', "
                f"got {variant!r}"
            )
        self.model = model
        self.pool = np.asarray(clean_pool, dtype=np.float32)
        self.variant = variant
        self.N = int(N)
        self.alpha = float(alpha)
        self.replace_prop = float(replace_prop)
        self.device = device or torch.device("cpu")
        self.chunk = int(chunk)
        self.rng = np.random.default_rng(seed)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def _perturb_block(self, xb: np.ndarray) -> np.ndarray:
        """Return perturbed tensor ``[B, N, D]`` for a block of inputs ``[B, D]``."""
        B, D = xb.shape
        idx = self.rng.integers(0, len(self.pool), size=(B, self.N))
        partners = self.pool[idx]  # [B, N, D]
        xexp = np.repeat(xb[:, None, :], self.N, axis=1)  # [B, N, D]

        if self.variant == "blend":
            return self.alpha * xexp + (1.0 - self.alpha) * partners

        if self.variant == "superpose":
            # canonical image STRIP: overlay two inputs (clipped sum to [0,1])
            return np.clip(xexp + partners, 0.0, 1.0)

        # replace / marginal: overwrite k random features per perturbation.
        k = max(1, int(round(self.replace_prop * D)))
        r = self.rng.random((B, self.N, D))
        sel = np.argsort(r, axis=2)[:, :, :k]  # k smallest -> k random features
        mask = np.zeros((B, self.N, D), dtype=bool)
        np.put_along_axis(mask, sel, True, axis=2)

        if self.variant == "marginal":
            # validity-preserving: draw each replaced feature INDEPENDENTLY from its
            # empirical marginal (a random pool row per feature). Every value is a
            # REAL observed value, so per-feature bounds/integrality hold — unlike
            # convex blending, which interpolates to out-of-support values (e.g.
            # fractional packet/flag counts) and can break relations such as
            # min <= mean <= max across related features.
            pidx = self.rng.integers(0, len(self.pool), size=(B, self.N, D))
            cols = np.broadcast_to(np.arange(D), (B, self.N, D))
            marg = self.pool[pidx, cols]  # [B, N, D], each feature from its own row
            return np.where(mask, marg, xexp)

        return np.where(mask, partners, xexp)  # replace: one partner's real subset

    @torch.no_grad()
    def score(self, X: np.ndarray) -> Tuple[np.ndarray, int]:
        """Average prediction entropy per input; also returns #forward-pass rows.

        Low entropy => the model's decision is insensitive to perturbation =>
        likely triggered.
        """
        X = np.asarray(X, dtype=np.float32)
        M, D = X.shape
        ent = np.empty(M, dtype=np.float64)
        n_forward = 0
        for s in range(0, M, self.chunk):
            xb = X[s : s + self.chunk]
            B = len(xb)
            pert = self._perturb_block(xb)  # [B, N, D]
            flat = pert.reshape(B * self.N, D).astype(np.float32)
            logits = self.model(torch.from_numpy(flat).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            n_forward += B * self.N
            H = prediction_entropy(probs).reshape(B, self.N)
            ent[s : s + B] = H.mean(axis=1)
        return ent, n_forward

    # ------------------------------------------------------------------
    def calibrate(
        self, holdout_X: np.ndarray, fpr_targets
    ) -> Dict[float, float]:
        """Set entropy thresholds from CLEAN holdout only.

        For a target FPR ``f``, the threshold is the ``f`` quantile (lower tail)
        of the clean-holdout entropy: flagging ``entropy < threshold`` marks
        about a fraction ``f`` of clean inputs. Returns ``{fpr: threshold}``.
        """
        clean_entropy, _ = self.score(holdout_X)
        return {
            float(f): float(np.quantile(clean_entropy, f)) for f in fpr_targets
        }

    @staticmethod
    def flag(entropy: np.ndarray, threshold: float) -> np.ndarray:
        """Flag inputs whose entropy is below the threshold (suspected trigger)."""
        return entropy < threshold
