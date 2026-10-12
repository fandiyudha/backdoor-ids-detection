"""MM-BD maximum-margin backdoor statistic, adapted to tabular flow features.

Reference: H. Wang, Z. Xiang, D. J. Miller, G. Kesidis, "MM-BD: Post-training
detection of backdoor attacks with arbitrary backdoor pattern types using a
maximum margin statistic," IEEE S&P 2024, doi:10.1109/SP54263.2024.00015.

MM-BD is a *model-level* detector that needs no clean data for the statistic
itself. For each class ``c`` it computes the maximum achievable classification
margin over the input domain ``X``:

    r_c = max_{x in X} [ g_c(x) - max_{k != c} g_k(x) ]

where ``g`` are the pre-softmax logits. A backdoor creates a shortcut to its
target class that yields an abnormally large margin for that class.

Tabular adaptation (documented in the v1.1 summaries):

* ``X`` is a per-feature box in the standardized feature space. Its bounds are
  set from the defender's clean holdout (min/max, or a percentile range), never
  from training data. The bounds are passed in explicitly and recorded.
* The maximization runs by projected gradient ascent in normalized box
  coordinates ``u in [0, 1]^D`` with ``x = lo + u (hi - lo)``: Adam on ``u``, then
  projection (clamp) onto ``[0, 1]^D`` after every step. Optimizing in normalized
  coordinates lets one learning rate serve features whose box widths differ by
  orders of magnitude. Multiple random starts are drawn uniformly in the box, and
  the best margin seen at any start and step is kept.
* The original decision rule (a null distribution fitted to the K-1 non-maximal
  statistics and an order-statistic p-value) needs K >= 3 classes. With K = 2 a
  single non-maximal statistic remains, so the rule is not identifiable; the
  v1.1 experiments instead use the paper's paired reference-relative rule (see
  ``src.eval.run_v11_mmbd``).
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch


class MaxMarginDetector:
    """Per-class maximum-margin statistic over a box-constrained input domain."""

    def __init__(self, model: torch.nn.Module, lo: np.ndarray, hi: np.ndarray, *,
                 n_starts: int = 50, steps: int = 500, lr: float = 0.02, seed: int = 42,
                 num_classes: int = 2, seed_points: np.ndarray | None = None,
                 n_seed_starts: int = 0) -> None:
        self.model = model.eval()
        self.lo = torch.tensor(np.asarray(lo, np.float32))
        self.width = torch.tensor(np.asarray(hi, np.float32) - np.asarray(lo, np.float32))
        if (self.width < 0).any():
            raise ValueError("box upper bound below lower bound")
        self.n_starts, self.steps, self.lr = int(n_starts), int(steps), float(lr)
        self.seed, self.num_classes = int(seed), int(num_classes)
        # Optional data-seeded starts: for class c, the n_seed_starts points of
        # ``seed_points`` (e.g. the clean holdout, which lies inside the box) with
        # the largest class-c margin. Random starts in a very loose box can stall
        # on a dead-ReLU plateau (constant logits, zero gradient) and return a
        # value below a known feasible point; seeded starts prevent that.
        self.seed_points = None if seed_points is None else np.asarray(seed_points, np.float32)
        self.n_seed_starts = int(n_seed_starts)

    def _seed_starts(self, c: int) -> torch.Tensor:
        X = torch.from_numpy(self.seed_points)
        with torch.no_grad():
            m = self._margin(self.model(X), c)
        top = torch.argsort(m, descending=True)[: self.n_seed_starts]
        w = torch.where(self.width > 0, self.width, torch.ones_like(self.width))
        return ((X[top] - self.lo) / w).clamp(0.0, 1.0)

    def _margin(self, logits: torch.Tensor, c: int) -> torch.Tensor:
        others = torch.cat([logits[:, :c], logits[:, c + 1:]], dim=1)
        return logits[:, c] - others.max(dim=1).values

    def class_margin(self, c: int) -> Dict[str, object]:
        """Maximize the margin of class ``c``; return the best value over starts and steps."""
        g = torch.Generator().manual_seed(self.seed)
        u = torch.rand(self.n_starts, self.lo.numel(), generator=g)
        n_random = u.shape[0]
        if self.seed_points is not None and self.n_seed_starts > 0:
            u = torch.cat([u, self._seed_starts(c)])
        u = u.requires_grad_(True)
        opt = torch.optim.Adam([u], lr=self.lr)
        best = torch.full((u.shape[0],), -np.inf)
        for step in range(self.steps + 1):
            m = self._margin(self.model(self.lo + u * self.width), c)
            best = torch.maximum(best, m.detach())
            if step == self.steps:
                break
            (grad,) = torch.autograd.grad(m.sum(), u)
            u.grad = -grad                      # ascent
            opt.step()
            with torch.no_grad():
                u.clamp_(0.0, 1.0)              # projection onto the box
        b = best.numpy().astype(float)
        r = float(b.max())
        tol = 0.01 * abs(r)
        return {"r": r, "per_start_best": b.tolist(),
                "frac_starts_within_1pct": float(np.mean(b >= r - tol)),
                "best_from": "random" if int(b.argmax()) < n_random else "seeded",
                "best_random": float(b[:n_random].max())}

    def score(self) -> Dict[str, object]:
        """Margins of every class (keys are class indices as strings) and timing."""
        t0 = time.perf_counter()
        per = {str(c): self.class_margin(c) for c in range(self.num_classes)}
        return {"margins": {c: v["r"] for c, v in per.items()},
                "convergence": {c: v["frac_starts_within_1pct"] for c, v in per.items()},
                "best_from": {c: v["best_from"] for c, v in per.items()},
                "best_random": {c: v["best_random"] for c, v in per.items()},
                "per_start_best": {c: v["per_start_best"] for c, v in per.items()},
                "time_s": time.perf_counter() - t0,
                "n_starts": self.n_starts, "steps": self.steps, "lr": self.lr}


def box_bounds(holdout_X: np.ndarray, kind: str, lo_pct: float = 1.0,
               hi_pct: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature box from the clean holdout: ``minmax`` or ``percentile``."""
    X = np.asarray(holdout_X, np.float32)
    if kind == "minmax":
        return X.min(axis=0), X.max(axis=0)
    if kind == "percentile":
        return (np.percentile(X, lo_pct, axis=0).astype(np.float32),
                np.percentile(X, hi_pct, axis=0).astype(np.float32))
    raise ValueError(f"unknown box kind {kind!r}")
