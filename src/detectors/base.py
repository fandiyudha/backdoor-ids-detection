"""Minimal detector interfaces and thin wrappers around the existing detectors.

Two interfaces:

* :class:`ModelLevelDetector` - ``score_model(model, holdout) -> dict``: screens a
  whole model with a clean holdout ``(X, y)``, before deployment.
* :class:`InputLevelDetector` - ``score_inputs(model, X) -> np.ndarray``: scores
  individual inputs at runtime (higher = more suspicious is NOT assumed; each
  wrapper documents its direction in ``higher_is_suspicious``).

The wrappers only forward to the existing implementations
(``NeuralCleanseDetector``, ``MaxMarginDetector``, ``StripDetector``, ``IBDPSC``)
with the settings used in the paper and the v1.1 experiments, so their outputs
are identical to those of the experiment runners (checked by
``tests/check_detector_wrappers.py`` against the committed metrics).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from src.detectors.ibd_psc import IBDPSC
from src.detectors.mmbd import MaxMarginDetector, box_bounds
from src.detectors.neural_cleanse import NeuralCleanseDetector
from src.detectors.strip import StripDetector

Holdout = Tuple[np.ndarray, np.ndarray]


class ModelLevelDetector(ABC):
    """Screens a model with a clean holdout ``(X, y)``."""

    name: str = "model-level detector"

    @abstractmethod
    def score_model(self, model: torch.nn.Module, holdout: Holdout) -> dict:
        """Return per-class statistics and a scalar ``statistic`` (larger = more suspicious)."""


class InputLevelDetector(ABC):
    """Scores individual inputs of a model at runtime."""

    name: str = "input-level detector"
    higher_is_suspicious: bool = True

    @abstractmethod
    def score_inputs(self, model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
        """Return one score per row of ``X``."""


# ---------------------------------------------------------------------------
class NeuralCleanseScreen(ModelLevelDetector):
    """Target-excluded, count-matched Neural Cleanse (the paper's phase7e protocol).

    ``statistic`` is the mask-norm ratio max_c L1 / min_c L1. ``n_source`` is the
    count-matched source-set size; the paper derives it from the first clean
    reference model with :meth:`count_match_n`.
    """

    name = "Neural Cleanse (target-excluded, count-matched)"

    def __init__(self, n_source: int, *, steps: int = 500, lr: float = 0.1, seed: int = 42,
                 success_threshold: float = 0.99) -> None:
        self.n_source, self.steps, self.lr, self.seed = int(n_source), int(steps), float(lr), int(seed)
        self.success_threshold = float(success_threshold)

    def _det(self, model, holdout):
        X, y = holdout
        return NeuralCleanseDetector(model, X, y, device=torch.device("cpu"), steps=self.steps,
                                     lr=self.lr, seed=self.seed,
                                     success_threshold=self.success_threshold)

    @staticmethod
    def count_match_n(model, holdout: Holdout, *, seed: int = 42) -> int:
        X, y = holdout
        p = NeuralCleanseDetector(model, X, y, device=torch.device("cpu"), seed=seed).predict_holdout()
        y = np.asarray(y).astype(int)
        return min(int(((y == 1) & (p == y)).sum()), int(((y == 0) & (p == y)).sum()))

    def score_model(self, model, holdout):
        det = self._det(model, holdout)
        preds = det.predict_holdout()
        r = {c: det.reverse_engineer(c, source_only=True, n_source=self.n_source, preds=preds)
             for c in (0, 1)}
        l1 = {str(c): float(r[c]["l1_norm"]) for c in (0, 1)}
        lo, hi = min(l1.values()), max(l1.values())
        return {"detector": self.name, "per_class": {"mask_l1": l1},
                "statistic": hi / (lo + 1e-9), "statistic_name": "mask-norm ratio max/min",
                "smallest_mask_class": int(min(l1, key=l1.get)),
                "reversal_success": {str(c): float(r[c]["reversal_success"]) for c in (0, 1)},
                "n_source": self.n_source}


class MMBDScreen(ModelLevelDetector):
    """MM-BD maximum margins over a box from the clean holdout (v1.1 Experiment B).

    Returns the raw per-class margins r_c; the binary reference-relative
    statistic needs clean reference models and is computed by the caller
    (e.g. ``src.gate.screen_model``). ``statistic`` here is max_c r_c.
    """

    name = "MM-BD (maximum margin)"

    def __init__(self, *, box: str = "minmax", lo_pct: float = 1.0, hi_pct: float = 99.0,
                 n_starts: int = 50, steps: int = 500, lr: float = 0.02, seed: int = 42,
                 holdout_starts: int = 50) -> None:
        self.box, self.lo_pct, self.hi_pct = box, float(lo_pct), float(hi_pct)
        self.n_starts, self.steps, self.lr, self.seed = int(n_starts), int(steps), float(lr), int(seed)
        self.holdout_starts = int(holdout_starts)

    def score_model(self, model, holdout):
        X, _ = holdout
        lo, hi = box_bounds(X, self.box, self.lo_pct, self.hi_pct)
        r = MaxMarginDetector(model, lo, hi, n_starts=self.n_starts, steps=self.steps, lr=self.lr,
                              seed=self.seed, seed_points=X, n_seed_starts=self.holdout_starts).score()
        return {"detector": self.name, "per_class": {"margin": r["margins"]},
                "statistic": max(r["margins"].values()), "statistic_name": "max_c r_c",
                "box": self.box, "time_s": r["time_s"]}


class StripRuntime(InputLevelDetector):
    """STRIP, the paper's final variant (blend, alpha 0.7, N = 100). Low entropy = suspicious.

    The perturbation RNG is seeded once per wrapper call, so scoring the same
    inputs in the same order as the experiment runners gives identical scores.
    """

    name = "STRIP (blend)"
    higher_is_suspicious = False

    def __init__(self, clean_pool: np.ndarray, *, alpha: float = 0.7, N: int = 100,
                 variant: str = "blend", seed: int = 42) -> None:
        self.pool, self.alpha, self.N, self.variant, self.seed = clean_pool, alpha, N, variant, seed

    def detector(self, model) -> StripDetector:
        return StripDetector(model, self.pool, variant=self.variant, alpha=self.alpha, N=self.N,
                             device=torch.device("cpu"), seed=self.seed, chunk=128)

    def score_inputs(self, model, X):
        return self.detector(model).score(X)[0]


class IBDPSCRuntime(InputLevelDetector):
    """IBD-PSC as adapted in v1.1 Experiment C. High PSC = suspicious."""

    name = "IBD-PSC"
    higher_is_suspicious = True

    def __init__(self, select_X: np.ndarray, select_y: np.ndarray, *, omega: float = 1.5,
                 xi: float = 0.6, n: int = 5, extra_omegas: Sequence[float] = (2.0, 2.5),
                 fixed_configs: Optional[Sequence[Tuple[int, float]]] = None) -> None:
        self.select_X, self.select_y = select_X, select_y
        self.kw = dict(omega=omega, xi=xi, n=n, extra_omegas=extra_omegas, fixed_configs=fixed_configs)

    def detector(self, model) -> IBDPSC:
        return IBDPSC(model, self.select_X, self.select_y, **self.kw)

    def score_inputs(self, model, X):
        return self.detector(model).score(X)
