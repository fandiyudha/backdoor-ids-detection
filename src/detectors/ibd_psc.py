"""IBD-PSC input-level backdoor detection, adapted to tabular IDS models.

Reference: L. Hou, R. Feng, Z. Hua, W. Luo, L. Y. Zhang, Y. Li, "IBD-PSC:
Input-level backdoor detection via parameter-oriented scaling consistency,"
ICML 2024, PMLR 235:18992-19022 (arXiv:2405.09786). Reference code:
BackdoorBox, ``core/defenses/IBD_PSC.py``. This is an independent tabular
re-implementation, not a copy of that code.

Idea: amplifying the batch-normalization parameters of the deepest layers
destroys the predictions on benign inputs, while poisoned (triggered) inputs keep
their predicted label with high confidence ("scaling consistency"). The PSC score
of an input is the mean confidence, across the amplified models, of the label the
original model predicts. A high PSC suggests a triggered input.

Procedure (as in the reference):

1. Amplification: multiply ``weight`` (gamma) and ``bias`` (beta) of the deepest
   ``m`` BN layers by ``omega``.
2. Adaptive start: increase ``m`` from 1 until the amplified model's error rate
   (against the true labels) on a small clean set exceeds ``xi``; call it ``k``.
3. Amplified models: model ``i`` (i = 1..n) scales the deepest ``k + i - 1`` BN
   layers. (BackdoorBox scales ``k + i`` layers, one more; the paper's
   description of ``k + i - 1`` is followed here.)
4. PSC(x) = mean over amplified models of softmax_{y_hat(x)}, with ``y_hat`` the
   original model's prediction; flag if PSC >= T.

Tabular adaptation for shallow models (documented in the v1.1 summaries):

* The IDS models have only 3 BN layers (``BatchNorm1d``; the reference counts
  ``BatchNorm2d``), so fewer than ``n`` layer counts may exist above ``k``. The
  model set is then completed by also varying ``omega`` over ``extra_omegas``:
  candidates are ordered by omega first, then by layer count (``k..L``), and the
  first ``n`` are used. If fewer than ``n`` candidates exist, all are used.
* If the error rate never exceeds ``xi`` even with all ``L`` BN layers scaled by
  ``omega``, ``k`` is set to ``L`` and the fact is recorded (``xi_reached``).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm


def bn_layers_deepest_first(model: torch.nn.Module) -> List[str]:
    """Names of the BN modules, deepest (last in forward order) first."""
    names = [n for n, m in model.named_modules() if isinstance(m, _BatchNorm)]
    return names[::-1]


def amplify(model: torch.nn.Module, layer_names: Sequence[str], omega: float) -> torch.nn.Module:
    """Copy of ``model`` with gamma and beta of the named BN layers multiplied by omega."""
    m = copy.deepcopy(model).eval()
    mods = dict(m.named_modules())
    with torch.no_grad():
        for n in layer_names:
            mods[n].weight.mul_(omega)
            mods[n].bias.mul_(omega)
    return m


@torch.no_grad()
def _probs(model, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    """Softmax in float64: float32 saturates confident predictions at exactly 1.0,
    which would create ties in the PSC score."""
    out = []
    for i in range(0, len(X), batch):
        logits = model(torch.from_numpy(np.asarray(X[i:i + batch], np.float32)))
        out.append(torch.softmax(logits.double(), 1))
    return torch.cat(out).numpy()


class IBDPSC:
    """Parameter-oriented scaling consistency detector for one model."""

    def __init__(self, model: torch.nn.Module, select_X: np.ndarray, select_y: np.ndarray, *,
                 omega: float = 1.5, xi: float = 0.6, n: int = 5,
                 extra_omegas: Sequence[float] = (2.0, 2.5),
                 fixed_configs: Sequence[Tuple[int, float]] | None = None) -> None:
        """``fixed_configs`` (list of ``(n_layers, omega)``) skips the adaptive
        selection and uses exactly these amplified models (see
        :func:`select_omega_all_layers` for the omega-search variant)."""
        self.model = model.eval()
        self.bn = bn_layers_deepest_first(model)
        if not self.bn:
            raise ValueError("IBD-PSC needs batch-normalization layers")
        self.omega, self.xi, self.n = float(omega), float(xi), int(n)
        if fixed_configs is None:
            self.k, self.xi_reached, self.selection_errors = self._select_k(select_X, select_y)
            self.configs = self._configs(extra_omegas)
        else:
            self.k, self.xi_reached, self.selection_errors = None, None, {}
            self.configs = [(int(m), float(w)) for m, w in fixed_configs]
        self.amplified = [amplify(model, self.bn[:m], w) for m, w in self.configs]

    def _select_k(self, X, y) -> Tuple[int, bool, Dict[int, float]]:
        y = np.asarray(y).astype(int)
        errs = {}
        for m in range(1, len(self.bn) + 1):
            pred = _probs(amplify(self.model, self.bn[:m], self.omega), X).argmax(1)
            errs[m] = float(np.mean(pred != y))
            if errs[m] > self.xi:
                return m, True, errs
        return len(self.bn), False, errs

    def _configs(self, extra_omegas) -> List[Tuple[int, float]]:
        L = len(self.bn)
        cands = [(m, w) for w in (self.omega, *extra_omegas) for m in range(self.k, L + 1)]
        return cands[: self.n]

    def describe(self) -> dict:
        return {"bn_layers_deepest_first": self.bn, "k": self.k, "xi_reached": self.xi_reached,
                "selection_error_by_layers": {str(m): e for m, e in self.selection_errors.items()},
                "amplified_models": [{"n_layers": m, "omega": w} for m, w in self.configs],
                "n_models": len(self.configs)}

    def score(self, X: np.ndarray) -> np.ndarray:
        """PSC score per input (higher = more consistent = more suspicious)."""
        y_hat = _probs(self.model, X).argmax(1)
        idx = np.arange(len(X))
        psc = np.zeros(len(X), np.float64)
        for am in self.amplified:
            psc += _probs(am, X)[idx, y_hat]
        return psc / len(self.amplified)


def select_omega_all_layers(model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                            grid: Sequence[float], xi: float) -> Tuple[float | None, Dict[str, float]]:
    """Omega-search variant of the adaptive start (all BN layers amplified).

    Returns the first ``omega`` in ``grid`` whose amplified model has a clean error
    rate (against the true labels) above ``xi``, or ``None`` if no grid value does,
    together with the error rate of every grid value tried.
    """
    bn = bn_layers_deepest_first(model)
    y = np.asarray(y).astype(int)
    errs = {}
    for w in grid:
        errs[str(w)] = float(np.mean(_probs(amplify(model, bn, w), X).argmax(1) != y))
        if errs[str(w)] > xi:
            return float(w), errs
    return None, errs
