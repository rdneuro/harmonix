"""Working-point finder: locate the coupling G nearest criticality.

Two routes, matching the design report: (1) the analytic/spectral route places the
working point where the spectral abscissa approaches 0 from below (cheapest, no
simulation); (2) the empirical route maximizes a fluctuation metric (susceptibility
or metastability) or minimizes the FCD-KS distance to an empirical target over a G
grid. Both express the same idea -- the model sits near-critical where it best
matches resting brain dynamics.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

from .spectral import spectral_abscissa


def working_point_spectral(model_factory: Callable, couplings: Sequence[float],
                           target_abscissa: float = -1e-3) -> Tuple[float, np.ndarray]:
    """Find the coupling whose spectral abscissa is closest to ``target_abscissa``.

    Parameters
    ----------
    model_factory : callable ``G -> ModalModel``.
    couplings : sequence of candidate global-coupling values.
    target_abscissa : float
        Desired near-critical abscissa (slightly below 0). Default -1e-3.

    Returns
    -------
    best_G : float
    abscissas : np.ndarray, the spectral abscissa at each candidate G.
    """
    couplings = np.asarray(list(couplings), dtype=np.float64)
    absc = np.array([spectral_abscissa(model_factory(float(G))) for G in couplings])
    stable = absc < 0
    if not np.any(stable):
        return float(couplings[np.argmin(absc)]), absc
    idx = np.where(stable)[0]
    best = idx[np.argmin(np.abs(absc[idx] - target_abscissa))]
    return float(couplings[best]), absc


def working_point_metric(couplings: Sequence[float], metric_values: Sequence[float],
                         mode: str = "max") -> float:
    """Pick the coupling optimizing an empirical metric over a grid.

    Parameters
    ----------
    couplings : sequence of G values.
    metric_values : sequence of metric values aligned to ``couplings``
        (e.g. susceptibility/metastability for ``mode='max'``, or FCD-KS distance
        for ``mode='min'``).
    mode : {'max', 'min'}.

    Returns
    -------
    float, the optimal coupling.
    """
    couplings = np.asarray(list(couplings), dtype=np.float64)
    vals = np.asarray(list(metric_values), dtype=np.float64)
    idx = int(np.argmax(vals)) if mode == "max" else int(np.argmin(vals))
    return float(couplings[idx])
