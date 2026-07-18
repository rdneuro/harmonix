"""Dynamic functional connectivity (FCD) and the FCD-KS fitting objective.

The FCD matrix is the correlation-of-correlations across sliding windows: compute
FC in each window, vectorize the upper triangle, and correlate windows pairwise.
The canonical Deco/Hansen model-fitting objective is the Kolmogorov-Smirnov
distance between the empirical and simulated FCD value distributions.
"""
from __future__ import annotations


import numpy as np


def _window_fc_vectors(timeseries, window: int, step: int) -> np.ndarray:
    x = np.asarray(timeseries, dtype=np.float64)
    n_nodes, n_time = x.shape
    iu = np.triu_indices(n_nodes, k=1)
    vecs = []
    for s in range(0, n_time - window + 1, step):
        fc = np.corrcoef(x[:, s:s + window])
        vecs.append(fc[iu])
    return np.asarray(vecs)                                    # (n_windows, n_edges)


def fcd_matrix(timeseries, window: int = 40, step: int = 3) -> np.ndarray:
    """Functional-connectivity-dynamics matrix (window-by-window FC correlation).

    Parameters
    ----------
    timeseries : np.ndarray, shape (n_nodes, n_time).
    window : int, sliding-window length in samples.
    step : int, window step in samples.

    Returns
    -------
    np.ndarray, shape (n_windows, n_windows).
    """
    vecs = _window_fc_vectors(timeseries, window, step)
    if vecs.shape[0] < 2:
        raise ValueError("not enough windows; reduce window or increase duration")
    return np.corrcoef(vecs)


def fcd_distribution(timeseries, window: int = 40, step: int = 3) -> np.ndarray:
    """Off-diagonal FCD values as a 1-D distribution."""
    F = fcd_matrix(timeseries, window, step)
    return F[np.triu_indices(F.shape[0], k=1)]


def fcd_ks_distance(timeseries_a, timeseries_b, window: int = 40, step: int = 3
                    ) -> float:
    """Kolmogorov-Smirnov distance between two FCD value distributions.

    The standard whole-brain model-fitting objective (lower = better match).
    """
    from scipy.stats import ks_2samp
    da = fcd_distribution(timeseries_a, window, step)
    db = fcd_distribution(timeseries_b, window, step)
    return float(ks_2samp(da, db).statistic)
