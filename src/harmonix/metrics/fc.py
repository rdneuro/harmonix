"""Static functional connectivity and FC-similarity metrics."""
from __future__ import annotations

import numpy as np


def functional_connectivity(timeseries) -> np.ndarray:
    """Pearson functional connectivity matrix over nodes.

    Parameters
    ----------
    timeseries : np.ndarray, shape (n_nodes, n_time).

    Returns
    -------
    np.ndarray, shape (n_nodes, n_nodes).
    """
    x = np.asarray(timeseries, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"timeseries must be (n_nodes, n_time); got {x.shape}")
    return np.corrcoef(x)


def fc_similarity(fc_a, fc_b) -> float:
    """Pearson correlation between the upper triangles of two FC matrices."""
    A = np.asarray(fc_a, dtype=np.float64)
    B = np.asarray(fc_b, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"FC shape mismatch {A.shape} vs {B.shape}")
    iu = np.triu_indices(A.shape[0], k=1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def edge_vector(fc) -> np.ndarray:
    """Upper-triangular edge vector of an FC matrix."""
    A = np.asarray(fc, dtype=np.float64)
    return A[np.triu_indices(A.shape[0], k=1)]
