"""Phase-based dynamic metrics: Kuramoto synchrony, metastability, LEiDA.

Uses the analytic (Hilbert) phase of each node's activity to compute the Kuramoto
order parameter, metastability (its temporal variance), and the LEiDA leading
eigenvector of the instantaneous phase-locking matrix (Cabral et al. 2017).
"""
from __future__ import annotations


import numpy as np


def instantaneous_phase(timeseries) -> np.ndarray:
    """Analytic (Hilbert) phase of each node's activity."""
    from scipy.signal import hilbert
    x = np.asarray(timeseries, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    return np.angle(hilbert(x, axis=1))


def kuramoto_order(timeseries) -> np.ndarray:
    """Kuramoto order parameter R(t) from node activity (via Hilbert phase)."""
    phi = instantaneous_phase(timeseries)
    return np.abs(np.mean(np.exp(1j * phi), axis=0))


def synchrony(timeseries) -> float:
    """Mean Kuramoto order parameter over time (global synchrony)."""
    return float(np.mean(kuramoto_order(timeseries)))


def metastability(timeseries) -> float:
    """Metastability: temporal standard deviation of the Kuramoto order parameter."""
    return float(np.std(kuramoto_order(timeseries)))


def leida_leading_eigenvector(timeseries) -> np.ndarray:
    """LEiDA leading eigenvectors of the instantaneous phase-locking matrix.

    Parameters
    ----------
    timeseries : np.ndarray, shape (n_nodes, n_time).

    Returns
    -------
    np.ndarray, shape (n_time, n_nodes)
        The leading eigenvector of the phase-locking matrix at each timepoint
        (sign-fixed so the majority of entries are negative, per LEiDA convention).
    """
    phi = instantaneous_phase(timeseries)                     # (n_nodes, n_time)
    n_nodes, n_time = phi.shape
    out = np.empty((n_time, n_nodes))
    for t in range(n_time):
        dphi = phi[:, t][:, None] - phi[:, t][None, :]
        plm = np.cos(dphi)                                    # phase-locking matrix
        w, V = np.linalg.eigh(plm)
        v1 = V[:, -1]
        if np.sum(v1 > 0) > n_nodes / 2:                     # sign convention
            v1 = -v1
        out[t] = v1
    return out
