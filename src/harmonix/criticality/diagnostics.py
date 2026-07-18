"""Classical criticality diagnostics computed from simulated activity.

These cross-check the spectral criticality measures on actual trajectories:
critical slowing down (rising lag-1 autocorrelation and variance near the
transition; Scheffer et al. 2009), susceptibility (variance of the global field),
metastability (temporal variance of the Kuramoto order parameter), and the
branching parameter from activity events.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def lag1_autocorrelation(signal) -> float:
    """Mean lag-1 autocorrelation across nodes (critical slowing down indicator)."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x = x - x.mean(axis=1, keepdims=True)
    num = np.sum(x[:, :-1] * x[:, 1:], axis=1)
    den = np.sum(x * x, axis=1)
    return float(np.mean(num / np.clip(den, 1e-30, None)))


def susceptibility(signal) -> float:
    """Susceptibility: variance of the spatially-averaged field over time.

    Peaks near criticality where fluctuations are maximally correlated.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    global_field = x.mean(axis=0)                             # (n_time,)
    return float(np.var(global_field))


def variance_diagnostic(signal) -> float:
    """Mean temporal variance across nodes (rises before some transitions)."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    return float(np.mean(np.var(x, axis=1)))


def kuramoto_order_parameter(phases) -> np.ndarray:
    """Instantaneous Kuramoto order parameter R(t) from node phases.

    Parameters
    ----------
    phases : np.ndarray, shape (n_nodes, n_time)
        Instantaneous phases (radians).

    Returns
    -------
    np.ndarray, shape (n_time,), the order parameter R(t) in [0, 1].
    """
    phi = np.asarray(phases, dtype=np.float64)
    return np.abs(np.mean(np.exp(1j * phi), axis=0))


def metastability(phases) -> float:
    """Metastability: standard deviation over time of the Kuramoto order parameter."""
    return float(np.std(kuramoto_order_parameter(phases)))


def branching_parameter(signal, threshold: Optional[float] = None) -> float:
    """Branching parameter from thresholded activity (sigma ~ 1 at criticality).

    Estimated as the mean ratio of active-node count at t+1 to t during events.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if threshold is None:
        threshold = x.mean() + x.std()
    active = (x > threshold).sum(axis=0).astype(np.float64)   # (n_time,)
    ancestors = active[:-1]
    descendants = active[1:]
    mask = ancestors > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(descendants[mask] / ancestors[mask]))

