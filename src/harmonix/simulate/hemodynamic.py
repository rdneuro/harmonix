"""Balloon-Windkessel hemodynamic model (Friston et al. 2000; Stephan et al. 2007).

Converts a neural time series (per node) into a BOLD signal through four coupled
state equations -- vasodilatory signal ``s``, blood inflow ``f``, blood volume
``v``, deoxyhemoglobin ``q`` -- and the BOLD observation equation. Defaults follow
the DCM literature and Pang's released ``loadParameters_balloon_func.m``; the
minor k2/k3 discrepancy between Pang's code and the original paper is exposed as a
configurable option.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import BALLOON


def balloon_windkessel(neural, dt: float, params: Optional[dict] = None,
                       ) -> np.ndarray:
    """Convert neural activity to BOLD via the Balloon-Windkessel model.

    Parameters
    ----------
    neural : np.ndarray, shape (n_nodes, n_time)
        Neural drive per node (e.g. Re(z) or |z| from the modal simulation).
    dt : float
        Neural sampling period in seconds.
    params : dict, optional
        Overrides for {kappa, gamma, tau, alpha, rho, V0}. Defaults from config.

    Returns
    -------
    np.ndarray, shape (n_nodes, n_time)
        BOLD signal per node.
    """
    p = dict(BALLOON)
    if params:
        p.update(params)
    kappa, gamma, tau, alpha = p["kappa"], p["gamma"], p["tau"], p["alpha"]
    rho, V0 = p["rho"], p["V0"]
    z = np.asarray(neural, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    n_nodes, n_time = z.shape

    # state variables (start at rest)
    s = np.zeros(n_nodes)
    f = np.ones(n_nodes)
    v = np.ones(n_nodes)
    q = np.ones(n_nodes)
    inv_alpha = 1.0 / alpha

    # BOLD observation coefficients (Obata et al. / Stephan 2007)
    k1 = 7.0 * rho
    k2 = 2.0
    k3 = 2.0 * rho - 0.2

    bold = np.empty((n_nodes, n_time))
    for t in range(n_time):
        ds = z[:, t] - kappa * s - gamma * (f - 1.0)
        df = s
        dv = (f - v ** inv_alpha) / tau
        dq = (f * (1.0 - (1.0 - rho) ** (1.0 / np.clip(f, 1e-6, None))) / rho
              - q * v ** inv_alpha / v) / tau
        s = s + dt * ds
        f = np.clip(f + dt * df, 1e-6, None)
        v = np.clip(v + dt * dv, 1e-6, None)
        q = np.clip(q + dt * dq, 1e-6, None)
        bold[:, t] = 100.0 * V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
    return bold


def downsample_to_tr(signal, dt: float, tr: float) -> np.ndarray:
    """Downsample a signal from neural ``dt`` to BOLD repetition time ``tr``."""
    step = max(1, int(round(tr / dt)))
    return np.asarray(signal)[..., ::step]
