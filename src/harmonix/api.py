"""High-level convenience API with best-value defaults.

Thin wrappers over the modular pieces for the most common workflows, so a user
can get an analytic FC or a simulated BOLD FC in one call, while every knob
remains accessible on the underlying objects.
"""
from __future__ import annotations


import numpy as np

from .bases.modal_basis import ModalBasis
from .config import SimConfig
from .metrics.fc import functional_connectivity
from .models.hopf_modal import HopfModal
from .models.ou_modal import OUModal
from .simulate.simulator import Simulator


def analytic_fc(basis: ModalBasis, bifurcation: float = -0.02, coupling: float = 0.0,
                intrinsic_freq: float = 0.05, noise: float = 0.02) -> np.ndarray:
    """Closed-form functional connectivity of the linear model (no simulation).

    The low-risk MVP path: builds an :class:`OUModal` and returns its analytic
    Lyapunov FC. Fast enough to sweep thousands of parameter sets.
    """
    model = OUModal(basis, bifurcation=bifurcation, coupling=coupling,
                    intrinsic_freq=intrinsic_freq, noise=noise)
    return model.functional_connectivity()


def simulate_fc(basis: ModalBasis, model: str = "hopf", bifurcation: float = -0.02,
                coupling: float = 0.05, intrinsic_freq: float = 0.05,
                noise: float = 0.02, duration: float = 600.0, dt: float = 0.1,
                bold: bool = False, seed: int = 0, **model_kwargs) -> np.ndarray:
    """Simulate a model and return its functional connectivity.

    Parameters
    ----------
    basis : ModalBasis.
    model : {'hopf', 'ou'}.
    bifurcation, coupling, intrinsic_freq, noise : model parameters.
    duration, dt : simulation length and step (s).
    bold : if True, FC is computed on the Balloon-Windkessel BOLD signal.
    seed : int.
    **model_kwargs : forwarded to the model (e.g. ``dealias`` for Hopf).

    Returns
    -------
    np.ndarray, the node-by-node FC matrix.
    """
    if model == "hopf":
        m = HopfModal(basis, bifurcation=bifurcation, coupling=coupling,
                      intrinsic_freq=intrinsic_freq, noise=noise, **model_kwargs)
    elif model == "ou":
        m = OUModal(basis, bifurcation=bifurcation, coupling=coupling,
                    intrinsic_freq=intrinsic_freq, noise=noise)
    else:
        raise ValueError(f"model must be 'hopf' or 'ou'; got {model!r}")
    cfg = SimConfig(n_modes=basis.n_modes, dt=dt, duration=duration, seed=seed)
    res = Simulator(m, cfg).run(seed=seed, store_nodes=True, bold=bold)
    signal = res.bold if bold else res.reconstruct_nodes()
    return functional_connectivity(signal)
