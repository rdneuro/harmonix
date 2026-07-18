"""Criticality as a spectral quantity.

For the linearized modal system, distance-to-criticality is read directly from the
Jacobian spectrum: the SPECTRAL ABSCISSA alpha = max_k Re(mu_k) is the distance to
the bifurcation (alpha < 0 stable, alpha -> 0 critical), the RELAXATION TIME is
tau = -1/alpha, and the slowest-relaxing mode is the critical one. This gives a
cheap, principled criticality diagnostic and working-point finder that needs no
simulation.

Caveat (stated honestly): this equivalence is exact for the linearized system.
For the full nonlinear Hopf model it is a heuristic -- exact at the bifurcation via
center-manifold reduction, approximate away from it. Whole-brain Jacobians are also
generally NON-NORMAL, so a stable system can still show transient amplification;
:func:`numerical_abscissa` reports that complementary reactivity measure.
"""
from __future__ import annotations


import numpy as np


def _jacobian_of(model):
    """Extract a modal Jacobian from a model (uses modal_jacobian or diag(rates))."""
    if hasattr(model, "modal_jacobian"):
        return np.asarray(model.modal_jacobian())
    return np.diag(np.asarray(model.linear_rates()))


def spectral_abscissa(model_or_jacobian) -> float:
    """Largest real part of the Jacobian eigenvalues (distance to bifurcation).

    Accepts a model (uses its modal Jacobian) or a Jacobian matrix directly.
    Negative = stable; approaching 0 from below = approaching criticality.
    """
    J = (model_or_jacobian if isinstance(model_or_jacobian, np.ndarray)
         else _jacobian_of(model_or_jacobian))
    if J.ndim == 1:
        return float(np.max(np.real(J)))
    return float(np.max(np.real(np.linalg.eigvals(J))))


def relaxation_time(model_or_jacobian) -> float:
    """Slowest relaxation time tau = -1 / spectral_abscissa (inf at criticality)."""
    alpha = spectral_abscissa(model_or_jacobian)
    if alpha >= 0:
        return np.inf
    return -1.0 / alpha


def distance_to_criticality(model_or_jacobian) -> float:
    """Distance below the bifurcation: -spectral_abscissa (0 = critical)."""
    return -spectral_abscissa(model_or_jacobian)


def spectral_gap(model_or_jacobian) -> float:
    """Gap between the two slowest-relaxing modes (real parts of the spectrum)."""
    J = (model_or_jacobian if isinstance(model_or_jacobian, np.ndarray)
         else _jacobian_of(model_or_jacobian))
    re = np.real(np.diag(J)) if J.ndim == 2 and np.allclose(J, np.diag(np.diag(J))) \
        else np.real(np.linalg.eigvals(J))
    re_sorted = np.sort(re)[::-1]
    if re_sorted.size < 2:
        return np.inf
    return float(re_sorted[0] - re_sorted[1])


def numerical_abscissa(model_or_jacobian) -> float:
    """Numerical abscissa: max eigenvalue of the Hermitian part (J+J^H)/2.

    Bounds the instantaneous growth rate; for non-normal Jacobians it can be
    positive even when the spectral abscissa is negative, signalling transient
    amplification (reactivity) that the eigenvalues alone miss.
    """
    J = (model_or_jacobian if isinstance(model_or_jacobian, np.ndarray)
         else _jacobian_of(model_or_jacobian))
    if J.ndim == 1:
        J = np.diag(J)
    H = 0.5 * (J + J.conj().T)
    return float(np.max(np.real(np.linalg.eigvalsh(H))))


def modal_relaxation_times(model_or_jacobian) -> np.ndarray:
    """Per-mode relaxation times tau_k = -1/Re(mu_k) (the slowest is critical)."""
    J = (model_or_jacobian if isinstance(model_or_jacobian, np.ndarray)
         else _jacobian_of(model_or_jacobian))
    re = np.real(np.diag(J)) if J.ndim == 2 else np.real(J)
    with np.errstate(divide="ignore"):
        return np.where(re < 0, -1.0 / re, np.inf)
