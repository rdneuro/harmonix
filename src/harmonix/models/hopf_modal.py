"""Nonlinear heterogeneous Hopf / Stuart-Landau modal model (Deco et al.).

Node-space dynamics:

    dz_j/dt = (a_j + i*omega_j - |z_j|^2) z_j + G * sum_k C_jk (z_k - z_j) + noise.

The linear + coupling terms diagonalize in the coupling eigenbasis; the coupling
-G*L contributes -G*lambda_k per mode. The cubic nonlinearity |z|^2 z does NOT
diagonalize -- it mixes modes through triadic interactions. harmonix handles it
with the standard Galerkin / pseudo-spectral round-trip: reconstruct z in node
space, evaluate the pointwise nonlinearity, project back to modal space.

    a_nonlinear = -Phi^T M ( |z_node|^2 * z_node ),   z_node = Phi a.

Mode truncation + a cubic term causes aliasing; an optional exponential spectral
filter (or the strict 2/4 dealiasing rule) damps energy leakage into the highest
retained modes. This is the least-established part of harmonix and is flagged as
such.

The model also exposes the modal Jacobian at the fixed point, from which the
criticality module reads the spectral abscissa and relaxation time.
"""
from __future__ import annotations


import numpy as np

from ..bases.modal_basis import ModalBasis
from ..config import (DEFAULT_BIFURCATION, DEFAULT_INTRINSIC_FREQ_HZ,
                      DEFAULT_NOISE)
from .base import ModalModel


class HopfModal(ModalModel):
    """Nonlinear Stuart-Landau oscillators evolved in modal space.

    Parameters
    ----------
    basis : ModalBasis
        Coupling eigenbasis (its eigenvalues give the modal coupling -G*lambda_k).
    bifurcation : float or np.ndarray
        Bifurcation parameter a (scalar or per-node, shape (n_nodes,)).
        a<0 stable focus, a>0 limit cycle, a~0 critical. Default -0.02.
    coupling : float
        Global coupling G. Default 0.
    intrinsic_freq : float or np.ndarray
        Intrinsic frequency in Hz (scalar or per-node). Default 0.05 Hz.
    noise : float
        Noise amplitude beta. Default 0.02.
    dealias : {'none', 'filter', 'two_thirds'}
        Spectral dealiasing for the cubic term. 'filter' applies a smooth
        exponential high-mode filter (recommended default); 'two_thirds' zeroes
        the top third of modes; 'none' disables (fastest, risk of aliasing).
    filter_strength : float
        Exponent for the exponential filter (larger = sharper cutoff).
    """

    state_kind = "complex1"
    is_nonlinear = True

    def __init__(self, basis: ModalBasis, bifurcation=DEFAULT_BIFURCATION,
                 coupling: float = 0.0, intrinsic_freq=DEFAULT_INTRINSIC_FREQ_HZ,
                 noise: float = DEFAULT_NOISE, dealias: str = "filter",
                 filter_strength: float = 36.0):
        super().__init__(basis)
        self.coupling = float(coupling)
        self.noise = float(noise)
        if dealias not in ("none", "filter", "two_thirds"):
            raise ValueError(f"unknown dealias {dealias!r}")
        self.dealias = dealias

        # broadcast per-node parameters
        a = np.asarray(bifurcation, dtype=np.float64)
        w = 2.0 * np.pi * np.asarray(intrinsic_freq, dtype=np.float64)
        self._a_node = np.full(basis.n_nodes, float(a)) if a.ndim == 0 else a
        self._w_node = np.full(basis.n_nodes, float(w)) if w.ndim == 0 else w
        if self._a_node.shape[0] != basis.n_nodes or self._w_node.shape[0] != basis.n_nodes:
            raise ValueError("per-node bifurcation/intrinsic_freq must have length n_nodes")
        self._homogeneous = (a.ndim == 0 and w.ndim == 0)
        self._a_scalar = float(a) if a.ndim == 0 else None
        self._w_scalar = float(w) if w.ndim == 0 else None

        # exponential spectral filter over mode index (1 -> 0 across the spectrum)
        if dealias == "filter":
            k = np.arange(self.n_modes) / max(self.n_modes - 1, 1)
            self._filt = np.exp(-filter_strength * k ** 8)
        elif dealias == "two_thirds":
            self._filt = (np.arange(self.n_modes) < (2 * self.n_modes) // 3).astype(float)
        else:
            self._filt = None

    def linear_rates(self) -> np.ndarray:
        """Per-mode complex linear rate (homogeneous case). mu_k = (a+i w) - G lambda_k."""
        if not self._homogeneous:
            # heterogeneous: linear part is not diagonal; use jacobian() instead.
            # Return the mean-field diagonal approximation for the integrator's
            # linear split (nonlinearity handles the rest via node-space eval).
            a_mean = self._a_node.mean()
            w_mean = self._w_node.mean()
            return (a_mean + 1j * w_mean) - self.coupling * self.basis.eigenvalues
        return (self._a_scalar + 1j * self._w_scalar) - self.coupling * self.basis.eigenvalues

    def noise_covariance(self) -> np.ndarray:
        return np.full(self.n_modes, self.noise ** 2, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Galerkin nonlinearity (node-space round-trip)
    # ------------------------------------------------------------------ #
    def nonlinear_drift(self, a):
        """Modal projection of the cubic term: -Phi^T M (|z|^2 z), z = Phi a.

        Also folds in the per-node heterogeneous linear part when parameters are
        node-specific (it does not diagonalize), so the total node-space drift is
        represented exactly up to truncation.
        """
        Phi = self.basis.Phi
        z_node = Phi @ a                                      # (n_nodes,) node reconstruction
        cubic = (np.abs(z_node) ** 2) * z_node               # pointwise nonlinearity
        if not self._homogeneous:
            # add the residual heterogeneous linear part (per-node minus mean used in linear_rates)
            a_res = self._a_node - self._a_node.mean()
            w_res = self._w_node - self._w_node.mean()
            hetero = (a_res + 1j * w_res) * z_node
            drift_node = -cubic + hetero
        else:
            drift_node = -cubic
        a_nl = self.basis.project(drift_node)                # Phi^T M drift_node
        if self._filt is not None:
            a_nl = a_nl * self._filt
        return a_nl

    # ------------------------------------------------------------------ #
    # Jacobian at the origin (for criticality)
    # ------------------------------------------------------------------ #
    def modal_jacobian(self) -> np.ndarray:
        """Modal Jacobian at the fixed point z=0 (linearization).

        At the origin the cubic term vanishes, so J = diag(mu_k) for homogeneous
        parameters, or Phi^T M [ (a_j + i w_j) I - G L ] Phi for heterogeneous
        parameters (dense).

        Returns
        -------
        np.ndarray, shape (n_modes, n_modes), complex.
        """
        lam = self.basis.eigenvalues
        if self._homogeneous:
            return np.diag((self._a_scalar + 1j * self._w_scalar) - self.coupling * lam)
        Phi = self.basis.Phi
        # node-space linear operator diag(a_j + i w_j) - G L, with L = Phi diag(lam) Phi^T M^{-1}
        node_lin = (self._a_node + 1j * self._w_node)         # (n_nodes,)
        # J_modal = Phi^T M diag(node_lin) Phi - G diag(lam)
        MPhi = Phi if self.basis.mass is None else (self.basis.mass[:, None] * Phi)
        J = Phi.T @ (node_lin[:, None] * MPhi) - self.coupling * np.diag(lam)
        return J
