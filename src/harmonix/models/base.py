"""Base class for modal dynamical models.

A modal model specifies how the eigenmode coefficients a_k(t) (or complex z_k(t))
evolve. Every model exposes a diagonal *linear* operator in modal space (the part
that diagonalizes exactly in the eigenbasis) plus, where present, a nonlinear
drift evaluated via a node-space round-trip (Galerkin projection). Linear/analytic
models additionally expose closed-form spectra and stationary covariance so that
functional connectivity can be computed WITHOUT time-stepping.

State conventions
-----------------
* Complex first-order models (OU, Hopf): state ``z`` has shape ``(n_modes,)`` or,
  batched, ``(batch, n_modes)``; dtype complex.
* Second-order real models (wave): state is ``(a, a_dot)`` each ``(n_modes,)``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..bases.modal_basis import ModalBasis


class ModalModel(ABC):
    """Abstract modal dynamical model on a fixed :class:`ModalBasis`."""

    #: 'complex1' (first-order complex) or 'real2' (second-order real)
    state_kind: str = "complex1"
    #: True if the model has a nonlinear term requiring a node-space round-trip
    is_nonlinear: bool = False

    def __init__(self, basis: ModalBasis):
        if not isinstance(basis, ModalBasis):
            raise TypeError("basis must be a harmonix ModalBasis")
        self.basis = basis
        self.n_modes = basis.n_modes

    @abstractmethod
    def linear_rates(self) -> np.ndarray:
        """Per-mode linear operator (diagonal in modal space).

        Returns
        -------
        np.ndarray, shape (n_modes,)
            For first-order complex models, the complex rate mu_k such that the
            linear part is da_k/dt = mu_k a_k. Stability requires Re(mu_k) < 0.
        """

    @abstractmethod
    def noise_covariance(self) -> np.ndarray:
        """Per-mode noise intensity Q_k (diagonal), shape (n_modes,)."""

    def nonlinear_drift(self, z):
        """Nonlinear contribution to da/dt in modal space (0 for linear models)."""
        return np.zeros_like(z)

    def drift(self, z, t: float = 0.0):
        """Deterministic drift da/dt = linear + nonlinear in modal space."""
        return self.linear_rates() * z + self.nonlinear_drift(z)
