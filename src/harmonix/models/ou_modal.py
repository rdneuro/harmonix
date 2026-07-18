"""Linear stochastic (Ornstein-Uhlenbeck / linearized-Hopf) modal model.

Below the Hopf bifurcation the Stuart-Landau oscillator linearizes to an OU
process. In modal space (the eigenbasis of the coupling Laplacian) the dynamics
diagonalize into independent complex OU modes

    dz_k/dt = mu_k z_k + noise,     mu_k = (a + i*omega) - G*lambda_k,

with stationary variance Var(z_k) = Q_k / (-2 Re(mu_k)) when Re(mu_k) < 0. The
node-space stationary covariance -- and hence functional connectivity -- then has
the CLOSED FORM  Cov = Phi diag(Var_k) Phi^T, so FC needs no simulation.

For heterogeneous parameters (per-node a_j or omega_j) the model does not
diagonalize in the coupling eigenbasis; harmonix then solves the full continuous
Lyapunov equation  J Cov + Cov J^H + Q = 0  directly. Both paths are exposed;
the diagonal path is the fast default and the low-risk MVP core (Nozari et al.
2024 show linear models describe macroscale fMRI best).
"""
from __future__ import annotations


import numpy as np
import scipy.linalg as sla

from ..bases.modal_basis import ModalBasis
from ..config import (DEFAULT_BIFURCATION, DEFAULT_INTRINSIC_FREQ_HZ,
                      DEFAULT_NOISE)
from .base import ModalModel


class OUModal(ModalModel):
    """Linear modal Ornstein-Uhlenbeck model with analytic FC.

    Parameters
    ----------
    basis : ModalBasis
        The eigenbasis; its eigenvalues supply the modal coupling -G*lambda_k.
    bifurcation : float
        The linear bifurcation parameter a (must be < 0 for stability). Default
        -0.02 (just below the Hopf bifurcation).
    coupling : float
        Global coupling G scaling the modal decay via -G*lambda_k. Default 0.
    intrinsic_freq : float
        Intrinsic oscillation frequency in Hz (omega = 2*pi*f). Default 0.05 Hz.
    noise : float
        Noise amplitude beta; modal noise intensity Q_k = beta^2. Default 0.02.
    """

    state_kind = "complex1"
    is_nonlinear = False

    def __init__(self, basis: ModalBasis, bifurcation: float = DEFAULT_BIFURCATION,
                 coupling: float = 0.0, intrinsic_freq: float = DEFAULT_INTRINSIC_FREQ_HZ,
                 noise: float = DEFAULT_NOISE):
        super().__init__(basis)
        if bifurcation >= 0:
            raise ValueError(
                f"bifurcation a must be < 0 for a stable linear model; got {bifurcation}")
        self.bifurcation = float(bifurcation)
        self.coupling = float(coupling)
        self.omega = 2.0 * np.pi * float(intrinsic_freq)
        self.noise = float(noise)

    def linear_rates(self) -> np.ndarray:
        lam = self.basis.eigenvalues
        return (self.bifurcation + 1j * self.omega) - self.coupling * lam

    def noise_covariance(self) -> np.ndarray:
        return np.full(self.n_modes, self.noise ** 2, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # closed-form (no-simulation) quantities
    # ------------------------------------------------------------------ #
    def modal_variance(self) -> np.ndarray:
        """Stationary variance of each modal coefficient (analytic).

        Var(z_k) = Q_k / (-2 Re(mu_k)); requires Re(mu_k) < 0.
        """
        mu = self.linear_rates()
        re = np.real(mu)
        if np.any(re >= 0):
            n_unstable = int(np.sum(re >= 0))
            raise ValueError(
                f"{n_unstable} modes are unstable (Re(mu_k) >= 0); reduce coupling "
                "or bifurcation. Use spectral_abscissa() to inspect.")
        return self.noise_covariance() / (-2.0 * re)

    def stationary_covariance(self) -> np.ndarray:
        """Node-space stationary covariance Cov = Phi diag(Var_k) Phi^T.

        Uses the complex modal amplitude variance Var_k = E|z_k|^2. For a
        circularly-symmetric complex OU process the covariance of the real part of
        node activity is half this; the factor cancels in the correlation, so
        :meth:`functional_connectivity` (which normalizes) is unaffected. Returned
        here is the (unnormalized) complex-amplitude covariance.

        Returns
        -------
        np.ndarray, shape (n_nodes, n_nodes)
        """
        var = self.modal_variance()
        Phi = self.basis.Phi
        return (Phi * var[None, :]) @ Phi.T

    def functional_connectivity(self) -> np.ndarray:
        """Analytic functional connectivity (correlation of node activity)."""
        cov = self.stationary_covariance()
        d = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
        fc = cov / np.outer(d, d)
        np.fill_diagonal(fc, 1.0)
        return fc

    def power_spectrum(self, freqs) -> np.ndarray:
        """Analytic per-mode power spectrum (Lorentzian).

        S_k(f) = Q_k / (|2*pi*i*f - mu_k|^2).

        Parameters
        ----------
        freqs : np.ndarray, shape (n_freqs,)
            Frequencies in Hz.

        Returns
        -------
        np.ndarray, shape (n_modes, n_freqs)
        """
        f = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        mu = self.linear_rates()[:, None]                     # (n_modes, 1)
        Q = self.noise_covariance()[:, None]
        denom = np.abs(2j * np.pi * f[None, :] - mu) ** 2
        return Q / denom

    def relaxation_times(self) -> np.ndarray:
        """Per-mode relaxation time tau_k = -1 / Re(mu_k)."""
        return -1.0 / np.real(self.linear_rates())

    def lyapunov_covariance(self, jacobian=None, noise_cov=None) -> np.ndarray:
        """Solve the full continuous Lyapunov equation for a general Jacobian.

        Use this for heterogeneous (per-node) parameters that do NOT diagonalize
        in the coupling eigenbasis. Solves J*Cov + Cov*J^H + Q = 0.

        Parameters
        ----------
        jacobian : np.ndarray, shape (n, n), optional
            System Jacobian. Defaults to the modal diagonal diag(mu_k).
        noise_cov : np.ndarray, shape (n, n), optional
            Noise covariance Q. Defaults to diag(Q_k).

        Returns
        -------
        np.ndarray, the stationary covariance in the Jacobian's coordinates.
        """
        if jacobian is None:
            jacobian = np.diag(self.linear_rates())
        if noise_cov is None:
            noise_cov = np.diag(self.noise_covariance().astype(np.complex128))
        # scipy solves A X + X A^H = Q  as solve_continuous_lyapunov(A, -Q)
        return sla.solve_continuous_lyapunov(jacobian, -noise_cov)
