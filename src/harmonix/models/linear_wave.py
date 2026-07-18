"""Linear damped-wave modal model (Pang et al. 2023 / Robinson neural field theory).

The isotropic damped cortical wave equation

    [(1/gamma_s^2) d2/dt2 + (2/gamma_s) d/dt + 1 - r_s^2 nabla^2] phi = Q

diagonalizes in the LBO eigenbasis (nabla^2 psi_k = -lambda_k psi_k) into one
independent damped harmonic oscillator per mode:

    (1/gamma_s^2) a_k'' + (2/gamma_s) a_k' + (1 + r_s^2 lambda_k) a_k = Q_k.

Defaults follow Pang's released ``loadParameters_wave_func.m``:
gamma_s = 116 s^-1 (damping rate), r_s = 30 mm (spatial length scale; the single
fitted parameter). The per-mode frequency-domain transfer function is a Lorentzian
resonance, giving an analytic power spectrum without simulation.
"""
from __future__ import annotations

import numpy as np

from ..bases.modal_basis import ModalBasis
from ..config import WAVE_GAMMA_S, WAVE_R_S_MM
from .base import ModalModel


class LinearWave(ModalModel):
    """Damped-wave modal model: one damped oscillator per eigenmode.

    Parameters
    ----------
    basis : ModalBasis
        Geometric eigenbasis (eigenvalues in 1/mm^2 if r_s is in mm).
    gamma_s : float
        Damping rate in s^-1 (default 116).
    r_s : float
        Spatial length scale in mm (default 30).
    noise : float
        White-noise drive amplitude on each mode.
    """

    state_kind = "real2"
    is_nonlinear = False

    def __init__(self, basis: ModalBasis, gamma_s: float = WAVE_GAMMA_S,
                 r_s: float = WAVE_R_S_MM, noise: float = 1.0):
        super().__init__(basis)
        self.gamma_s = float(gamma_s)
        self.r_s = float(r_s)
        self.noise = float(noise)

    def stiffness(self) -> np.ndarray:
        """Per-mode stiffness coefficient (1 + r_s^2 lambda_k)."""
        return 1.0 + (self.r_s ** 2) * self.basis.eigenvalues

    def linear_rates(self) -> np.ndarray:
        """Complex rates of the first-order (a, a') reduction, per mode.

        Rewriting the 2nd-order ODE as a 2-D first-order system per mode, the two
        characteristic rates are gamma_s*(-1 +/- sqrt(1 - stiffness)). Returned as
        the pair with negative real part (the physical decaying branch).
        """
        k = self.stiffness()
        disc = np.sqrt(np.complex128(1.0) - k)
        return self.gamma_s * (-1.0 + disc)

    def noise_covariance(self) -> np.ndarray:
        return np.full(self.n_modes, self.noise ** 2, dtype=np.float64)

    def natural_frequencies(self) -> np.ndarray:
        """Undamped natural frequency per mode (Hz): gamma_s*sqrt(stiffness)/(2 pi)."""
        return self.gamma_s * np.sqrt(np.clip(self.stiffness(), 0, None)) / (2.0 * np.pi)

    def transfer_function(self, freqs) -> np.ndarray:
        """Per-mode frequency response a_k(f)/Q_k(f) (complex Lorentzian).

        H_k(omega) = gamma_s^2 / (-omega^2 - 2 i omega gamma_s + gamma_s^2 * stiffness_k),
        with omega = 2*pi*f.

        Parameters
        ----------
        freqs : np.ndarray, shape (n_freqs,), frequencies in Hz.

        Returns
        -------
        np.ndarray, shape (n_modes, n_freqs), complex.
        """
        f = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        omega = 2.0 * np.pi * f[None, :]
        k = self.stiffness()[:, None]
        g = self.gamma_s
        return (g ** 2) / (-omega ** 2 - 2j * omega * g + g ** 2 * k)

    def power_spectrum(self, freqs) -> np.ndarray:
        """Analytic per-mode power spectrum |H_k(f)|^2 * Q_k."""
        H = self.transfer_function(freqs)
        return (np.abs(H) ** 2) * self.noise_covariance()[:, None]
