"""Stochastic integrators for modal dynamics.

For first-order complex modal models (OU, Hopf) the linear operator is DIAGONAL in
modal space (a per-mode complex rate vector ``mu``), which makes the exponential
time-differencing scheme ETDRK4 essentially free: the matrix exponential reduces
to elementwise ``exp(mu*dt)`` and the ETD phi-functions become scalar functions of
``mu*dt`` per mode. This is the elegant payoff of working in the eigenbasis.

Three integrators are provided:

* ``euler_maruyama`` -- simplest, O(dt) weak order; good for quick runs.
* ``stochastic_heun`` -- predictor-corrector, better weak accuracy (default).
* ``etdrk4`` -- exponential integrator exact on the diagonal linear part, ideal
  for stiff/oscillatory modes; nonlinearity via the node-space round-trip.

All operate on the modal state ``z`` (shape ``(n_modes,)`` or ``(batch, n_modes)``)
and take pre-drawn noise increments so the caller controls the (Philox) RNG.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def euler_maruyama(z, mu, nonlinear: Callable, noise_amp, dt: float, dW):
    """One Euler-Maruyama step of dz = (mu*z + N(z)) dt + noise_amp * dW.

    Parameters
    ----------
    z : ndarray, current modal state (..., n_modes) complex.
    mu : ndarray, per-mode linear rate (n_modes,) complex.
    nonlinear : callable z -> (..., n_modes), the nonlinear drift.
    noise_amp : ndarray or float, per-mode noise amplitude.
    dt : float, timestep.
    dW : ndarray, complex noise increment (same shape as z), unit variance.

    Returns
    -------
    ndarray, the state after one step.
    """
    drift = mu * z + nonlinear(z)
    return z + dt * drift + noise_amp * np.sqrt(dt) * dW


def stochastic_heun(z, mu, nonlinear: Callable, noise_amp, dt: float, dW):
    """One stochastic Heun (predictor-corrector) step."""
    f0 = mu * z + nonlinear(z)
    noise = noise_amp * np.sqrt(dt) * dW
    z_pred = z + dt * f0 + noise
    f1 = mu * z_pred + nonlinear(z_pred)
    return z + 0.5 * dt * (f0 + f1) + noise


def _etdrk4_coeffs(mu, dt, n_contour: int = 32):
    """Scalar ETDRK4 coefficients for a diagonal linear operator ``mu``.

    Uses the Kassam-Trefethen contour-integral evaluation of the phi-functions to
    avoid cancellation when ``mu*dt`` is small. Returns (E, E2, Q, f1, f2, f3),
    each shape ``(n_modes,)``.
    """
    mu = np.asarray(mu)
    E = np.exp(mu * dt)
    E2 = np.exp(mu * dt / 2.0)
    # contour of points around each mu*dt
    roots = np.exp(1j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
    LR = (mu * dt)[:, None] + roots[None, :]                  # (n_modes, n_contour)
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))
    return E, E2, Q, f1, f2, f3


class ETDRK4Stepper:
    """Reusable ETDRK4 stepper (precomputes coefficients once for fixed dt).

    Because the linear operator is diagonal, the additive stochastic forcing is
    propagated *exactly*: over one step the linear part contributes noise variance
    ``Q_k * (exp(2 Re(mu_k) dt) - 1) / (2 Re(mu_k))`` per mode (the exact
    Ornstein-Uhlenbeck increment), which reduces to ``Q_k dt`` as dt -> 0 but is
    accurate for larger steps. This makes ETDRK4 exact on both the linear drift and
    the additive noise, with only the nonlinearity treated by the RK4 stages.
    """

    def __init__(self, mu, dt: float):
        self.E, self.E2, self.Q, self.f1, self.f2, self.f3 = _etdrk4_coeffs(mu, dt)
        self.dt = dt
        # exact per-mode noise standard-deviation factor for the diagonal linear part
        re = np.real(np.asarray(mu))
        with np.errstate(divide="ignore", invalid="ignore"):
            var_factor = np.where(np.abs(re) > 1e-12,
                                  (np.exp(2.0 * re * dt) - 1.0) / (2.0 * re),
                                  dt)
        self._noise_scale = np.sqrt(np.clip(var_factor, 0.0, None))

    def step(self, z, nonlinear: Callable, noise_amp, dW):
        """One ETDRK4 step with exact additive-noise propagation."""
        E, E2, Q, f1, f2, f3 = self.E, self.E2, self.Q, self.f1, self.f2, self.f3
        Nu = nonlinear(z)
        a = E2 * z + Q * Nu
        Na = nonlinear(a)
        b = E2 * z + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2.0 * Nb - Nu)
        Nc = nonlinear(c)
        z_next = E * z + Nu * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        z_next = z_next + noise_amp * self._noise_scale * dW    # exact OU noise increment
        return z_next


INTEGRATORS = {"euler": euler_maruyama, "heun": stochastic_heun}
