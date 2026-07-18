"""Modal-agnostic validation metrics (the second validation route).

Per-mode power spectra, timescales, the modal energy spectrum (power vs mode /
wavelength), spectral entropy, and reconstruction accuracy vs number of modes --
the quantities Pang et al. 2023 used to validate the modal decomposition itself,
independent of any hemodynamic model.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def modal_power(modal_timeseries) -> np.ndarray:
    """Time-averaged power per mode: <|a_k(t)|^2>_t."""
    a = np.asarray(modal_timeseries)
    return np.mean(np.abs(a) ** 2, axis=1)


def modal_power_spectrum(modal_timeseries, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-mode power spectral density via the FFT.

    Modal coefficients are complex, so the full (two-sided) FFT is used and the
    non-negative-frequency half is returned.

    Returns
    -------
    freqs : np.ndarray, shape (n_freqs,)
    psd : np.ndarray, shape (n_modes, n_freqs)
    """
    a = np.asarray(modal_timeseries)
    n_time = a.shape[1]
    freqs_full = np.fft.fftfreq(n_time, d=dt)
    A = np.fft.fft(a, axis=1)
    psd_full = (np.abs(A) ** 2) / n_time
    pos = freqs_full >= 0
    order = np.argsort(freqs_full[pos])
    return freqs_full[pos][order], psd_full[:, pos][:, order]


def modal_timescales(modal_timeseries, dt: float) -> np.ndarray:
    """Per-mode decorrelation timescale from the lag-1 autocorrelation.

    tau_k = -dt / ln(rho_k), with rho_k the lag-1 autocorrelation of |a_k(t)|.
    """
    a = np.abs(np.asarray(modal_timeseries))
    a = a - a.mean(axis=1, keepdims=True)
    num = np.sum(a[:, :-1] * a[:, 1:], axis=1)
    den = np.sum(a * a, axis=1)
    rho = np.clip(num / np.clip(den, 1e-30, None), 1e-6, 1 - 1e-6)
    return -dt / np.log(rho)


def spectral_entropy(modal_timeseries) -> float:
    """Shannon entropy of the normalized modal power distribution (in bits).

    High entropy = power spread across many modes; low = concentrated in few.
    """
    p = modal_power(modal_timeseries)
    p = p / np.clip(p.sum(), 1e-30, None)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def reconstruction_curve(field, basis, n_modes_grid) -> np.ndarray:
    """Reconstruction accuracy (Pearson r) vs number of modes used.

    Parameters
    ----------
    field : np.ndarray, shape (n_nodes,) or (n_nodes, k).
    basis : ModalBasis.
    n_modes_grid : sequence of ints.

    Returns
    -------
    np.ndarray, shape (len(n_modes_grid),), the accuracy at each truncation.
    """
    y = np.asarray(field, dtype=np.float64).reshape(basis.n_nodes, -1)
    accs = []
    for n in n_modes_grid:
        b = basis.truncate(int(n))
        yhat = b.reconstruct(b.project(y))
        accs.append(np.corrcoef(y.ravel(), yhat.ravel())[0, 1])
    return np.asarray(accs)
