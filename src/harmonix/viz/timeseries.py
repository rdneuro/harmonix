"""Time-series and spectral line plots for modal simulations."""
from __future__ import annotations


import numpy as np

from ._style import OKABE_ITO, _get_ax


def plot_modal_amplitudes(modal_timeseries, times=None, n_show: int = 6, ax=None,
                          title: str = "Modal amplitudes"):
    """Plot |a_k(t)| for the first ``n_show`` modes."""
    a = np.abs(np.asarray(modal_timeseries))
    if times is None:
        times = np.arange(a.shape[1])
    fig, ax = _get_ax(ax, figsize=(6, 3.5))
    for k in range(min(n_show, a.shape[0])):
        ax.plot(times, a[k], lw=1.0, color=OKABE_ITO[k % len(OKABE_ITO)],
                label=f"mode {k}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|a_k(t)|")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    return fig, ax


def plot_power_spectrum(freqs, psd, modes=None, ax=None, loglog: bool = True,
                        title: str = "Modal power spectra"):
    """Plot per-mode power spectra (subset of modes)."""
    psd = np.asarray(psd)
    if modes is None:
        modes = range(min(6, psd.shape[0]))
    fig, ax = _get_ax(ax, figsize=(5.5, 3.5))
    for i, k in enumerate(modes):
        ax.plot(freqs, psd[k], lw=1.0, color=OKABE_ITO[i % len(OKABE_ITO)],
                label=f"mode {k}")
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    return fig, ax


def plot_order_parameter(order_parameter, times=None, ax=None,
                         title: str = "Kuramoto order parameter"):
    """Plot the Kuramoto order parameter R(t)."""
    R = np.asarray(order_parameter)
    if times is None:
        times = np.arange(R.shape[0])
    fig, ax = _get_ax(ax, figsize=(6, 3))
    ax.plot(times, R, lw=1.0, color=OKABE_ITO[0])
    ax.axhline(np.mean(R), ls="--", lw=0.8, color=OKABE_ITO[1],
               label=f"mean = {np.mean(R):.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("R(t)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    return fig, ax
