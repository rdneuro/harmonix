"""Criticality curve plots (metric or spectral abscissa vs coupling)."""
from __future__ import annotations


import numpy as np

from ._style import OKABE_ITO, _get_ax


def plot_susceptibility_curve(couplings, values, ax=None, working_point=None,
                              ylabel: str = "susceptibility",
                              title: str = "Working-point search"):
    """Plot a fluctuation metric vs coupling, marking the working point."""
    G = np.asarray(couplings, dtype=float)
    v = np.asarray(values, dtype=float)
    fig, ax = _get_ax(ax, figsize=(5.5, 3.5))
    ax.plot(G, v, "-o", ms=3, color=OKABE_ITO[0])
    if working_point is not None:
        ax.axvline(working_point, ls="--", color=OKABE_ITO[1],
                   label=f"G* = {working_point:.3f}")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("global coupling G")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_abscissa_curve(couplings, abscissas, ax=None,
                        title: str = "Spectral abscissa vs coupling"):
    """Plot the spectral abscissa vs coupling with the criticality line at 0."""
    G = np.asarray(couplings, dtype=float)
    a = np.asarray(abscissas, dtype=float)
    fig, ax = _get_ax(ax, figsize=(5.5, 3.5))
    ax.plot(G, a, "-o", ms=3, color=OKABE_ITO[2])
    ax.axhline(0, ls="--", color="k", lw=1, label="criticality")
    ax.set_xlabel("global coupling G")
    ax.set_ylabel("spectral abscissa  max Re(mu)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    return fig, ax
