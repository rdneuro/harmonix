"""Panel plots: bar, scatter, and volcano for modal/regional summaries."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._style import OKABE_ITO, _get_ax


def plot_variance_explained(eigenvalues_or_power, ax=None, cumulative: bool = True,
                            title: str = "Variance explained per mode"):
    """Bar plot of per-mode power / variance, optionally with a cumulative curve."""
    p = np.asarray(eigenvalues_or_power, dtype=float)
    p = p / np.clip(p.sum(), 1e-30, None)
    fig, ax = _get_ax(ax, figsize=(6, 3.5))
    ax.bar(np.arange(len(p)), p, color=OKABE_ITO[0], width=0.9)
    ax.set_xlabel("mode")
    ax.set_ylabel("fraction of power")
    ax.set_title(title)
    if cumulative:
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(p)), np.cumsum(p), color=OKABE_ITO[1], lw=1.5)
        ax2.set_ylabel("cumulative", color=OKABE_ITO[1])
        ax2.set_ylim(0, 1.02)
    return fig, ax


def plot_fc_scatter(fc_empirical, fc_simulated, ax=None,
                    title: str = "Empirical vs simulated FC"):
    """Scatter of empirical vs simulated FC edges with the identity line and r."""
    A = np.asarray(fc_empirical)
    B = np.asarray(fc_simulated)
    iu = np.triu_indices(A.shape[0], k=1)
    ea, eb = A[iu], B[iu]
    r = np.corrcoef(ea, eb)[0, 1]
    fig, ax = _get_ax(ax, figsize=(4.5, 4.2))
    ax.scatter(ea, eb, s=4, alpha=0.3, color=OKABE_ITO[0])
    lim = [min(ea.min(), eb.min()), max(ea.max(), eb.max())]
    ax.plot(lim, lim, ls="--", lw=1, color="k")
    ax.set_xlabel("empirical FC")
    ax.set_ylabel("simulated FC")
    ax.set_title(f"{title} (r = {r:.3f})")
    return fig, ax


def plot_volcano(effect_sizes, pvalues, ax=None, labels: Optional[Sequence[str]] = None,
                 alpha: float = 0.05, title: str = "Volcano"):
    """Volcano plot of effect size vs -log10(p), highlighting significant items."""
    es = np.asarray(effect_sizes, dtype=float)
    p = np.asarray(pvalues, dtype=float)
    neglogp = -np.log10(np.clip(p, 1e-300, None))
    sig = p < alpha
    fig, ax = _get_ax(ax, figsize=(5, 4))
    ax.scatter(es[~sig], neglogp[~sig], s=10, color="0.6", label="ns")
    ax.scatter(es[sig], neglogp[sig], s=14, color=OKABE_ITO[1], label=f"p<{alpha}")
    ax.axhline(-np.log10(alpha), ls="--", lw=0.8, color="k")
    ax.axvline(0, ls="-", lw=0.5, color="0.7")
    ax.set_xlabel("effect size")
    ax.set_ylabel("-log10(p)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    return fig, ax
