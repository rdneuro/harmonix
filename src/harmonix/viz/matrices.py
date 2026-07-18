"""Matrix visualizations: FC/FCD heatmaps and the evolutionary heatmap grid.

The flagship is :func:`evolutionary_heatmap_grid`: a grid (e.g. 3x4 or 4x3) of
matrices at successive stages/parameters, sharing x/y axes and a SINGLE colorbar
with fixed ``vmin``/``vmax`` so the panels are directly comparable -- exactly the
"partial results across processing time" figure requested.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._style import DIVERGING_CMAP, FCD_CMAP, _get_ax


def plot_fc(fc, ax=None, cmap: str = DIVERGING_CMAP, vmin: float = -1.0,
            vmax: float = 1.0, title: str = "Functional connectivity",
            colorbar: bool = True):
    """Plot a functional-connectivity matrix as a heatmap."""
    fig, ax = _get_ax(ax, figsize=(5, 4.2))
    im = ax.imshow(np.asarray(fc), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("node")
    ax.set_ylabel("node")
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")
    return fig, ax


def plot_fcd(fcd, ax=None, cmap: str = FCD_CMAP, title: str = "FCD",
             colorbar: bool = True):
    """Plot a functional-connectivity-dynamics matrix."""
    fig, ax = _get_ax(ax, figsize=(5, 4.2))
    im = ax.imshow(np.asarray(fcd), cmap=cmap, vmin=-1, vmax=1,
                   aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("window")
    ax.set_ylabel("window")
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="FC correlation")
    return fig, ax


def evolutionary_heatmap_grid(matrices: Sequence[np.ndarray], nrows: int = 3,
                              ncols: int = 4, titles: Optional[Sequence[str]] = None,
                              cmap: str = DIVERGING_CMAP, vmin: Optional[float] = None,
                              vmax: Optional[float] = None,
                              suptitle: str = "Evolution",
                              cbar_label: str = "value", figsize=None):
    """Grid of heatmaps with shared axes and a single shared colorbar.

    Parameters
    ----------
    matrices : sequence of 2-D arrays
        Up to ``nrows*ncols`` matrices, shown in row-major order.
    nrows, ncols : int
        Grid shape (e.g. 3x4 or 4x3).
    titles : sequence of str, optional
        Per-panel titles (e.g. stage or parameter value).
    cmap : str
    vmin, vmax : float, optional
        Shared colour limits; if None, taken from the global min/max across all
        matrices so panels are directly comparable.
    suptitle : str
    cbar_label : str
    figsize : tuple, optional

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    mats = [np.asarray(m) for m in matrices]
    if len(mats) > nrows * ncols:
        raise ValueError(f"{len(mats)} matrices exceed grid capacity {nrows*ncols}")
    if vmin is None:
        vmin = min(float(m.min()) for m in mats)
    if vmax is None:
        vmax = max(float(m.max()) for m in mats)
    if figsize is None:
        figsize = (2.6 * ncols, 2.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True,
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    im = None
    for i, ax in enumerate(axes):
        if i < len(mats):
            im = ax.imshow(mats[i], cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="equal", interpolation="nearest")
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
        else:
            ax.axis("off")
    # single shared colorbar spanning all panels
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label=cbar_label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig, axes
