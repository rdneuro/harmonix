"""Publication style: colorblind-safe palette and sensible matplotlib defaults.

The Okabe-Ito qualitative palette is colorblind-safe; diverging maps (FC in
[-1, 1]) use RdBu_r centered at 0, sequential maps use viridis/magma. All figure
helpers accept an ``ax`` for composition and return the Axes/Figure.
"""
from __future__ import annotations


# Okabe-Ito colorblind-safe qualitative palette
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
             "#56B4E9", "#F0E442", "#000000"]

DIVERGING_CMAP = "RdBu_r"       # FC / signed maps, centered at 0
SEQUENTIAL_CMAP = "viridis"     # power, positive quantities
FCD_CMAP = "magma"              # FCD matrices


def apply_style() -> None:
    """Apply harmonix publication defaults to matplotlib rcParams."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "image.cmap": SEQUENTIAL_CMAP,
        "figure.constrained_layout.use": True,
    })


def _get_ax(ax=None, figsize=(5, 4)):
    """Return (fig, ax), creating them if ``ax`` is None."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax
