"""Visualization: 2D panels, heatmap grids, and 3D surface rendering.

All 2D helpers use matplotlib/seaborn (colorblind-safe defaults from
:mod:`harmonix.viz._style`); 3D helpers are import-guarded over yabplot / nilearn
/ vedo. Call :func:`apply_style` once to set publication rcParams.
"""
from ._style import (DIVERGING_CMAP, FCD_CMAP, OKABE_ITO, SEQUENTIAL_CMAP,
                     apply_style)
from .criticality import plot_abscissa_curve, plot_susceptibility_curve
from .matrices import evolutionary_heatmap_grid, plot_fc, plot_fcd
from .metrics_panels import (plot_fc_scatter, plot_variance_explained,
                             plot_volcano)
from .surface3d import plot_eigenmode, plot_surface
from .timeseries import (plot_modal_amplitudes, plot_order_parameter,
                         plot_power_spectrum)

__all__ = ["apply_style", "OKABE_ITO", "DIVERGING_CMAP", "SEQUENTIAL_CMAP",
           "FCD_CMAP", "plot_fc", "plot_fcd", "evolutionary_heatmap_grid",
           "plot_modal_amplitudes", "plot_power_spectrum", "plot_order_parameter",
           "plot_variance_explained", "plot_fc_scatter", "plot_volcano",
           "plot_susceptibility_curve", "plot_abscissa_curve",
           "plot_surface", "plot_eigenmode"]
