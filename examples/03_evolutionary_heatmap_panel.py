"""Example 03 -- The evolutionary heatmap grid.

Show how FC reorganizes as the global coupling G increases, as a single 3x4 grid
of heatmaps sharing axes and one colorbar.
"""
# %% imports
import numpy as np
import matplotlib.pyplot as plt

import harmonix as hx
from harmonix.viz import apply_style, evolutionary_heatmap_grid
apply_style()

# %% analytic FC across a coupling sweep (fast, no simulation)
basis = hx.datasets.demo_basis(n_modes=100)
couplings = np.linspace(0.0, 0.55, 12)
fcs = [hx.analytic_fc(basis, bifurcation=-0.05, coupling=float(G)) for G in couplings]

# %% one 3x4 grid, shared axes + single colorbar, fixed color limits
fig, axes = evolutionary_heatmap_grid(
    fcs, nrows=3, ncols=4,
    titles=[f"G = {G:.2f}" for G in couplings],
    vmin=-1, vmax=1, cbar_label="FC (r)",
    suptitle="Functional connectivity vs global coupling",
)
plt.show()
