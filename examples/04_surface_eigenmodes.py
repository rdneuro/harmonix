"""Example 04 -- Render geometric eigenmodes on a cortical surface (3D).

Requires a 3D backend (yabplot / nilearn / vedo): pip install harmonix[viz].
"""
# %% imports
import harmonix as hx
from harmonix.viz import plot_surface

# %% basis + surface
V, F = hx.datasets.demo_surface(subdivisions=3)
basis = hx.geometric_modes(V, F, n_modes=50)

# %% render a few eigenmodes (offscreen-capable for headless movie export)
for mode in (1, 5, 12):
    plot_surface(V, F, basis.Phi[:, mode], title=f"geometric eigenmode {mode}",
                 offscreen=True, outfile=f"eigenmode_{mode}.png")
