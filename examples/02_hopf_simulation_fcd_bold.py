"""Example 02 -- Nonlinear Hopf simulation, FCD, and BOLD.

Simulate the nonlinear Hopf model in modal space, convert to BOLD via Balloon-
Windkessel, and compute static FC and dynamic FC (FCD).
"""
# %% imports
import numpy as np
import matplotlib.pyplot as plt

import harmonix as hx
from harmonix.models import HopfModal
from harmonix.config import SimConfig
from harmonix.simulate import Simulator
from harmonix.metrics import functional_connectivity, fcd_matrix
from harmonix.viz import apply_style, plot_fc, plot_fcd, plot_modal_amplitudes
apply_style()

# %% model + simulation
basis = hx.datasets.demo_basis(n_modes=100)
model = HopfModal(basis, bifurcation=-0.02, coupling=0.05, intrinsic_freq=0.05,
                  noise=0.02, dealias="filter")
cfg = SimConfig(n_modes=basis.n_modes, dt=0.1, duration=600, warmup=60,
                integrator="heun", tr=0.72, seed=0)
res = Simulator(model, cfg).run(store_nodes=True, bold=True)
print(res)

# %% modal amplitudes over time
fig, ax = plot_modal_amplitudes(res.modal_timeseries, times=res.times, n_show=6)
plt.show()

# %% static FC from BOLD
fc = functional_connectivity(res.bold)
fig, ax = plot_fc(fc, title="Simulated FC (Hopf -> BOLD)")
plt.show()

# %% dynamic FC (FCD)
F = fcd_matrix(res.bold, window=40, step=3)
fig, ax = plot_fcd(F, title="FCD (Hopf -> BOLD)")
plt.show()
