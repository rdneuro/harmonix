"""Example 01 -- Closed-form FC and the spectral working point (no simulation).

The low-risk MVP path: build a geometric LBO basis, get the analytic Lyapunov FC
of the linear model, and locate the near-critical coupling from the spectral
abscissa -- all without time-stepping.
"""
# %% imports
import numpy as np
import matplotlib.pyplot as plt

import harmonix as hx
from harmonix.models import OUModal, HopfModal
from harmonix.criticality import spectral_abscissa, working_point_spectral
from harmonix.viz import apply_style, plot_fc, plot_abscissa_curve
apply_style()

# %% basis (use your own cortical mesh via hx.geometric_modes(vertices, faces))
basis = hx.datasets.demo_basis(n_modes=120)
print(basis)

# %% analytic functional connectivity -- closed form, no simulation
model = OUModal(basis, bifurcation=-0.02, coupling=0.1, intrinsic_freq=0.05, noise=0.02)
fc = model.functional_connectivity()
fig, ax = plot_fc(fc, title="Analytic FC (linear model)")
plt.show()

# %% spectral working point: sweep coupling, read the abscissa
make = lambda G: HopfModal(basis, bifurcation=-0.05, coupling=G, intrinsic_freq=0.05)
couplings = np.linspace(0.0, 0.4, 41)
G_star, abscissas = working_point_spectral(make, couplings, target_abscissa=-1e-3)
print(f"near-critical coupling G* = {G_star:.3f}")

fig, ax = plot_abscissa_curve(couplings, abscissas)
ax.axvline(G_star, ls=":", color="k")
plt.show()

# %% analytic power spectra of the first few modes
freqs = np.linspace(0.005, 0.2, 400)
psd = model.power_spectrum(freqs)
fig, ax = plt.subplots(figsize=(6, 3.5))
for k in range(1, 6):
    ax.loglog(freqs, psd[k], label=f"mode {k}")
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power"); ax.legend(frameon=False)
ax.set_title("Analytic modal power spectra")
plt.show()
