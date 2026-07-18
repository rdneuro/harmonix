# harmonix

**Modal whole-brain simulation** — evolve the *amplitudes of eigenmodes*, not node-wise ODEs.

harmonix expands brain activity in an eigenbasis,

$$ y(x, t) = \sum_k a_k(t)\, \psi_k(x), $$

and *simulates by evolving the modal coefficients* $a_k(t)$. The eigenmodes $\psi_k$ can be **geometric** (Laplace–Beltrami on the cortical surface; Pang et al. 2023), **connectome harmonics** (Atasoy et al. 2016), or **graph-Laplacian** eigenvectors of a connectome. Working in the eigenbasis is not a cosmetic change: the linear dynamics *diagonalize exactly*, so functional connectivity has a **closed form** (a Lyapunov equation — no time-stepping), the nonlinear Hopf term reduces to a Galerkin node-space round-trip, and **criticality becomes a spectral quantity** (distance-to-criticality = spectral abscissa, relaxation time = $-1/\text{abscissa}$).

---

## Why modal?

A classical whole-brain model integrates $N$ coupled neural-mass ODEs on the nodes. harmonix instead evolves a handful of mode amplitudes. Three consequences:

1. **The linear operator is diagonal.** Coupling through the connectome/geometry Laplacian $L$ contributes $-G\lambda_k$ to each mode, so the linear stochastic (Ornstein–Uhlenbeck / linearized-Hopf) model has an analytic stationary covariance $\text{Cov} = \Phi\,\text{diag}(\text{Var}_k)\,\Phi^\top$. **FC without simulation.** (Linear models describe macroscale fMRI best; Nozari et al. 2024.)
2. **The nonlinearity is local.** The cubic Stuart–Landau term $|z|^2 z$ does not diagonalize, but it *is* pointwise in node space — so harmonix reconstructs $z = \Phi a$, evaluates $|z|^2z$, and projects back ($-\Phi^\top M(|z|^2 z)$). Standard Galerkin pseudo-spectral integration, with optional dealiasing.
3. **Criticality is spectral.** Linearize and read the Jacobian: the largest real eigenvalue (**spectral abscissa**) is the distance to the bifurcation, the slowest-relaxing mode is the critical one, and the working point is where the abscissa approaches zero from below.

harmonix supports **both validation routes**: BOLD/FCD (Balloon–Windkessel hemodynamics, FCD with the Kolmogorov–Smirnov objective) and modal-agnostic diagnostics (modal power spectra, timescales, spectral entropy, reconstruction curves).

> **On the mass matrix.** LBO finite-element eigenmodes are *mass*-orthonormal, $\Phi^\top M \Phi = I$, not Euclidean-orthonormal. The correct projection is $a = \Phi^\top M y$. harmonix carries $M$ explicitly and defaults to the mass-weighted projection — using the Euclidean $\Phi^\top y$ silently rescales every coefficient.

---

## Installation

```bash
git clone https://github.com/rdneuro/harmonix && cd harmonix && pip install -e .'[all]'
```

or

```bash
pip install git+https://github.com/rdneuro/harmonix.git
```

The core depends only on numpy/scipy/joblib. Optional extras: `torch`, `jax`, `cupy` (GPU backends), `viz` (matplotlib/seaborn/nilearn), `lbo` (lapy for exact geometric eigenmodes).

---

## Quick start

```python
import harmonix as hx

# a geometric Laplace-Beltrami basis (here on a demo icosphere; use your own mesh)
basis = hx.datasets.demo_basis(n_modes=100)

# 1) closed-form functional connectivity of the linear model -- NO simulation
fc = hx.analytic_fc(basis, bifurcation=-0.02, coupling=0.1)

# 2) simulated FC of the nonlinear Hopf model
fc_sim = hx.simulate_fc(basis, model="hopf", coupling=0.05, duration=600)

# 3) simulated FC through a Balloon-Windkessel BOLD forward model
fc_bold = hx.simulate_fc(basis, model="ou", coupling=0.1, bold=True)
```

### Bring your own cortical mesh

```python
import harmonix as hx
# vertices (V, 3) and faces (F, 3) from FreeSurfer / fsLR
basis = hx.geometric_modes(vertices, faces, n_modes=200)   # Pang default N=200
```

### Connectome harmonics or graph-Laplacian modes

```python
basis_ch = hx.connectome_harmonics(adjacency=structural_connectome, n_modes=200)
basis_gl = hx.graph_laplacian_modes(structural_connectome, normalization="symmetric")
```

---

## The models

| Model | Class | Use |
|-------|-------|-----|
| Linear damped wave | `LinearWave` | Pang/Robinson neural field; analytic Lorentzian spectra |
| Linear OU / linearized Hopf | `OUModal` | **closed-form Lyapunov FC**, analytic spectra, the MVP core |
| Nonlinear Hopf/Stuart–Landau | `HopfModal` | Galerkin nonlinearity, criticality via the Jacobian |

```python
from harmonix.models import OUModal

model = OUModal(basis, bifurcation=-0.02, coupling=0.1, intrinsic_freq=0.05, noise=0.02)
fc   = model.functional_connectivity()      # analytic
psd  = model.power_spectrum(freqs)          # per-mode Lorentzian
tau  = model.relaxation_times()             # per-mode timescale
```

---

## Simulation, integrators, backends

```python
from harmonix import Simulator, SimConfig
from harmonix.models import HopfModal

cfg = SimConfig(dt=0.1, duration=600, warmup=60, integrator="heun",  # or "euler"/"etdrk4"
                backend="cpu", n_threads=1, seed=0)
sim = Simulator(HopfModal(basis, coupling=0.05), cfg)

res  = sim.run(store_nodes=True, bold=True)   # one trajectory
runs = sim.run_batch(n_runs=50, n_threads=-1) # batched over seeds, all cores
```

- **Integrators**: Euler–Maruyama, stochastic Heun (default), and **ETDRK4** (exponential integrator — free here because the linear operator is diagonal).
- **Backends**: `cpu` (numpy), `torch`, `jax`, `cupy` behind one array protocol. The time loop is sequential; scale-out comes from batching independent runs.
- **`n_threads` convention**: `1` serial (default), `>=2` that many joblib workers, `-1` all cores. BLAS threads are pinned per worker to avoid oversubscription.
- **Memory**: GPU sweeps are chunked to fit VRAM and guarded by a device semaphore.

---

## Criticality

```python
from harmonix.criticality import (spectral_abscissa, relaxation_time,
                                  working_point_spectral)

alpha = spectral_abscissa(model)        # distance to bifurcation (<0 stable)
tau   = relaxation_time(model)          # -1/alpha (diverges at criticality)

# locate the near-critical coupling
make = lambda G: HopfModal(basis, bifurcation=-0.05, coupling=G)
G_star, abscissas = working_point_spectral(make, couplings=np.linspace(0, 0.3, 31))
```

Also available: `spectral_gap`, `numerical_abscissa` (non-normal reactivity), and empirical diagnostics `susceptibility`, `lag1_autocorrelation`, `metastability`, `branching_parameter`.

---

## Metrics (both validation routes)

```python
from harmonix.metrics import (functional_connectivity, fc_similarity,
                              fcd_matrix, fcd_ks_distance,        # BOLD/FCD route
                              modal_power_spectrum, modal_timescales,
                              spectral_entropy, reconstruction_curve,  # modal route
                              synchrony, metastability, leida_leading_eigenvector)
```

---

## Visualization

```python
import harmonix as hx
from harmonix.viz import (apply_style, plot_fc, plot_fcd, evolutionary_heatmap_grid,
                         plot_modal_amplitudes, plot_power_spectrum,
                         plot_fc_scatter, plot_volcano, plot_abscissa_curve,
                         plot_surface)
apply_style()

# the flagship: a 3x4 grid of FC snapshots, shared axes + a single colorbar
fig, axes = evolutionary_heatmap_grid(fc_snapshots, nrows=3, ncols=4,
                                      titles=[...], vmin=-1, vmax=1, cbar_label="r")

# 3D activity on a cortical surface (yabplot / nilearn / vedo, import-guarded)
plot_surface(vertices, faces, basis.Phi[:, 5], title="eigenmode 5")
```

Colorblind-safe (Okabe–Ito) palettes throughout; 2D panels for lines, bars, scatter, volcano, and heatmaps; 3D surface rendering with offscreen/headless support.

---

## Design notes & caveats

- The analytic linear core is the low-risk default; the nonlinear Hopf model is powerful but its **mode-truncation + cubic term can alias** — use `dealias="filter"` (default) or `"two_thirds"` and check mode convergence.
- "Distance-to-criticality = spectral gap" is exact for the *linearized* system; for the full nonlinear model it is a well-motivated heuristic.
- Geometric vs connectome eigenmodes is an open debate — harmonix supports both so you can test on your own data.

## References

Pang et al. 2023 *Nature* (geometric eigenmodes) · Atasoy et al. 2016 *Nat Commun* (connectome harmonics) · Deco et al. 2017 (Hopf whole-brain) · Ponce-Alvarez & Deco 2024 (linear Hopf / Lyapunov FC) · Nozari et al. 2024 *Nat Biomed Eng* (linearity of macroscale dynamics) · Cox & Matthews 2002 / Kassam & Trefethen 2005 (ETDRK4) · Friston 2000 / Stephan 2007 (Balloon–Windkessel).

## License

MIT © 2026 Rodrigo Debona
