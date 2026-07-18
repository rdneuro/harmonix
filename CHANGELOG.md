# Changelog

## 0.1.0

Initial release.

- **Modal bases**: geometric Laplace-Beltrami (default), connectome harmonics,
  graph-Laplacian; mass-orthonormal projection `a = Phi^T M y`.
- **Models**: linear damped wave (Pang/Robinson), linear OU with closed-form
  Lyapunov functional connectivity, nonlinear Hopf/Stuart-Landau with Galerkin
  node-space nonlinearity and optional dealiasing.
- **Simulation**: Euler-Maruyama, stochastic Heun, and ETDRK4 integrators;
  Balloon-Windkessel BOLD; Philox-seeded batched runs and parameter sweeps.
- **Criticality**: spectral abscissa, relaxation time, spectral gap, numerical
  abscissa; classical diagnostics (critical slowing down, susceptibility,
  metastability, branching); spectral working-point finder.
- **Metrics**: static FC, FCD and FCD-KS distance, modal power/timescales/
  spectral entropy/reconstruction curve, Kuramoto synchrony/metastability/LEiDA.
- **Backends**: numpy / torch / jax / cupy behind one array protocol, the
  `n_threads` joblib convention, and VRAM-safe semaphore batching.
- **Visualization**: colorblind-safe 2D panels, FC/FCD heatmaps, the
  shared-colorbar evolutionary heatmap grid, and import-guarded 3D surface
  rendering (yabplot / nilearn / vedo).
