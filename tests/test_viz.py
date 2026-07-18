import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from harmonix.viz import (apply_style, evolutionary_heatmap_grid, plot_abscissa_curve,
                         plot_fc, plot_fc_scatter, plot_fcd, plot_modal_amplitudes,
                         plot_order_parameter, plot_power_spectrum,
                         plot_susceptibility_curve, plot_variance_explained,
                         plot_volcano)


def _fc(n=30, seed=0):
    rng = np.random.default_rng(seed)
    fc = rng.standard_normal((n, n)); fc = (fc + fc.T) / 2
    np.fill_diagonal(fc, 1); return np.clip(fc / np.abs(fc).max(), -1, 1)


def test_apply_style():
    apply_style()


def test_fc_and_fcd_plots():
    for fn in (plot_fc, plot_fcd):
        f, ax = fn(_fc()); plt.close(f)


def test_evolutionary_grid_single_colorbar():
    mats = [_fc(seed=i) for i in range(12)]
    f, axes = evolutionary_heatmap_grid(mats, 3, 4, vmin=-1, vmax=1)
    n_cbar = sum(1 for a in f.axes if a.get_label() == "<colorbar>")
    assert n_cbar == 1
    plt.close(f)


def test_grid_capacity_check():
    with pytest.raises(ValueError):
        evolutionary_heatmap_grid([_fc() for _ in range(13)], 3, 4)


def test_line_and_panel_plots():
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((8, 100)) + 1j * rng.standard_normal((8, 100))
    for f, ax in (plot_modal_amplitudes(ts),
                  plot_power_spectrum(np.linspace(0.01, 0.2, 50), rng.random((8, 50))),
                  plot_order_parameter(rng.random(100) * 0.4 + 0.3),
                  plot_variance_explained(rng.random(20) ** 2),
                  plot_fc_scatter(_fc(), _fc(seed=1)),
                  plot_volcano(rng.standard_normal(40), rng.random(40)),
                  plot_susceptibility_curve(np.linspace(0, 0.3, 10), rng.random(10)),
                  plot_abscissa_curve(np.linspace(0, 0.3, 10), np.linspace(-0.1, 0.02, 10))):
        plt.close(f)


def test_api_smoke():
    import harmonix as hx
    basis = hx.datasets.demo_basis(n_modes=40)
    assert hx.analytic_fc(basis, coupling=0.1).shape[0] == basis.n_nodes
