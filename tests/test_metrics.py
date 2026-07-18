import numpy as np
import pytest

from harmonix.config import SimConfig
from harmonix.metrics import (fc_similarity, fcd_ks_distance, fcd_matrix,
                              functional_connectivity, kuramoto_order,
                              metastability, modal_power, modal_power_spectrum,
                              modal_timescales, reconstruction_curve,
                              spectral_entropy, synchrony)
from harmonix.models import OUModal
from harmonix.simulate import Simulator


@pytest.fixture(scope="module")
def sim_signal(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    res = Simulator(m, SimConfig(n_modes=demo_basis.n_modes, dt=0.2, duration=500,
                                 warmup=40)).run(store_nodes=True)
    return res, m


def test_fc_similarity_self_is_one(sim_signal):
    res, _ = sim_signal
    fc = functional_connectivity(res.reconstruct_nodes())
    assert abs(fc_similarity(fc, fc) - 1.0) < 1e-9


def test_fc_matches_analytic(sim_signal):
    res, m = sim_signal
    fc = functional_connectivity(res.reconstruct_nodes())
    assert fc_similarity(fc, m.functional_connectivity()) > 0.9


def test_fcd_symmetric(sim_signal):
    res, _ = sim_signal
    F = fcd_matrix(res.reconstruct_nodes(), window=40, step=5)
    assert np.allclose(F, F.T)


def test_fcd_ks_self_small(sim_signal):
    res, _ = sim_signal
    node = res.reconstruct_nodes()
    assert fcd_ks_distance(node, node, window=40, step=5) < 1e-9


def test_modal_metrics(sim_signal):
    res, _ = sim_signal
    assert modal_power(res.modal_timeseries).shape == (res.n_modes,)
    freqs, psd = modal_power_spectrum(res.modal_timeseries, dt=0.2)
    assert psd.shape[0] == res.n_modes
    assert modal_timescales(res.modal_timeseries, dt=0.2).shape == (res.n_modes,)
    assert spectral_entropy(res.modal_timeseries) > 0


def test_reconstruction_curve_monotone(demo_basis):
    rng = np.random.default_rng(0)
    y = demo_basis.Phi[:, 1:20] @ rng.standard_normal(19)
    rc = reconstruction_curve(y, demo_basis, [5, 10, 20, 40])
    assert np.all(np.diff(rc) >= -1e-9)


def test_phase_metrics_bounded(sim_signal):
    res, _ = sim_signal
    node = res.reconstruct_nodes()
    assert 0 <= synchrony(node) <= 1
    assert metastability(node) >= 0
