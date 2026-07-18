import numpy as np
import pytest

from harmonix.config import SimConfig
from harmonix.models import HopfModal, OUModal
from harmonix.simulate import Simulator


def test_analytic_fc_matches_simulation(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    cfg = SimConfig(n_modes=demo_basis.n_modes, dt=0.1, duration=600, warmup=40, seed=0)
    res = Simulator(m, cfg).run(store_nodes=True)
    fc_sim = np.corrcoef(res.reconstruct_nodes())
    fc_ana = m.functional_connectivity()
    iu = np.triu_indices(demo_basis.n_nodes, 1)
    assert np.corrcoef(fc_sim[iu], fc_ana[iu])[0, 1] > 0.85


def test_hopf_bounded_below_bifurcation(demo_basis):
    h = HopfModal(demo_basis, bifurcation=-0.02, coupling=0.05, noise=0.02)
    res = Simulator(h, SimConfig(n_modes=demo_basis.n_modes, dt=0.1, duration=100,
                                 warmup=20, seed=1)).run()
    assert np.all(np.isfinite(res.modal_timeseries))
    assert np.abs(res.modal_timeseries).max() < 10


def test_bold_finite(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    res = Simulator(m, SimConfig(n_modes=demo_basis.n_modes, dt=0.1, duration=120,
                                 warmup=20, tr=0.72, seed=0)).run(bold=True)
    assert res.bold is not None and np.all(np.isfinite(res.bold))


def test_batch_distinct_seeds(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    sim = Simulator(m, SimConfig(n_modes=demo_basis.n_modes, dt=0.2, duration=60,
                                 warmup=10))
    results = sim.run_batch(n_runs=3, base_seed=0)
    assert len(results) == 3
    assert len({r.meta["seed"] for r in results}) == 3


def test_integrators_run(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    for integ in ("euler", "heun", "etdrk4"):
        cfg = SimConfig(n_modes=demo_basis.n_modes, dt=0.1, duration=40, warmup=10,
                        integrator=integ)
        res = Simulator(m, cfg).run()
        assert np.all(np.isfinite(res.modal_timeseries))


def test_sweep_analytic(demo_basis):
    from harmonix.criticality import spectral_abscissa
    from harmonix.simulate import sweep
    make = lambda G: HopfModal(demo_basis, bifurcation=-0.05, coupling=G)
    out = sweep(make, [0.0, 0.1, 0.2], spectral_abscissa, analytic=True)
    assert out.shape == (3,)
    assert np.all(out < 0)


def test_sweep_simulated(demo_basis):
    from harmonix.metrics import synchrony
    from harmonix.simulate import sweep
    from harmonix.config import SimConfig
    make = lambda a: OUModal(demo_basis, bifurcation=a, coupling=0.1, noise=0.05)
    cfg = SimConfig(n_modes=demo_basis.n_modes, dt=0.2, duration=60, warmup=10)
    out = sweep(make, [-0.1, -0.05], lambda r: synchrony(r.reconstruct_nodes()),
                config=cfg, analytic=False)
    assert out.shape == (2,)
    assert np.all((out >= 0) & (out <= 1))


def test_vectorized_batch_matches_analytic(demo_basis):
    from harmonix.config import SimConfig
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    cfg = SimConfig(n_modes=demo_basis.n_modes, dt=0.2, duration=400, warmup=40,
                    integrator="heun", seed=0)
    node = Simulator(m, cfg).run_batch_vectorized(n_runs=6, base_seed=0)
    assert node.shape[0] == 6
    fc_mean = np.mean([np.corrcoef(node[r]) for r in range(6)], axis=0)
    iu = np.triu_indices(demo_basis.n_nodes, 1)
    assert np.corrcoef(fc_mean[iu], m.functional_connectivity()[iu])[0, 1] > 0.9


def test_vectorized_batch_hopf_bounded(demo_basis):
    from harmonix.config import SimConfig
    h = HopfModal(demo_basis, bifurcation=-0.02, coupling=0.05, noise=0.02)
    cfg = SimConfig(n_modes=demo_basis.n_modes, dt=0.1, duration=100, warmup=20)
    node, modal = Simulator(h, cfg).run_batch_vectorized(n_runs=3, return_modal=True)
    assert np.all(np.isfinite(node)) and np.abs(modal).max() < 10
    assert not np.allclose(node[0], node[1])
