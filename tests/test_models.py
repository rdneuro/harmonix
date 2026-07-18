import numpy as np
import pytest

from harmonix.models import HopfModal, LinearWave, OUModal


def test_ou_analytic_variance_positive(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    assert np.all(m.modal_variance() > 0)


def test_ou_fc_wellformed(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    fc = m.functional_connectivity()
    assert np.allclose(fc, fc.T)
    assert np.allclose(np.diag(fc), 1.0)
    assert fc.min() >= -1.001 and fc.max() <= 1.001


def test_lyapunov_matches_diagonal(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.05, coupling=0.1, noise=0.05)
    cov_diag = m.stationary_covariance()
    cov_lyap = np.real(demo_basis.Phi @ m.lyapunov_covariance() @ demo_basis.Phi.T)
    assert np.allclose(cov_diag, cov_lyap, atol=1e-8)


def test_ou_unstable_raises(demo_basis):
    m = OUModal(demo_basis, bifurcation=-0.01, coupling=-1.0, noise=0.05)  # negative coupling destabilizes
    with pytest.raises(ValueError):
        m.modal_variance()


def test_ou_rejects_positive_bifurcation(demo_basis):
    with pytest.raises(ValueError):
        OUModal(demo_basis, bifurcation=0.01)


def test_wave_stiffness(demo_basis):
    w = LinearWave(demo_basis, gamma_s=116.0, r_s=30.0)
    assert np.all(w.stiffness() >= 1.0 - 1e-9)
    assert np.all(w.power_spectrum(np.linspace(0.1, 50, 50)) >= 0)


def test_hopf_jacobian_homogeneous_diagonal(demo_basis):
    h = HopfModal(demo_basis, bifurcation=-0.02, coupling=0.1)
    J = h.modal_jacobian()
    assert np.allclose(J, np.diag(np.diag(J)))
    assert np.all(np.real(np.diag(J)) < 0)


def test_hopf_jacobian_heterogeneous_dense(demo_basis):
    rng = np.random.default_rng(0)
    a = -0.02 + 0.01 * rng.standard_normal(demo_basis.n_nodes)
    h = HopfModal(demo_basis, bifurcation=a, coupling=0.1)
    J = h.modal_jacobian()
    assert not np.allclose(J, np.diag(np.diag(J)))


def test_hopf_cubic_scaling(demo_basis):
    h = HopfModal(demo_basis, bifurcation=-0.02, coupling=0.1)
    rng = np.random.default_rng(0)
    a = 0.01 * (rng.standard_normal(demo_basis.n_modes) + 1j * rng.standard_normal(demo_basis.n_modes))
    ratio = np.abs(h.nonlinear_drift(2 * a)).sum() / np.abs(h.nonlinear_drift(a)).sum()
    assert 7.0 < ratio < 9.0


def test_hopf_nonlinear_drift_complex(demo_basis):
    h = HopfModal(demo_basis, bifurcation=-0.02, coupling=0.1)
    a = 0.01 * (np.arange(demo_basis.n_modes) + 1j)
    assert np.iscomplexobj(h.nonlinear_drift(a))
