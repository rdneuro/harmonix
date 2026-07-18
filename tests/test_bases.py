import numpy as np
import pytest

from harmonix.bases import (ModalBasis, connectome_harmonics, geometric_modes,
                            graph_laplacian_modes, icosphere)


def test_mass_orthonormal(demo_basis):
    assert demo_basis.is_orthonormal()


def test_first_eigenvalue_zero(demo_basis):
    assert abs(demo_basis.eigenvalues[0]) < 1e-6


def test_projection_reconstruction_roundtrip(demo_basis):
    rng = np.random.default_rng(0)
    y = demo_basis.Phi[:, 1:15] @ rng.standard_normal(14)
    yhat = demo_basis.reconstruct(demo_basis.project(y))
    assert np.corrcoef(y, yhat)[0, 1] > 0.999


def test_complex_projection_preserved(demo_basis):
    z = demo_basis.Phi[:, :5] @ (np.arange(5) + 1j * np.arange(5))
    a = demo_basis.project(z)
    assert np.iscomplexobj(a)


def test_mass_weighting_matters(demo_basis):
    rng = np.random.default_rng(1)
    y = demo_basis.Phi[:, 1:10] @ rng.standard_normal(9)
    a_mass = demo_basis.project(y)
    a_euclid = demo_basis.Phi.T @ y
    assert not np.allclose(a_mass, a_euclid)


def test_graph_basis_euclidean():
    rng = np.random.default_rng(2)
    A = (rng.random((50, 50)) < 0.1).astype(float); A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    gl = graph_laplacian_modes(A, n_modes=30)
    assert gl.mass is None
    assert gl.is_orthonormal()


def test_connectome_spectrum_bounded():
    rng = np.random.default_rng(3)
    A = (rng.random((50, 50)) < 0.1).astype(float); A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    ch = connectome_harmonics(adjacency=A, n_modes=30)
    assert ch.eigenvalues.min() >= -1e-9 and ch.eigenvalues.max() <= 2.01


def test_truncate(demo_basis):
    assert demo_basis.truncate(20).n_modes == 20


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        ModalBasis(np.zeros((10, 5)), np.zeros(4))
