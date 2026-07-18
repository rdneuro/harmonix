import numpy as np
import pytest

from harmonix.backends import (available_backends, bytes_per_simulation,
                               get_backend, parallel_map, resolve_n_jobs,
                               safe_chunk_size)
from harmonix.backends.rng import (complex_white_noise, spawn_streams,
                                   white_noise)


def test_numpy_backend_primitives():
    bk = get_backend("cpu")
    A = np.random.default_rng(0).standard_normal((6, 6)); A = (A + A.T) / 2
    w, V = bk.eigh(bk.to_device(A))
    assert w.shape == (6,)
    assert bk.expm(bk.to_device(A)).shape == (6, 6)
    x = np.random.randn(64)
    assert np.allclose(bk.to_host(bk.irfft(bk.rfft(bk.to_device(x)), n=64)), x)


def test_complex_preserved():
    bk = get_backend("cpu")
    z = np.random.randn(4) + 1j * np.random.randn(4)
    assert np.iscomplexobj(bk.to_device(z))


def test_rng_reproducible_and_distinct():
    assert np.allclose(white_noise(0, 3, (10,)), white_noise(0, 3, (10,)))
    assert not np.allclose(white_noise(0, 3, (10,)), white_noise(0, 4, (10,)))
    assert len(set(spawn_streams(0, 8).tolist())) == 8


def test_complex_noise_unit_variance():
    z = complex_white_noise(0, 0, (100000,))
    assert abs(np.mean(np.abs(z) ** 2) - 1.0) < 0.02


def test_n_threads_convention():
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(4) == 4
    with pytest.raises(ValueError):
        resolve_n_jobs(0)


def test_parallel_map_matches_serial():
    f = lambda x: x * x
    assert parallel_map(f, [1, 2, 3], 1) == parallel_map(f, [1, 2, 3], 2) == [1, 4, 9]


def test_memory_sizing():
    bk = get_backend("cpu")
    assert bytes_per_simulation(200, 8000) > 0
    assert safe_chunk_size(200, 8000, bk) >= 1


def test_available_backends():
    assert available_backends()["numpy"] is True
