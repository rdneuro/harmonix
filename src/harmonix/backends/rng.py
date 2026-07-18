"""Counter-based (Philox) RNG for reproducible parallel stochastic integration.

Each independent simulation (parameter set, seed, subject) draws from its own
Philox stream derived from a ``(seed, stream)`` pair, so a given simulation is
bit-reproducible regardless of batch size, execution order, or device
(Salmon et al. 2011). This is essential when sweeping thousands of trajectories
in parallel on CPU (joblib) or GPU (vmap).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def philox_generator(seed: int, stream: int = 0) -> np.random.Generator:
    """A NumPy ``Generator`` backed by a Philox bit generator for one stream."""
    return np.random.Generator(np.random.Philox(key=int(seed), counter=int(stream)))


def spawn_streams(seed: int, n: int) -> np.ndarray:
    """Deterministically spawn ``n`` independent integer seeds from a base seed."""
    ss = np.random.SeedSequence(int(seed))
    return np.array([int(c.generate_state(1)[0]) for c in ss.spawn(n)],
                    dtype=np.uint64)


def white_noise(seed: int, stream: int, shape: Tuple[int, ...],
                dtype=np.float64) -> np.ndarray:
    """Standard-normal white-noise increments for one simulation stream."""
    return philox_generator(seed, stream).standard_normal(size=shape).astype(dtype)


def complex_white_noise(seed: int, stream: int, shape: Tuple[int, ...]
                        ) -> np.ndarray:
    """Circularly-symmetric complex Gaussian noise (for Hopf oscillators)."""
    gen = philox_generator(seed, stream)
    re = gen.standard_normal(size=shape)
    im = gen.standard_normal(size=shape)
    return (re + 1j * im) / np.sqrt(2.0)
