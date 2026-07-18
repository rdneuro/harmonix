"""Synthetic datasets and demo bases for testing and examples."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..bases import ModalBasis, geometric_modes, icosphere


def demo_basis(n_modes: int = 100, subdivisions: int = 3) -> ModalBasis:
    """A geometric LBO basis on an icosphere, for tests and examples."""
    V, F = icosphere(subdivisions)
    return geometric_modes(V, F, n_modes=n_modes, name="icosphere-demo")


def demo_surface(subdivisions: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) of the demo icosphere surface."""
    return icosphere(subdivisions)


def synthetic_connectome(n_nodes: int = 100, density: float = 0.1,
                         seed: int = 0) -> np.ndarray:
    """A random symmetric non-negative connectome for graph/connectome bases."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n_nodes, n_nodes)) < density).astype(float)
    A = np.maximum(A, A.T)
    A *= rng.random((n_nodes, n_nodes))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


__all__ = ["demo_basis", "demo_surface", "synthetic_connectome"]
