"""Graph-Laplacian eigenmodes of a (structural) connectome.

Eigenvectors of a connectome graph Laplacian -- the "structural eigenmodes" used
in graph-signal-processing of brain activity (Preti & Van De Ville 2019). These
are Euclidean-orthonormal (mass = None). Supports combinatorial, symmetric-
normalized, and random-walk Laplacians; the symmetric-normalized form is the
default (well-conditioned, bounded spectrum in [0, 2]).
"""
from __future__ import annotations

import numpy as np

from .modal_basis import ModalBasis


def graph_laplacian_modes(adjacency, n_modes: int = 200, normalization: str = "symmetric",
                          name: str = "graph") -> ModalBasis:
    """Compute graph-Laplacian eigenmodes of a weighted connectome.

    Parameters
    ----------
    adjacency : np.ndarray, shape (N, N)
        Symmetric non-negative connectivity matrix.
    n_modes : int
        Number of eigenmodes (lowest graph frequencies).
    normalization : {'combinatorial', 'symmetric', 'random_walk'}
        'combinatorial' L = D - A; 'symmetric' L = I - D^-1/2 A D^-1/2;
        'random_walk' L = I - D^-1 A (symmetrized for a real spectrum).
    name : str

    Returns
    -------
    ModalBasis, Euclidean-orthonormal (mass=None).
    """
    A = np.asarray(adjacency, dtype=np.float64)
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    deg = A.sum(axis=1)
    if normalization == "combinatorial":
        L = np.diag(deg) - A
    elif normalization == "symmetric":
        dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
        L = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
    elif normalization == "random_walk":
        dinv = 1.0 / np.clip(deg, 1e-12, None)
        P = dinv[:, None] * A
        L = np.eye(n) - 0.5 * (P + P.T)
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    L = 0.5 * (L + L.T)
    evals, evecs = np.linalg.eigh(L)
    k = min(n_modes, n)
    return ModalBasis(evecs[:, :k], evals[:k], mass=None, kind="graph", name=name)
