"""Connectome harmonics (Atasoy et al. 2016).

Eigenvectors of the symmetric normalized graph Laplacian of a dense structural
connectome (local surface-mesh adjacency + long-range tractography edges).
Formally a graph-Laplacian basis with the connectome-harmonic convention; this
thin wrapper documents the provenance and builds the combined adjacency.
"""
from __future__ import annotations


import numpy as np

from .graph_laplacian import graph_laplacian_modes
from .modal_basis import ModalBasis


def connectome_harmonics(adjacency=None, local_adjacency=None, long_range=None,
                         n_modes: int = 200, name: str = "connectome") -> ModalBasis:
    """Compute connectome harmonics from a dense structural connectome.

    Parameters
    ----------
    adjacency : np.ndarray, shape (N, N), optional
        Full combined connectome adjacency. If given, ``local_adjacency`` and
        ``long_range`` are ignored.
    local_adjacency : np.ndarray, optional
        Local surface-mesh adjacency (nearest-neighbour cortical edges).
    long_range : np.ndarray, optional
        Long-range tractography adjacency; summed with ``local_adjacency``.
    n_modes : int
    name : str

    Returns
    -------
    ModalBasis (symmetric-normalized graph Laplacian; connectome-harmonic kind).
    """
    if adjacency is None:
        if local_adjacency is None:
            raise ValueError("provide adjacency= or local_adjacency= (+ long_range=)")
        A = np.asarray(local_adjacency, dtype=np.float64)
        if long_range is not None:
            A = A + np.asarray(long_range, dtype=np.float64)
    else:
        A = np.asarray(adjacency, dtype=np.float64)
    basis = graph_laplacian_modes(A, n_modes=n_modes, normalization="symmetric",
                                  name=name)
    basis.kind = "connectome"
    return basis
