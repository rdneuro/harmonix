"""Geometric (Laplace-Beltrami) eigenmodes on a cortical surface mesh.

The default harmonix basis (Pang et al. 2023). Solves the Helmholtz eigenproblem
Delta psi = -lambda psi on a triangle mesh via the cotangent finite-element
Laplacian with a lumped mass matrix, returning MASS-orthonormal eigenmodes
(psi^T M psi = I). Prefers ``lapy`` when installed (exact parity with the Pang
reference pipeline); otherwise uses this self-contained scipy implementation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .modal_basis import ModalBasis


def _cotangent_laplacian(vertices, faces) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Cotangent stiffness matrix and lumped vertex mass for a triangle mesh."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    n = V.shape[0]
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    # edge vectors
    e0 = V[i2] - V[i1]
    e1 = V[i0] - V[i2]
    e2 = V[i1] - V[i0]
    area = 0.5 * np.linalg.norm(np.cross(e0, -e1), axis=1)     # (F,)
    area = np.clip(area, 1e-12, None)

    def _cot(u, w):
        cross = np.linalg.norm(np.cross(u, w), axis=1)
        return np.sum(u * w, axis=1) / np.clip(cross, 1e-12, None)

    cot0 = _cot(-e1, -e2) / 2.0                                # angle at v0
    cot1 = _cot(-e2, -e0) / 2.0                                # angle at v1
    cot2 = _cot(-e0, -e1) / 2.0                                # angle at v2

    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    W = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
    Woff = sp.csr_matrix((-W, (I, J)), shape=(n, n))
    L = Woff - sp.diags(np.asarray(Woff.sum(axis=1)).ravel())
    L = -L                                                    # positive semidefinite

    mass = np.zeros(n)
    for idx in (i0, i1, i2):
        np.add.at(mass, idx, area / 3.0)
    return L.tocsr(), np.clip(mass, 1e-12, None)


def geometric_modes(vertices, faces, n_modes: int = 200, name: str = "geometric"
                    ) -> ModalBasis:
    """Compute the geometric LBO eigenbasis of a cortical surface.

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3)
        Vertex coordinates (mm).
    faces : np.ndarray, shape (F, 3)
        Triangle indices (0-based).
    n_modes : int
        Number of eigenmodes including the constant mode 0 (default 200, Pang 2023).
    name : str
        Basis name.

    Returns
    -------
    ModalBasis
        Mass-orthonormal geometric eigenbasis.
    """
    try:
        from lapy import Solver, TriaMesh
        tria = TriaMesh(np.asarray(vertices, float), np.asarray(faces, int))
        solver = Solver(tria)
        evals, evecs = solver.eigs(k=n_modes)
        try:
            mass = np.asarray(solver.mass.diagonal(), float)
        except Exception:
            mass = np.ones(np.asarray(evecs).shape[0])
        return ModalBasis(np.asarray(evecs, float), np.asarray(evals, float),
                          mass=mass, kind="geometric", name=name)
    except ImportError:
        pass

    L, mass = _cotangent_laplacian(vertices, faces)
    M = sp.diags(mass)
    k = min(n_modes, L.shape[0] - 2)
    evals, evecs = spla.eigsh(L, k=k, M=M, sigma=-1e-8, which="LM")
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    # enforce mass-orthonormality of columns
    for j in range(evecs.shape[1]):
        nrm = np.sqrt(evecs[:, j] @ (mass * evecs[:, j]))
        evecs[:, j] /= max(nrm, 1e-12)
    return ModalBasis(evecs, evals, mass=mass, kind="geometric", name=name)
