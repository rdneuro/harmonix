"""The ModalBasis object: an orthonormal eigenbasis for modal simulation.

Everything in harmonix is evaluated in a basis Phi (columns = eigenmodes) with
eigenvalues Lambda and, for finite-element geometric modes, a mass matrix M. The
one invariant that must never be violated:

    LBO finite-element eigenmodes are MASS-orthonormal:  Phi^T M Phi = I,
    NOT Euclidean-orthonormal. The correct projection is therefore

        a = Phi^T M y        (modal coefficients)
        y = Phi a            (reconstruction)

    Using the Euclidean a = Phi^T y silently inflates the coefficient scale by
    ~1/mean(mass) and corrupts every downstream quantity. harmonix carries M
    explicitly and defaults to the mass-weighted projection.

A ``ModalBasis`` is a stateful object (it caches the eigenbasis and derived
operators) -- justified because it is constructed once from an expensive
eigensolve and then reused across thousands of simulations.
"""
from __future__ import annotations


import numpy as np


class ModalBasis:
    """A precomputed eigenbasis (Phi, Lambda, M) with modal projection/reconstruction.

    Parameters
    ----------
    eigenmodes : np.ndarray, shape (n_nodes, n_modes)
        Eigenmodes as columns. For LBO geometric modes these are mass-orthonormal.
    eigenvalues : np.ndarray, shape (n_modes,)
        Ascending eigenvalues (the first is ~0 for the constant mode).
    mass : np.ndarray, shape (n_nodes,) or None
        Diagonal of the FEM mass matrix M for mass-orthonormal bases. ``None``
        means the basis is Euclidean-orthonormal (graph/connectome Laplacian
        eigenvectors) and projection uses M = I.
    kind : str
        Provenance label ('geometric', 'connectome', 'graph', 'custom').
    name : str
        Optional human-readable name.

    Attributes
    ----------
    n_nodes, n_modes : int
    wavelengths : np.ndarray
        Approximate spatial wavelength per mode, 2*pi / sqrt(lambda) (inf for the
        constant mode), following the Weyl/Helmholtz relation.
    """

    def __init__(self, eigenmodes, eigenvalues, mass=None, kind: str = "custom",
                 name: str = ""):
        Phi = np.asarray(eigenmodes, dtype=np.float64)
        lam = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
        if Phi.ndim != 2:
            raise ValueError(f"eigenmodes must be 2-D (n_nodes, n_modes); got {Phi.shape}")
        if Phi.shape[1] != lam.shape[0]:
            raise ValueError(
                f"n_modes mismatch: eigenmodes has {Phi.shape[1]} columns but "
                f"eigenvalues has {lam.shape[0]} entries")
        if np.any(np.diff(lam) < -1e-6 * (np.abs(lam[:-1]) + 1e-12)):
            raise ValueError("eigenvalues must be ascending")
        self.Phi = Phi
        self.eigenvalues = lam
        self.n_nodes, self.n_modes = Phi.shape
        self.kind = kind
        self.name = name

        if mass is None:
            self.mass = None
            self._Mrow = None
        else:
            m = np.asarray(mass, dtype=np.float64).reshape(-1)
            if m.shape[0] != self.n_nodes:
                raise ValueError(
                    f"mass length {m.shape[0]} != n_nodes {self.n_nodes}")
            self.mass = m
            self._Mrow = m[:, None]                           # (n_nodes, 1)

        # approximate wavelengths (Weyl): lambda ~ (2 pi / wavelength)^2
        with np.errstate(divide="ignore"):
            self.wavelengths = np.where(lam > 1e-12,
                                        2.0 * np.pi / np.sqrt(np.clip(lam, 1e-12, None)),
                                        np.inf)

    # ------------------------------------------------------------------ #
    # projection / reconstruction (mass-weighted by construction)
    # ------------------------------------------------------------------ #
    def project(self, field) -> np.ndarray:
        """Modal coefficients a = Phi^T M y of a field on the nodes.

        Parameters
        ----------
        field : np.ndarray, shape (n_nodes,) or (n_nodes, k)
            Field(s) on the mesh nodes.

        Returns
        -------
        np.ndarray, shape (n_modes,) or (n_modes, k)
        """
        y = np.asarray(field)
        if not (np.issubdtype(y.dtype, np.floating)
                or np.issubdtype(y.dtype, np.complexfloating)):
            y = y.astype(np.float64)
        if y.shape[0] != self.n_nodes:
            raise ValueError(f"field first axis {y.shape[0]} != n_nodes {self.n_nodes}")
        My = y if self.mass is None else (self._Mrow * y if y.ndim == 2
                                          else self.mass * y)
        return self.Phi.T @ My

    def reconstruct(self, coefficients) -> np.ndarray:
        """Reconstruct a field y = Phi a from modal coefficients.

        Parameters
        ----------
        coefficients : np.ndarray, shape (n_modes,) or (n_modes, k)

        Returns
        -------
        np.ndarray, shape (n_nodes,) or (n_nodes, k)
        """
        a = np.asarray(coefficients)
        if a.shape[0] != self.n_modes:
            raise ValueError(f"coefficients first axis {a.shape[0]} != n_modes {self.n_modes}")
        return self.Phi @ a

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def is_orthonormal(self, atol: float = 1e-6) -> bool:
        """Check Phi^T M Phi == I (mass-weighted) to tolerance ``atol``."""
        if self.mass is None:
            gram = self.Phi.T @ self.Phi
        else:
            gram = self.Phi.T @ (self._Mrow * self.Phi)
        return np.allclose(gram, np.eye(self.n_modes), atol=atol)

    def reconstruction_accuracy(self, field) -> float:
        """Pearson r between a field and its ``n_modes``-mode reconstruction."""
        y = np.asarray(field, dtype=np.float64).reshape(self.n_nodes, -1)
        yhat = self.reconstruct(self.project(y))
        return float(np.corrcoef(y.ravel(), yhat.ravel())[0, 1])

    def truncate(self, n_modes: int) -> "ModalBasis":
        """Return a new basis keeping the first ``n_modes`` modes."""
        n = min(n_modes, self.n_modes)
        return ModalBasis(self.Phi[:, :n], self.eigenvalues[:n],
                          mass=self.mass, kind=self.kind, name=self.name)

    def __repr__(self):
        return (f"ModalBasis(kind={self.kind!r}, n_nodes={self.n_nodes}, "
                f"n_modes={self.n_modes}, mass={'yes' if self.mass is not None else 'no'})")
