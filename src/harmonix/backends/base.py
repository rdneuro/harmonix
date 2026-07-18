"""Backend abstract base: the array/compute primitives modal simulation needs.

Modal whole-brain simulation is dominated by a small set of dense linear-algebra
operations -- mode<->node projections (GEMMs ``Phi @ a`` and ``Phi.T @ (M y)``),
matrix exponentials (exact linear propagator and ETDRK4 coefficients), symmetric
eigendecompositions (Jacobian spectra, basis construction), FFTs (power spectra),
and reductions (functional-connectivity statistics). Rather than reimplement every
model per device, the heavy math is written once against this protocol; concrete
backends (numpy / torch / jax / cupy) supply the array namespace and a few
primitives.

Design doctrine (from the harmonix design report):

* **float64 for the analytic linear core and eigendecompositions** -- Lyapunov
  covariance, Jacobian spectra, and mode ordering within near-degenerate groups
  are numerically delicate. float32 is reserved for large batched stochastic
  integration where speed dominates.
* **The time loop is sequential**; parallelism comes from *batching independent
  simulations* (parameter/seed sweeps), which every backend exposes through the
  same array-leading-axis convention.
* **Counter-based (Philox) RNG** gives per-stream reproducibility independent of
  device and batch size (see :mod:`harmonix.backends.rng`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Backend(ABC):
    """Abstract array/compute backend for modal simulation."""

    name: str = "base"
    supports_gpu: bool = False

    # ------------------------------------------------------------------ #
    # array namespace + host/device movement
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def xp(self):
        """The array namespace (numpy / torch / jax.numpy / cupy)."""

    @abstractmethod
    def to_device(self, x, dtype: Optional[str] = None):
        """Move a host array onto this backend's device with the given dtype.

        Complex dtypes are always preserved (they carry oscillator phase in the
        Hopf model and Fourier coefficients in spectral estimates).
        """

    @abstractmethod
    def to_host(self, x) -> np.ndarray:
        """Return a contiguous host ``numpy.ndarray`` copy of ``x``."""

    # ------------------------------------------------------------------ #
    # dense linear algebra primitives
    # ------------------------------------------------------------------ #
    @abstractmethod
    def matmul(self, a, b):
        """Matrix product ``a @ b`` on device."""

    @abstractmethod
    def eigh(self, a):
        """Symmetric/Hermitian eigendecomposition; ascending eigenvalues."""

    @abstractmethod
    def expm(self, a):
        """Dense matrix exponential of a 2-D array."""

    @abstractmethod
    def rfft(self, x, axis: int = -1):
        """Real FFT along ``axis``."""

    @abstractmethod
    def irfft(self, x, n: int, axis: int = -1):
        """Inverse real FFT along ``axis`` returning ``n`` real samples."""

    def empty_cache(self) -> None:
        """Release cached device memory if supported (no-op by default)."""

    # ------------------------------------------------------------------ #
    # convenience shared across backends
    # ------------------------------------------------------------------ #
    def asarray(self, x, dtype: Optional[str] = None):
        """Alias for :meth:`to_device` (namespace-friendly)."""
        return self.to_device(x, dtype=dtype)

    def zeros(self, shape, dtype: str = "float64"):
        """Allocate a zero array on device."""
        return self.to_device(np.zeros(shape), dtype=dtype)
