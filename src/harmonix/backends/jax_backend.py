"""JAX backend (CPU/GPU/TPU); jit + lax.scan + vmap friendly for the time loop."""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.scipy.linalg as jsla
    _HAS_JAX = True
except ImportError:  # pragma: no cover - optional
    jax = jnp = jsla = None
    _HAS_JAX = False

from .base import Backend

_JDT = {"float64": "float64", "float32": "float32",
        "complex128": "complex128", "complex64": "complex64"}


class JaxBackend(Backend):
    """CPU/GPU/TPU backend backed by JAX (natural winner for the scanned time loop)."""

    name = "jax"
    supports_gpu = True

    def __init__(self, device: Optional[str] = None, dtype: str = "float64"):
        if not _HAS_JAX:
            raise ImportError("JaxBackend requires jax ('pip install harmonix[jax]')")
        self._dtype = _JDT[dtype]

    @property
    def xp(self):
        return jnp

    def to_device(self, x, dtype: Optional[str] = None):
        arr = np.asarray(x)
        if np.iscomplexobj(arr) and dtype is None:
            return jnp.asarray(arr)
        if np.issubdtype(arr.dtype, np.integer) and dtype is None:
            return jnp.asarray(arr)
        return jnp.asarray(arr, dtype=self._dtype if dtype is None else _JDT[dtype])

    def to_host(self, x) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(x))

    def matmul(self, a, b):
        return a @ b

    def eigh(self, a):
        return jnp.linalg.eigh(a)

    def expm(self, a):
        return jsla.expm(a)

    def rfft(self, x, axis: int = -1):
        return jnp.fft.rfft(x, axis=axis)

    def irfft(self, x, n: int, axis: int = -1):
        return jnp.fft.irfft(x, n=n, axis=axis)
