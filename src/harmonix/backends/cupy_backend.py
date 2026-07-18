"""CuPy backend (CUDA GPU) for large non-differentiable batched simulation."""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.linalg as csla
    _HAS_CUPY = True
except ImportError:  # pragma: no cover - optional
    cp = csla = None
    _HAS_CUPY = False

from .base import Backend

_CDT = {"float64": "float64", "float32": "float32",
        "complex128": "complex128", "complex64": "complex64"}


class CupyBackend(Backend):
    """CUDA GPU backend backed by CuPy."""

    name = "cupy"
    supports_gpu = True

    def __init__(self, device: Optional[int] = None, dtype: str = "float64"):
        if not _HAS_CUPY:
            raise ImportError("CupyBackend requires cupy ('pip install harmonix[cupy]')")
        self._dtype = _CDT[dtype]
        self._device_id = 0 if device is None else int(device)

    @property
    def xp(self):
        return cp

    def to_device(self, x, dtype: Optional[str] = None):
        arr = np.asarray(x)
        with cp.cuda.Device(self._device_id):
            if (np.iscomplexobj(arr) or np.issubdtype(arr.dtype, np.integer)) and dtype is None:
                return cp.asarray(arr)
            return cp.asarray(arr, dtype=self._dtype if dtype is None else _CDT[dtype])

    def to_host(self, x) -> np.ndarray:
        return np.ascontiguousarray(cp.asnumpy(x))

    def matmul(self, a, b):
        return a @ b

    def eigh(self, a):
        return cp.linalg.eigh(a)

    def expm(self, a):
        try:
            return csla.expm(a)
        except Exception:
            # fallback: eigendecomposition-based expm for diagonalizable a
            w, V = cp.linalg.eig(a)
            return (V * cp.exp(w)) @ cp.linalg.inv(V)

    def rfft(self, x, axis: int = -1):
        return cp.fft.rfft(x, axis=axis)

    def irfft(self, x, n: int, axis: int = -1):
        return cp.fft.irfft(x, n=n, axis=axis)

    def empty_cache(self) -> None:
        if _HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
