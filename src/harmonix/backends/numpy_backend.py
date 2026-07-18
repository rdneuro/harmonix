"""NumPy CPU backend (reference; always available)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.fft as _fft
import scipy.linalg as _sla

from .base import Backend

_DTYPES = {"float64": np.float64, "float32": np.float32,
           "complex128": np.complex128, "complex64": np.complex64}


class NumpyBackend(Backend):
    """CPU backend backed by numpy/scipy. The correctness reference."""

    name = "numpy"
    supports_gpu = False

    @property
    def xp(self):
        return np

    def to_device(self, x, dtype: Optional[str] = None):
        arr = np.asarray(x)
        if np.iscomplexobj(arr) and dtype is None:
            return np.ascontiguousarray(arr)
        if dtype is not None:
            arr = arr.astype(_DTYPES[dtype])
        elif not (np.issubdtype(arr.dtype, np.floating)
                  or np.issubdtype(arr.dtype, np.integer)):
            arr = arr.astype(np.float64)
        return np.ascontiguousarray(arr)

    def to_host(self, x) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(x))

    def matmul(self, a, b):
        return a @ b

    def eigh(self, a):
        return np.linalg.eigh(a)

    def expm(self, a):
        return _sla.expm(a)

    def rfft(self, x, axis: int = -1):
        return _fft.rfft(x, axis=axis)

    def irfft(self, x, n: int, axis: int = -1):
        return _fft.irfft(x, n=n, axis=axis)
