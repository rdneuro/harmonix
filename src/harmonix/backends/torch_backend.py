"""PyTorch backend (CPU or CUDA) for large batched modal integration."""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional
    torch = None
    _HAS_TORCH = False

from .base import Backend

_TDT = {}
if _HAS_TORCH:
    _TDT = {"float64": torch.float64, "float32": torch.float32,
            "complex128": torch.complex128, "complex64": torch.complex64}


class _TorchNS:
    """Minimal numpy-like namespace over torch for shared model code."""

    def __getattr__(self, name):                              # delegate the rest
        return getattr(torch, name)

    def sum(self, x, axis=None):
        return torch.sum(x) if axis is None else torch.sum(x, dim=axis)

    def mean(self, x, axis=None):
        return torch.mean(x) if axis is None else torch.mean(x, dim=axis)

    @property
    def linalg(self):
        return torch.linalg


class TorchBackend(Backend):
    """GPU/CPU backend backed by PyTorch."""

    name = "torch"
    supports_gpu = True

    def __init__(self, device: Optional[str] = None, dtype: str = "float64"):
        if not _HAS_TORCH:
            raise ImportError("TorchBackend requires torch ('pip install harmonix[torch]')")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._dtype = _TDT[dtype]
        self._ns = _TorchNS()

    @property
    def xp(self):
        return self._ns

    def _cdtype(self):
        return torch.complex128 if self._dtype == torch.float64 else torch.complex64

    def to_device(self, x, dtype: Optional[str] = None):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            arr = np.asarray(x)
            if np.iscomplexobj(arr) and dtype is None:
                return torch.as_tensor(np.ascontiguousarray(arr), device=self._device)
            if np.issubdtype(arr.dtype, np.integer) and dtype is None:
                return torch.as_tensor(np.ascontiguousarray(arr), device=self._device)
            t = torch.as_tensor(np.ascontiguousarray(arr), device=self._device)
        if dtype is not None:
            return t.to(device=self._device, dtype=_TDT[dtype])
        if t.is_complex():
            return t.to(device=self._device)
        return t.to(device=self._device, dtype=self._dtype)

    def to_host(self, x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return np.ascontiguousarray(x.detach().cpu().numpy())
        return np.ascontiguousarray(np.asarray(x))

    def matmul(self, a, b):
        return a @ b

    def eigh(self, a):
        return torch.linalg.eigh(a)

    def expm(self, a):
        return torch.linalg.matrix_exp(a)

    def rfft(self, x, axis: int = -1):
        return torch.fft.rfft(x, dim=axis)

    def irfft(self, x, n: int, axis: int = -1):
        return torch.fft.irfft(x, n=n, dim=axis)

    def empty_cache(self) -> None:
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
