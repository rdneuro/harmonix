"""Compute backends and parallelism for harmonix.

``get_backend`` returns a backend instance; ``'gpu'``/``'auto'`` selects the
first available GPU backend (torch-CUDA, cupy, jax), falling back to numpy. The
``n_threads`` convention (1 serial / >=2 workers / -1 all cores) is implemented in
:mod:`harmonix.backends.parallel`; VRAM-safe batching in
:mod:`harmonix.backends.memory`.
"""
from __future__ import annotations

from typing import Dict

from .base import Backend
from .memory import (DeviceSemaphore, bytes_per_simulation, iter_chunks,
                     safe_chunk_size, total_device_memory_bytes)
from .numpy_backend import NumpyBackend
from .parallel import parallel_map, resolve_n_jobs

__all__ = ["Backend", "get_backend", "available_backends", "parallel_map",
           "resolve_n_jobs", "DeviceSemaphore", "safe_chunk_size",
           "bytes_per_simulation", "iter_chunks", "total_device_memory_bytes"]


def _first_gpu(device=None, dtype="float64"):
    try:
        import torch
        if torch.cuda.is_available():
            from .torch_backend import TorchBackend
            return TorchBackend(device=device, dtype=dtype)
    except Exception:
        pass
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            from .cupy_backend import CupyBackend
            return CupyBackend(device=device, dtype=dtype)
    except Exception:
        pass
    try:
        import jax
        if any(d.platform in ("gpu", "tpu") for d in jax.devices()):
            from .jax_backend import JaxBackend
            return JaxBackend(device=device, dtype=dtype)
    except Exception:
        pass
    return NumpyBackend()


def get_backend(backend: str = "cpu", device=None, dtype: str = "float64") -> Backend:
    """Instantiate a compute backend.

    Parameters
    ----------
    backend : {'cpu', 'gpu', 'numpy', 'torch', 'jax', 'cupy', 'auto'}
        ``'cpu'``/``'numpy'`` -> numpy; ``'gpu'``/``'auto'`` -> first available GPU
        backend then numpy.
    device : backend-specific device selector.
    dtype : {'float64', 'float32'}; float64 default (analytic core precision).
    """
    b = backend.lower()
    if b in ("cpu", "numpy"):
        return NumpyBackend()
    if b in ("gpu", "auto"):
        return _first_gpu(device=device, dtype=dtype)
    if b == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend(device=device, dtype=dtype)
    if b == "jax":
        from .jax_backend import JaxBackend
        return JaxBackend(device=device, dtype=dtype)
    if b == "cupy":
        from .cupy_backend import CupyBackend
        return CupyBackend(device=device, dtype=dtype)
    raise ValueError(
        f"unknown backend {backend!r}; choose 'cpu', 'gpu', 'numpy', 'torch', "
        "'jax', 'cupy', or 'auto'")


def available_backends() -> Dict[str, bool]:
    """Report which backends are importable right now."""
    status = {"numpy": True}
    for mod in ("torch", "jax", "cupy"):
        try:
            __import__(mod)
            status[mod] = True
        except ImportError:
            status[mod] = False
    return status
