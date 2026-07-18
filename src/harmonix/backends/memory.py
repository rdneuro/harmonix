"""RAM/VRAM-aware batching for large parameter/seed sweeps.

Batched modal simulation stores an array of shape roughly
``(batch, n_modes, n_timesteps)``; on a bounded GPU (design target: ~7.6 GB
usable on an RTX 4070) the sweep must be split into VRAM-safe chunks. This module
estimates per-simulation memory, derives a safe chunk size, and exposes a
semaphore/token bucket that bounds the number of concurrent GPU jobs so several
workers never oversubscribe device memory at once.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional


_BYTES = {"float64": 8, "complex128": 16, "float32": 4, "complex64": 8}


def bytes_per_simulation(n_modes: int, n_timesteps: int, dtype: str = "complex128",
                         overhead: float = 3.0) -> int:
    """Estimate device bytes for one modal trajectory.

    Parameters
    ----------
    n_modes, n_timesteps : int
        Trajectory shape ``(n_modes, n_timesteps)``.
    dtype : str
        State dtype ('complex128' for Hopf, 'float64' for linear/wave).
    overhead : float
        Multiplier accounting for temporaries (nonlinearity buffer, noise,
        propagator, node-space reconstruction). Default 3x is conservative.

    Returns
    -------
    int, estimated bytes.
    """
    return int(overhead * n_modes * n_timesteps * _BYTES[dtype])


def total_device_memory_bytes(backend) -> Optional[int]:
    """Total memory of the backend's device, or None on CPU/unknown."""
    if getattr(backend, "name", "") == "torch":
        try:
            import torch
            if torch.cuda.is_available():
                # NOTE: the attribute is `total_memory`, never `total_mem`
                return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            return None
    if getattr(backend, "name", "") == "cupy":
        try:
            import cupy as cp
            free, total = cp.cuda.runtime.memGetInfo()
            return int(total)
        except Exception:
            return None
    return None


def safe_chunk_size(n_modes: int, n_timesteps: int, backend,
                    dtype: str = "complex128", usable_fraction: float = 0.8,
                    fallback_bytes: int = 7_600_000_000) -> int:
    """Largest simulation batch that fits in the device's usable memory.

    On CPU (no device query) returns a large default so batching is effectively
    unbounded and controlled instead by the joblib worker count.
    """
    total = total_device_memory_bytes(backend)
    budget = int(usable_fraction * (total if total is not None else fallback_bytes))
    per = bytes_per_simulation(n_modes, n_timesteps, dtype)
    return max(1, budget // max(per, 1))


class DeviceSemaphore:
    """Bounded token bucket limiting concurrent device (GPU) jobs.

    Use as a context manager around each device-resident batch so that, when
    several joblib/threads workers run, they do not collectively exceed VRAM.
    On CPU backends the bound defaults to the worker count and is effectively a
    no-op guard.
    """

    def __init__(self, max_concurrent: int = 1):
        self._sem = threading.BoundedSemaphore(max(1, int(max_concurrent)))
        self.max_concurrent = max(1, int(max_concurrent))

    @contextmanager
    def acquire(self):
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


def iter_chunks(n_items: int, chunk_size: int):
    """Yield ``(start, stop)`` index pairs covering ``range(n_items)``."""
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n_items, chunk_size):
        yield start, min(start + chunk_size, n_items)
