"""CPU parallelization with the harmonix ``n_threads`` convention.

The convention, applied uniformly across the library:

* ``n_threads = 1`` (default) -- run serially, no joblib overhead.
* ``n_threads >= 2``          -- parallelize with that many joblib workers.
* ``n_threads = -1``          -- use every core the machine reports.

When workers run, the per-worker BLAS thread pool is pinned to 1 (via
``threadpoolctl`` when available) to prevent oversubscription -- the classic
"joblib x OpenMP" slowdown where ``n_workers x n_blas_threads`` exceeds the core
count. This matters because each modal simulation already calls into BLAS
(``Phi @ a``), so nesting thread pools would thrash.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Iterable, List

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    _HAS_JOBLIB = False


def resolve_n_jobs(n_threads: int) -> int:
    """Translate the ``n_threads`` convention into a joblib ``n_jobs`` count."""
    if n_threads == -1:
        return os.cpu_count() or 1
    if n_threads < 1:
        raise ValueError(
            f"n_threads must be 1, an integer >= 2, or -1; got {n_threads}")
    return int(n_threads)


@contextmanager
def _single_threaded_blas():
    """Pin BLAS/OpenMP to one thread per worker for the duration of the block."""
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            yield
    except ImportError:                                       # graceful fallback
        prev = {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        for k in prev:
            os.environ[k] = "1"
        try:
            yield
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def parallel_map(func: Callable, items: Iterable, n_threads: int = 1,
                 backend: str = "loky", verbose: int = 0) -> List:
    """Map ``func`` over ``items`` honouring the ``n_threads`` convention.

    Parameters
    ----------
    func : callable, applied to each item.
    items : iterable of inputs.
    n_threads : int, 1 = serial (default); >=2 = that many workers; -1 = all cores.
    backend : str, joblib backend ('loky' processes, 'threading' threads).
    verbose : int, joblib verbosity.

    Returns
    -------
    list of ``func(item)`` in input order.
    """
    items = list(items)
    n_jobs = resolve_n_jobs(n_threads)
    if n_jobs == 1 or not _HAS_JOBLIB:
        return [func(x) for x in items]
    with _single_threaded_blas():
        return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(func)(x) for x in items)
