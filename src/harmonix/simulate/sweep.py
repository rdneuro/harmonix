"""Parameter sweeps over modal models (coupling G, bifurcation a).

Sweeps are embarrassingly parallel across parameter values; this module builds a
fresh model per value and runs a metric callback, dispatching across joblib
workers (``n_threads`` convention). Because each sweep point is an independent
simulation, this is the primary scale-out axis for harmonix.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ..backends import parallel_map
from ..config import SimConfig
from .simulator import Simulator


def sweep(model_factory: Callable, values: Sequence[float],
          metric: Callable, config: Optional[SimConfig] = None,
          n_threads: int = 1, seed: int = 0, analytic: bool = False) -> np.ndarray:
    """Evaluate a metric over a 1-D parameter grid.

    Parameters
    ----------
    model_factory : callable ``value -> ModalModel``
        Builds a model for each parameter value.
    metric : callable
        If ``analytic`` is True, ``metric(model) -> float`` (no simulation, e.g.
        spectral abscissa or analytic FC similarity). Otherwise
        ``metric(SimulationResult) -> float``.
    values : sequence of floats, the parameter grid.
    config : SimConfig, optional (used only when ``analytic`` is False).
    n_threads : int, parallelism (1 serial / >=2 workers / -1 all cores).
    seed : int, base seed.
    analytic : bool, if True the metric reads the model directly with no simulation.

    Returns
    -------
    np.ndarray, shape (len(values),), the metric at each grid point.
    """
    cfg = config or SimConfig()
    values = list(values)

    def _one(value):
        model = model_factory(value)
        if analytic:
            return float(metric(model))
        return float(metric(Simulator(model, cfg).run(seed=seed, store_nodes=True)))

    return np.array(parallel_map(_one, values, n_threads=n_threads), dtype=np.float64)
