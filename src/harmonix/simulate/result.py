"""Container for a modal simulation's outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SimulationResult:
    """Outputs of a modal simulation.

    Attributes
    ----------
    modal_timeseries : np.ndarray, shape (n_modes, n_time)
        Complex modal coefficients over time (post-warmup, at neural dt).
    times : np.ndarray, shape (n_time,)
        Time stamps (s).
    dt : float
        Neural timestep (s).
    basis : ModalBasis
        The basis used (for reconstruction to node space).
    node_timeseries : np.ndarray or None
        Node-space activity Re(Phi @ a), if requested.
    bold : np.ndarray or None
        BOLD signal per node (at TR), if the hemodynamic model was applied.
    tr : float or None
        BOLD repetition time (s), if BOLD was computed.
    meta : dict
        Model/config metadata.
    """

    modal_timeseries: np.ndarray
    times: np.ndarray
    dt: float
    basis: object
    node_timeseries: Optional[np.ndarray] = None
    bold: Optional[np.ndarray] = None
    tr: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        return self.modal_timeseries.shape[0]

    @property
    def n_time(self) -> int:
        return self.modal_timeseries.shape[1]

    def reconstruct_nodes(self) -> np.ndarray:
        """Reconstruct real node activity Re(Phi @ a) over time (cached)."""
        if self.node_timeseries is None:
            self.node_timeseries = np.real(self.basis.reconstruct(self.modal_timeseries))
        return self.node_timeseries

    def __repr__(self):
        return (f"SimulationResult(n_modes={self.n_modes}, n_time={self.n_time}, "
                f"dt={self.dt}, bold={'yes' if self.bold is not None else 'no'})")
