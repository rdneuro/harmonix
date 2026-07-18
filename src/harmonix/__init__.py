"""harmonix -- modal whole-brain simulation.

harmonix evolves the *amplitudes of eigenmodes* rather than node-wise ODEs. Brain
activity y(x, t) = sum_k a_k(t) psi_k(x) is expanded in a geometric (Laplace-
Beltrami), connectome-harmonic, or graph-Laplacian eigenbasis, and simulation
means evolving the modal coefficients a_k(t). The linear terms diagonalize exactly
in the eigenbasis (giving closed-form functional connectivity via a Lyapunov
equation -- no time-stepping); the nonlinear Hopf term is handled by a Galerkin
node-space round-trip. Criticality becomes a spectral quantity: distance-to-
criticality is the spectral abscissa, and the relaxation time is -1/abscissa.

Quick start
-----------
>>> import harmonix as hx
>>> basis = hx.datasets.demo_basis(n_modes=100)          # geometric LBO basis
>>> fc = hx.analytic_fc(basis, coupling=0.1)             # closed-form FC, no sim
>>> fc_sim = hx.simulate_fc(basis, model="hopf", coupling=0.05)  # simulated FC

Core objects: :class:`ModalBasis`, the models :class:`OUModal`,
:class:`LinearWave`, :class:`HopfModal`, and the :class:`Simulator`.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import backends, bases, criticality, datasets, metrics, models, simulate
from .api import analytic_fc, simulate_fc
from .bases import (ModalBasis, connectome_harmonics, geometric_modes,
                    graph_laplacian_modes, icosphere)
from .config import SimConfig
from .models import HopfModal, LinearWave, ModalModel, OUModal
from .simulate import Simulator, SimulationResult, sweep

# viz is imported lazily (pulls in matplotlib) via __getattr__


def __getattr__(name):
    if name == "viz":
        import importlib
        return importlib.import_module("harmonix.viz")
    raise AttributeError(f"module 'harmonix' has no attribute {name!r}")


__all__ = ["__version__", "ModalBasis", "geometric_modes", "connectome_harmonics",
           "graph_laplacian_modes", "icosphere", "OUModal", "LinearWave",
           "HopfModal", "ModalModel", "Simulator", "SimulationResult", "sweep",
           "SimConfig", "analytic_fc", "simulate_fc", "backends", "bases",
           "models", "simulate", "metrics", "criticality", "datasets", "viz"]
