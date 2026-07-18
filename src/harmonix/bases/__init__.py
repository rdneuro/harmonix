"""Modal bases: geometric (LBO), connectome harmonics, and graph-Laplacian."""
from ._meshes import icosphere
from .connectome import connectome_harmonics
from .geometric import geometric_modes
from .graph_laplacian import graph_laplacian_modes
from .modal_basis import ModalBasis

__all__ = ["ModalBasis", "geometric_modes", "connectome_harmonics",
           "graph_laplacian_modes", "icosphere"]
