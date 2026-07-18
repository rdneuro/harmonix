"""Modal dynamical models: linear wave, OU, and nonlinear Hopf."""
from .base import ModalModel
from .hopf_modal import HopfModal
from .linear_wave import LinearWave
from .ou_modal import OUModal

__all__ = ["ModalModel", "OUModal", "LinearWave", "HopfModal"]
