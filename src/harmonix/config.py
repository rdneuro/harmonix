"""Configuration defaults for harmonix.

Research-grounded defaults (see the design report): the low-risk analytic linear
model on a 200-mode geometric LBO basis with the closed-form Lyapunov FC is the
default working point; the Hopf model integrates with stochastic Heun; hemodynamic
and criticality parameters follow the Deco/Pang conventions.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- modal defaults ----
DEFAULT_N_MODES = 200                    # Pang et al. 2023

# ---- Hopf / Stuart-Landau defaults (Deco et al.) ----
DEFAULT_BIFURCATION = -0.02              # a: just below the Hopf bifurcation
DEFAULT_NOISE = 0.02                     # beta: additive noise amplitude
DEFAULT_INTRINSIC_FREQ_HZ = 0.05         # omega / 2pi: BOLD band peak

# ---- linear wave defaults (Pang loadParameters_wave_func) ----
WAVE_GAMMA_S = 116.0                     # 1/s damping rate
WAVE_R_S_MM = 30.0                       # mm spatial length scale

# ---- integration defaults ----
DEFAULT_DT = 0.1                         # s (neural step; downsample to TR)
DEFAULT_TR = 0.72                        # s (HCP repetition time)
DEFAULT_WARMUP_S = 60.0                  # s discarded transient
DEFAULT_INTEGRATOR = "heun"             # stochastic Heun

# ---- Balloon-Windkessel defaults (Friston 2000 / Stephan 2007) ----
BALLOON = {"kappa": 0.65, "gamma": 0.41, "tau": 0.98, "alpha": 0.32,
           "rho": 0.34, "V0": 0.02}

# ---- FCD defaults ----
FCD_WINDOW_S = 30.0
FCD_STEP_S = 2.0


@dataclass(frozen=True)
class SimConfig:
    """Global simulation defaults.

    Attributes
    ----------
    n_modes : int, number of eigenmodes retained.
    dt : float, integration step (s).
    duration : float, simulated duration (s), excluding warmup.
    warmup : float, discarded transient (s).
    tr : float, BOLD sampling period (s) for downsampling.
    integrator : {'euler', 'heun', 'etdrk4'}.
    backend : {'cpu', 'gpu', 'numpy', 'torch', 'jax', 'cupy', 'auto'}.
    dtype : {'float64', 'float32'}.
    seed : int, base seed for Philox streams.
    n_threads : int, 1 serial / >=2 workers / -1 all cores.
    """

    n_modes: int = DEFAULT_N_MODES
    dt: float = DEFAULT_DT
    duration: float = 600.0
    warmup: float = DEFAULT_WARMUP_S
    tr: float = DEFAULT_TR
    integrator: str = DEFAULT_INTEGRATOR
    backend: str = "cpu"
    dtype: str = "float64"
    seed: int = 0
    n_threads: int = 1

    def __post_init__(self):
        if self.integrator not in ("euler", "heun", "etdrk4"):
            raise ValueError(f"integrator must be 'euler', 'heun', or 'etdrk4'; "
                             f"got {self.integrator!r}")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0; got {self.dt}")
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1; got {self.n_modes}")
