"""The Simulator: orchestrates the modal time loop, batching, and parallelism.

A ``Simulator`` binds a :class:`ModalModel` to a :class:`SimConfig` and runs the
stochastic integration. The time loop is sequential (one trajectory), so scale-out
comes from *batching independent runs* -- seeds or parameter sets -- which
:meth:`run_batch` and the sweep helpers dispatch across joblib workers
(``n_threads`` convention) and, on GPU backends, VRAM-safe chunks guarded by a
device semaphore.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..backends import (DeviceSemaphore, get_backend, iter_chunks, parallel_map,
                        resolve_n_jobs, safe_chunk_size)
from ..backends.rng import complex_white_noise, spawn_streams
from ..config import SimConfig
from ..models.base import ModalModel
from .hemodynamic import balloon_windkessel, downsample_to_tr
from .integrators import ETDRK4Stepper, euler_maruyama, stochastic_heun
from .result import SimulationResult

_STEP_FUNCS = {"euler": euler_maruyama, "heun": stochastic_heun}


class Simulator:
    """Run modal simulations for a model under a configuration.

    Parameters
    ----------
    model : ModalModel
        The modal dynamical model (OU, Hopf, or wave).
    config : SimConfig, optional
        Simulation configuration; defaults to :class:`SimConfig` defaults.
    """

    def __init__(self, model: ModalModel, config: Optional[SimConfig] = None):
        self.model = model
        self.config = config or SimConfig()
        self._backend = get_backend(self.config.backend, dtype=self.config.dtype)

    # ------------------------------------------------------------------ #
    # single trajectory
    # ------------------------------------------------------------------ #
    def run(self, seed: Optional[int] = None, store_nodes: bool = False,
            bold: bool = False) -> SimulationResult:
        """Run one modal trajectory.

        Parameters
        ----------
        seed : int, optional
            Stream seed; defaults to the config seed.
        store_nodes : bool
            Also reconstruct and store node-space activity.
        bold : bool
            Also compute the Balloon-Windkessel BOLD signal (downsampled to TR).

        Returns
        -------
        SimulationResult
        """
        if self.model.state_kind == "real2":
            return self._run_wave(seed, store_nodes, bold)
        return self._run_complex1(seed if seed is not None else self.config.seed,
                                  store_nodes, bold)

    def _run_complex1(self, seed: int, store_nodes: bool, bold: bool) -> SimulationResult:
        cfg = self.config
        model = self.model
        n_modes = model.n_modes
        dt = cfg.dt
        n_warm = int(round(cfg.warmup / dt))
        n_keep = int(round(cfg.duration / dt))
        n_total = n_warm + n_keep

        mu = model.linear_rates().astype(np.complex128)       # (n_modes,)
        noise_amp = np.sqrt(model.noise_covariance()).astype(np.float64)
        nonlinear = model.nonlinear_drift

        # pre-draw all noise for this stream (reproducible, vectorized)
        dW = complex_white_noise(seed, 0, (n_total, n_modes))  # (n_total, n_modes)

        z = np.zeros(n_modes, dtype=np.complex128)
        out = np.empty((n_keep, n_modes), dtype=np.complex128)

        if cfg.integrator == "etdrk4":
            stepper = ETDRK4Stepper(mu, dt)
            for t in range(n_total):
                z = stepper.step(z, nonlinear, noise_amp, dW[t])
                if t >= n_warm:
                    out[t - n_warm] = z
        else:
            step = _STEP_FUNCS[cfg.integrator]
            for t in range(n_total):
                z = step(z, mu, nonlinear, noise_amp, dt, dW[t])
                if t >= n_warm:
                    out[t - n_warm] = z

        modal_ts = out.T                                      # (n_modes, n_keep)
        times = np.arange(n_keep) * dt
        result = SimulationResult(modal_timeseries=modal_ts, times=times, dt=dt,
                                  basis=model.basis,
                                  meta={"model": type(model).__name__,
                                        "integrator": cfg.integrator, "seed": seed})
        if store_nodes or bold:
            result.reconstruct_nodes()
        if bold:
            neural = np.real(result.node_timeseries)          # (n_nodes, n_keep)
            bold_full = balloon_windkessel(neural, dt)
            result.bold = downsample_to_tr(bold_full, dt, cfg.tr)
            result.tr = cfg.tr
        return result

    def _run_wave(self, seed, store_nodes, bold) -> SimulationResult:
        """Second-order damped-wave integration in (a, a_dot) per mode."""
        cfg = self.config
        model = self.model
        n_modes = model.n_modes
        dt = cfg.dt
        n_warm = int(round(cfg.warmup / dt))
        n_keep = int(round(cfg.duration / dt))
        n_total = n_warm + n_keep
        g = model.gamma_s
        stiff = model.stiffness()                             # (n_modes,)
        noise_amp = np.sqrt(model.noise_covariance())
        seed = seed if seed is not None else cfg.seed

        dW = complex_white_noise(seed, 0, (n_total, n_modes)).real
        a = np.zeros(n_modes)
        adot = np.zeros(n_modes)
        out = np.empty((n_keep, n_modes))
        for t in range(n_total):
            # a'' = gamma^2 (Q - stiff*a) - 2 gamma a'
            accel = g ** 2 * (noise_amp * np.sqrt(dt) * dW[t] / dt - stiff * a) - 2 * g * adot
            adot = adot + dt * accel
            a = a + dt * adot
            if t >= n_warm:
                out[t - n_warm] = a
        modal_ts = out.T.astype(np.complex128)
        times = np.arange(n_keep) * dt
        result = SimulationResult(modal_timeseries=modal_ts, times=times, dt=dt,
                                  basis=model.basis,
                                  meta={"model": "LinearWave", "seed": seed})
        if store_nodes or bold:
            result.reconstruct_nodes()
        if bold:
            result.bold = downsample_to_tr(
                balloon_windkessel(np.real(result.node_timeseries), dt), dt, cfg.tr)
            result.tr = cfg.tr
        return result

    # ------------------------------------------------------------------ #
    # batched / parallel runs
    # ------------------------------------------------------------------ #
    def run_batch(self, n_runs: int, base_seed: Optional[int] = None,
                  store_nodes: bool = False, bold: bool = False,
                  n_threads: Optional[int] = None) -> List[SimulationResult]:
        """Run ``n_runs`` independent trajectories with distinct seeds.

        Parallelized across runs via joblib (``n_threads`` convention). On GPU
        backends the runs are chunked to fit device memory, guarded by a
        semaphore so concurrent workers never oversubscribe VRAM.
        """
        base_seed = base_seed if base_seed is not None else self.config.seed
        n_threads = self.config.n_threads if n_threads is None else n_threads
        seeds = spawn_streams(base_seed, n_runs).astype(np.int64)

        def _one(s):
            return self.run(seed=int(s), store_nodes=store_nodes, bold=bold)

        if self._backend.supports_gpu:
            chunk = safe_chunk_size(self.model.n_modes,
                                    int(self.config.duration / self.config.dt),
                                    self._backend)
            sem = DeviceSemaphore(max_concurrent=max(1, resolve_n_jobs(n_threads)))
            results: List[SimulationResult] = []
            for start, stop in iter_chunks(n_runs, chunk):
                with sem.acquire():
                    results.extend(parallel_map(_one, seeds[start:stop],
                                                n_threads=n_threads))
                self._backend.empty_cache()
            return results
        return parallel_map(_one, seeds, n_threads=n_threads)

    # ------------------------------------------------------------------ #
    # vectorized batched run (the GPU-worthy path: many trajectories at once)
    # ------------------------------------------------------------------ #
    def run_batch_vectorized(self, n_runs: int, base_seed: Optional[int] = None,
                             return_modal: bool = False):
        """Integrate ``n_runs`` trajectories at once through the compute backend.

        Unlike :meth:`run_batch` (which parallelizes independent numpy runs over
        joblib workers), this keeps the entire batch ``(n_runs, n_modes)`` resident
        on the backend device and advances all trajectories together with backend
        array ops -- the Galerkin round-trip becomes two batched GEMMs
        (``Z @ Phi^T`` and ``cubic @ (M*Phi)``). This is the path that benefits from
        torch/jax/cupy GPUs; on the numpy backend it runs identically on CPU.

        Only first-order complex models (OU, Hopf) are supported here.

        Parameters
        ----------
        n_runs : int, number of trajectories.
        base_seed : int, optional base seed for Philox streams.
        return_modal : bool, if True also return the full modal time series
            ``(n_runs, n_modes, n_time)`` (memory-heavy); otherwise only the
            per-run node time series needed for FC.

        Returns
        -------
        node_timeseries : np.ndarray, shape (n_runs, n_nodes, n_time)
        modal_timeseries : np.ndarray, optional, shape (n_runs, n_modes, n_time)
        """
        if self.model.state_kind != "complex1":
            raise ValueError("run_batch_vectorized supports only complex1 models "
                             "(OUModal, HopfModal)")
        cfg = self.config
        model = self.model
        bk = self._backend
        xp = bk.xp
        base_seed = base_seed if base_seed is not None else cfg.seed

        n_modes = model.n_modes
        dt = cfg.dt
        n_warm = int(round(cfg.warmup / dt))
        n_keep = int(round(cfg.duration / dt))
        n_total = n_warm + n_keep

        # device-resident operators
        mu = bk.to_device(model.linear_rates().astype(np.complex128))          # (n_modes,)
        Phi = bk.to_device(model.basis.Phi)                                    # (n_nodes, n_modes)
        noise_amp = bk.to_device(np.sqrt(model.noise_covariance()))            # (n_modes,)
        if model.basis.mass is None:
            MPhi = Phi
        else:
            MPhi = bk.to_device(model.basis.mass[:, None] * model.basis.Phi)   # (n_nodes, n_modes)
        # optional Hopf dealias filter and homogeneity
        filt = getattr(model, "_filt", None)
        filt_dev = None if filt is None else bk.to_device(filt)
        is_hopf = getattr(model, "is_nonlinear", False)

        # batched noise (host Philox -> device), shape (n_total, n_runs, n_modes)
        seeds = spawn_streams(base_seed, n_runs)
        dW_host = np.stack([complex_white_noise(int(s), 0, (n_total, n_modes))
                            for s in seeds], axis=1)
        dW = bk.to_device(dW_host)

        Z = bk.to_device(np.zeros((n_runs, n_modes), dtype=np.complex128))
        keep = bk.to_device(np.zeros((n_keep, n_runs, n_modes), dtype=np.complex128))
        sqdt = float(np.sqrt(dt))

        def _nonlinear(Zc):
            if not is_hopf:
                return 0.0
            node = bk.matmul(Zc, Phi.T)                            # (n_runs, n_nodes)
            cubic = (xp.abs(node) ** 2) * node
            a_nl = -bk.matmul(cubic, MPhi)                          # (n_runs, n_modes)
            if filt_dev is not None:
                a_nl = a_nl * filt_dev[None, :]
            return a_nl

        for t in range(n_total):
            drift = mu[None, :] * Z + _nonlinear(Z)
            if cfg.integrator == "heun":
                noise = noise_amp[None, :] * sqdt * dW[t]
                Zp = Z + dt * drift + noise
                drift2 = mu[None, :] * Zp + _nonlinear(Zp)
                Z = Z + 0.5 * dt * (drift + drift2) + noise
            else:  # euler (default vectorized)
                Z = Z + dt * drift + noise_amp[None, :] * sqdt * dW[t]
            if t >= n_warm:
                keep[t - n_warm] = Z

        modal = bk.to_host(keep).transpose(1, 2, 0)                # (n_runs, n_modes, n_time)
        node = np.real(np.einsum("vm,rmt->rvt", model.basis.Phi, modal))
        bk.empty_cache()
        if return_modal:
            return node, modal
        return node
