"""Configuration loading and progress reporting (input_output.f90, hydro).

Two responsibilities survive from the Fortran module in the hydro port:

* :func:`load_config` — the analogue of ``read_namelist_file``; it reads the
  TOML run file into a :class:`~aliakmon_py.parameters.Config`.
* :func:`print_progress` — a trimmed ``print_progress`` that gathers the hydro
  diagnostics (energy, dissipation, length scales, Reynolds numbers) and prints
  the per-step status box on the root rank, optionally logging ``hydro.dat``.

MHD, passive-scalar, particle and forcing-feedback reporting are omitted.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass
from pathlib import Path

from .backend import IS_ROOT
from .data import State
from .parameters import Config, KENTAR
from . import numerics as N


_FORCING_TOL = 1.0e-2   # KE dead-band around KENTAR (Fortran: tol)
_FORCING_DFS = 0.05     # fscale step size per timestep (Fortran: dfs)


def _update_fscale(state: State, ke: float) -> None:
    """Kaneda et al. (2004) fscale feedback; called each step on all ranks.

    Adjusts state.fscale so kinetic energy tracks KENTAR: ramp fscale up when
    KE is below target and falling, down when above target and rising. All
    ranks compute the identical update (ke is a global allreduce), so no
    broadcast is needed.
    """
    cfg = state.cfg
    # Needs a sink for the injected energy: molecular viscosity, or a subgrid
    # model of either family (an LES run may well set viscous = false).
    if not (cfg.forced and cfg.variable_forcing
            and (cfg.viscous or cfg.les_active)):
        return
    ke_prev = state._ke_prev
    if ke < KENTAR - _FORCING_TOL and ke <= ke_prev:
        state.fscale[:] += _FORCING_DFS
    elif ke > KENTAR + _FORCING_TOL and ke >= ke_prev:
        state.fscale[:] -= _FORCING_DFS
    state._ke_prev = ke


def load_config(path: str | Path = "config.toml") -> Config:
    """Read the TOML run configuration (replaces ``read_namelist_file``)."""
    return Config.from_toml(path)


@dataclass
class Progress:
    """Wall-clock bookkeeping for the estimated-time-left readout."""

    t_start: float = 0.0

    def start(self) -> None:
        self.t_start = _time.perf_counter()

    def elapsed(self) -> float:
        return _time.perf_counter() - self.t_start


def _split_hms(seconds: float):
    """Break a duration into (days, hours, minutes, seconds)."""
    seconds = max(seconds, 0.0)
    days, seconds = divmod(int(seconds), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return days, hours, minutes, seconds


def compute_diagnostics(state: State) -> dict:
    """Evaluate the hydro diagnostics reported each timestep.

    Mirrors the hydro portion of ``print_progress``: divergence, kinetic
    energy, dissipation, kinetic helicity, the integral length scale and the
    derived Taylor/Kolmogorov scales and Reynolds numbers.
    """
    # The Taylor/Kolmogorov scales and their Reynolds numbers are defined
    # against the molecular viscosity, so use the scalar floor of nu(k) rather
    # than any mode-dependent (eddy) part sitting on top of it.
    nu = state.nu_mol
    ke = N.kinetic_energy(state)
    rmsu = math.sqrt(2.0 / 3.0 * ke) if ke > 0.0 else 1.0
    # Total dissipation = resolved viscous drain + the subgrid transfer. A
    # structural model drains through the RHS force, not through nu(k), so it
    # is a separate term; eps_sgs is 0.0 unless such a model is running.
    emean_visc = N.mean_dissipation(state) if state.cfg.diffusive else 0.0
    eps_sgs = N.sgs_dissipation(state)
    emean = emean_visc + eps_sgs
    ils, _ = N.integral_length_scale(state)

    # Taylor microscale and its Reynolds number use the dissipation estimate;
    # guard against the zero-dissipation (inviscid / initial) case.
    eps = emean if emean > 0.0 else 1.0
    lam = math.sqrt(15.0 * nu * rmsu ** 2 / eps) if nu > 0.0 else 0.0
    rel = rmsu * lam / nu if nu > 0.0 else 0.0
    eta = (nu ** 3 / eps) ** 0.25 if nu > 0.0 else 0.0
    ett = ils / (3.0 * rmsu) if rmsu > 0.0 else 0.0
    re = rmsu * ils / nu if nu > 0.0 else 0.0

    return dict(
        maxdiv=N.incompressibility(state),
        ke=ke, rmsu=rmsu, emean=emean,
        emean_visc=emean_visc, eps_sgs=eps_sgs,
        mkh=N.mean_kinetic_helicity(state),
        ils=ils, lam=lam, rel=rel, eta=eta, ett=ett, re=re,
        maxvel=getattr(state, "maxvel", 0.0),
        maxvort=getattr(state, "maxvort", 0.0),
    )


def print_progress(state: State, ntimestep: int, t: float, dt: float,
                   progress: Progress, hydro_log=None) -> dict:
    """Print the per-step status box and return the computed diagnostics.

    ``hydro_log`` is an optional open text file; when given a row of the key
    scalars is appended (the hydro.dat stream of the original).
    """
    diag = compute_diagnostics(state)
    _update_fscale(state, diag["ke"])   # all ranks keep fscale in sync
    if not IS_ROOT:
        return diag

    cfg = state.cfg
    if cfg.timesteps != 0:
        percent = ntimestep / cfg.timesteps * 100.0
    elif cfg.tmax > 0.0:
        percent = t / cfg.tmax * 100.0
    else:
        percent = 0.0

    elapsed = progress.elapsed()
    if percent > 0.0:
        etl = elapsed * (100.0 - percent) / percent
    else:
        etl = 0.0
    ed = _split_hms(elapsed)
    ld = _split_hms(etl)
    kmax_eta = state.kmax * diag["eta"]

    bar = "-" * 72
    star = "*" * 72
    print(star)
    print(f"| {percent:6.2f}% | elapsed {ed[0]:d}:{ed[1]:02d}:{ed[2]:02d}:{ed[3]:02d}"
          f" | ETL {ld[0]:d}:{ld[1]:02d}:{ld[2]:02d}:{ld[3]:02d}")
    print(bar)
    print(f"| step {ntimestep:6d} | t {t:9.4f} | div {diag['maxdiv']:9.2e}"
          f" | maxw {diag['maxvort']:9.2e} | maxu {diag['maxvel']:9.2e}")
    print(bar)
    fhd_str = f" | FHD {state.fscale[0]:+8.4f}" if cfg.forced else ""
    print(f"| kmax*eta {kmax_eta:7.3f} | Rel {diag['rel']:9.3f}"
          f"{fhd_str} | dt {dt:9.2e} | KE {diag['ke']:9.5f}")
    print(bar)
    print(f"| RE {diag['re']:8.3f} | eta {diag['eta']:8.4f}"
          f" | lambda {diag['lam']:8.4f} | L {diag['ils']:8.4f}"
          f" | ETT {diag['ett']:8.4f}")
    les = state.les_info
    if les is not None:
        print(bar)
        if les["kind"] == "tensor":
            print(f"| LES {les['model']} | k_c {les['kc']:7.2f}"
                  f" | Delta {les['delta']:8.4f}"
                  f" | eps_sgs {diag['eps_sgs']:+9.3e}"
                  f" | eps_nu {diag['emean_visc']:9.3e}")
        else:
            print(f"| nu_mol {state.nu_mol:9.3e} | nu_t plateau {les['plateau']:9.3e}"
                  f" | cusp {les['peak']:9.3e} | E(k_c) {les['e_kc']:9.3e}")
    print(star, flush=True)

    if hydro_log is not None:
        hydro_log.write(
            f"{t:24.8e} {diag['ke']:24.8e} {diag['mkh']:24.8e} "
            f"{diag['emean']:24.8e} {diag['ils']:24.8e} {diag['lam']:24.8e} "
            f"{diag['eta']:24.8e} {diag['re']:24.8e} {diag['rel']:24.8e} "
            f"{state.fscale[0]:24.8e}\n")
        hydro_log.flush()

    return diag
