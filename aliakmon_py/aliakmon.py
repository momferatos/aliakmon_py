"""Main driver for the hydrodynamic solver (port of aliakmon.f90).

Wires the modules together into a runnable simulation: load the configuration,
build the distributed :class:`~aliakmon_py.data.State`, fix the viscosity from
the target Reynolds number, set the initial condition (or restart from file),
then march the incompressible Navier-Stokes equations in time while reporting
diagnostics and writing HDF5 output.

Run it as::

    .venv/bin/python -m aliakmon_py.aliakmon [config.toml]
    mpiexec -n 4 .venv/bin/python -m aliakmon_py.aliakmon config.toml

The MHD, radiation, particle, passive-scalar, forcing-feedback and
post-processing paths of the original program are out of scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import hdf5_io
from . import initial_conditions as IC
from . import numerics as N
from .backend import IS_ROOT, backend_name, on_root
from .data import State
from .input_output import Progress, load_config, print_progress
from .validation import DissipationTest


def _banner(state: State, info: dict) -> None:
    """Print the startup banner with the resolved run parameters."""
    if not IS_ROOT:
        return
    cfg = state.cfg
    print("=" * 72)
    print(" ALIAKMON_py — pseudo-spectral incompressible DNS (hydro)")
    print(f" backend: {backend_name()}")
    print("=" * 72)
    print(f"| N {cfg.n:5d} | kmax {state.kmax:8.3f} | k_max(active) "
          f"{state.k_max_active:8.3f} | modes {state.nmodes}")
    print(f"| Rel {info['re']:10.3f} | RE {info['re_tmp']:10.3e} "
          f"| nu {info['nu']:10.3e}")
    print(f"| eta {info['eta']:10.3e} | kmax*eta {state.kmax * info['eta']:7.3f}"
          f" | lambda {info['lambda_target']:10.3e}")
    print(f"| IC {cfg.initcond.name} | integration {cfg.integration_method.name}"
          f" | truncation {cfg.truncation.name}")
    print("=" * 72, flush=True)


def _initial_condition(state: State) -> float:
    """Set the starting velocity field; return the starting time.

    Either restarts from ``input_field_filename`` or evaluates the analytic /
    stochastic initial condition and rescales it to unit mean-square velocity
    (the Fortran ``rescale``, giving KE = 0.5 and rms = RMSUTAR).
    """
    cfg = state.cfg
    if cfg.input_field:
        on_root(f"Reading restart file {cfg.input_field_filename}")
        return hdf5_io.read_field(state, cfg.input_field_filename)

    on_root("Setting initial conditions...")
    IC.set_initial_conditions(state)
    IC.normalize_ms(state, 1.0)
    on_root(f"max(div) = {N.incompressibility(state):.3e}")
    return 0.0


def _keep_running(cfg, time: float, t_start: float, k: int) -> bool:
    """Time-loop continuation test (mirrors the Fortran ``timeloop`` guard)."""
    if cfg.timesteps == 0:
        return time - t_start <= cfg.tmax
    return k <= cfg.timesteps


def main(config_path: str | Path = "config.toml") -> None:
    cfg = load_config(config_path)
    state = State(cfg)
    info = N.setup_viscosity(state)
    _banner(state, info)

    t_start = _initial_condition(state)
    time = t_start
    nhdf5 = 0
    # Write the t=0 frame (frame 0) for both fresh and restarted runs so the
    # output numbering tracks the simulation time consistently.
    if cfg.outputfiles:
        hdf5_io.write_field(state, time, nhdf5)
        nhdf5 += 1

    validator = DissipationTest(state) if cfg.valid else None
    hydro_log = open("hydro.dat", "w") if IS_ROOT else None

    progress = Progress()
    progress.start()
    on_root("Entering main loop...")

    k = 1
    try:
        while _keep_running(cfg, time, t_start, k):
            print_progress(state, k, time, getattr(state, "_dt", 0.0),
                           progress, hydro_log)

            dt = N.cfl_dt(state)
            state._dt = dt
            N.timestep(state, dt)

            if validator is not None:
                validator.update(dt)

            time += dt
            k += 1

            if cfg.outputfiles and cfg.hdf5frate != 0.0:
                if int(time * cfg.hdf5frate) >= nhdf5:
                    hdf5_io.write_field(state, time, nhdf5)
                    nhdf5 += 1
    finally:
        if hydro_log is not None:
            hydro_log.close()

    if cfg.outputfiles:
        hdf5_io.write_field(state, time, 888888)
    _write_spectrum(state)

    on_root(f"Done. {k - 1} steps, t = {time:.4f}, "
            f"elapsed {progress.elapsed():.2f}s.")


def _write_spectrum(state: State, fname: str = "espec.final.dat") -> None:
    """Write the final kinetic-energy spectrum E(k) (output_spectra)."""
    k, e = N.energy_spectrum(state)
    if not IS_ROOT:
        return
    with open(fname, "w") as fh:
        for ki, ei in zip(k, e):
            fh.write(f"{ki:30.14e} {ei:30.14e}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.toml")
