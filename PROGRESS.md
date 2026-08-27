# ALIAKMON_py — progress & resume notes

Python/CuPy→**MPI+CPU** port of the hydro core of ALIAKMON (Fortran DNS CFD).
Source of truth for the original: `fortran_reference/` (cloned from
github.com/momferatos/aliakmon). Hydro only — no MHD, radiation, particles.

NOTE: `fortran_reference/` now DIVERGES from upstream — two numerical bugs
found during the port were fixed in place (not yet propagated to the build at
`/home/giorgos/src/aliakmon/` or upstream):
- `initial_conditions.f90:102` random_field: `exp(2π r2)` → `exp(ii*2π r2)`
  (missing imaginary unit; phase was a real exponential). Matches erandom_field.
- `numerics.f90:2293-2295` integral_length_scale: `cmplx(fu(i,j,k),fu(i,j,k))`
  → `cmplx(fu(i,j,k),fu(i+1,j,k))` (imag part was ignored; L/λ were wrong).
  Matches the correct pattern in vector_spectrum/scalar_spectrum.

## Stack
numpy + mpi4py-fft (distributed slab/pencil real FFTs) + Numba
`@njit(parallel=True)` kernels + mpi4py reductions. Dev venv: `.venv/`
(run `.venv/bin/python`). No GPU on this machine.

## Done & VALIDATED
- `parameters.py` — enums, constants, TOML `Config`
- `backend.py` — numpy as `xp` + MPI environment (replaces Fortran `mpivars`)
- `kernels.py` — Numba kernels (cross product, curl, projection, viscous,
  RK substage, max-norm); verified vs numpy references
- `transforms.py` — `Transforms`: PFFT wrapper, local wavenumbers, coords
- `data.py` — `State`: fields, mask (3 truncation criteria), phase factors
- `numerics.py` — RHS (rotational form), RK2/RK4, projection, CFL, diagnostics
  (+ kinetic helicity, shell energy spectrum, integral length scale,
  `setup_viscosity` Re/eta/nu from aliakmon.f90:163-185)
- `initial_conditions.py` — zero, stochastic, Orszag-Tang, ABC, Taylor-Green
- `input_output.py` — `load_config` + `print_progress` (hydro diagnostics box,
  hydro.dat log) + `Progress` timer
- `hdf5_io.py` — h5py `/u` vector field output + XDMF; `read_field` restart
  (serial + gather/broadcast for MPI)
- `validation.py` — `DissipationTest`: viscous dE/dt+eps budget, inviscid
  TE/MKH drift
- `aliakmon.py` — main driver: load → State → setup_viscosity → IC/restart →
  time loop (print_progress → cfl → timestep → validate → output) → final dump.
  Run: `.venv/bin/python -m aliakmon_py.aliakmon [config.toml]`

**Validation:** Taylor-Green vortex, n=48, nu=0.02 → initial KE=0.125 (exact
1/8), monotonic energy decay, divergence ~1e-17, energy budget dKE/dt=-eps
holds. The solver core is correct. Shell spectrum sums to KE exactly; TG
helicity ~1e-20. Full driver verified end-to-end serial AND 2-rank MPI
(identical diagnostics, E test ~0.03%); HDF5 write/restart roundtrips.

## VALIDATED against the Fortran reference (compare/)
Method: both codes integrate the SAME divergence-free initial field (the
Fortran's `output.000000.h5`, read into Python via `transpose(3,2,1,0)`), with
matched params (periodic, RK4, polyhedral, Patterson-Orszag). This isolates the
operators+time-stepping and sidesteps the differing IC RNGs. Generic harness:
`compare/python/compare_run.py <fortran_dir> <n> <timesteps>`.

Results (rel diff = single-precision floor, since Fortran stores fields rks=sp):
| case                  | steps | max KE rel diff | final-field rel-L2 |
|-----------------------|-------|-----------------|--------------------|
| TG  n=32              |  10   | 4e-8            | 4.5e-7             |
| TG  n=32 (longer)     |  80   | 1.7e-7          | 1.0e-6             |
| ABC n=32 (nonlinear)  |  30   | 1.1e-7          | 7.9e-7             |

The difference stays BOUNDED at ~1e-7 through the dissipation peak and into
decay (does not grow); time grids stay locked to <3e-8. ABC exercises the
nonlinear/dealiasing path (active small-scale modes) and still matches. n=64
is feasible but slow (~45 s/step; heFFTe single-rank per-FFT overhead).

Running the Fortran reference here:
- exe: `/home/giorgos/src/aliakmon/aliakmon.mkl.exe` (nvfortran/MKL build)
- `export LD_LIBRARY_PATH=/home/giorgos/lib/lib:/home/giorgos/src/lib/build_hdf5/bin`
- `mpiexec --mca btl self,vader -n 1 ...`; set `OMP_NUM_THREADS` (n=64 is ~40s/step!)
- nml MUST disable non-hydro physics: PASSIVE_SCALAR/RADIATION/BOUSSINESQ=.false.

Two interop gotchas:
1. HDF5 `/u` axis order. `hdf5_io` now writes/reads the FORTRAN-COMPATIBLE
   layout `(N,N,N,3)` = component-last, spatial axes (z,y,x). So Python and
   Fortran `.h5` files interchange directly (byte-comparable, overlay in
   ParaView) and `read_field` reads either code's output. The legacy harness
   `compare_run.py` instead loads the Fortran file with `transpose(3,2,1,0)` ->
   `(component,i,j,k)` and compares in-memory; both routes agree. Verify any
   axis work via divergence (~1e-7 right vs ~0.25 wrong).
2. `BOUNDCOND=1` (free-slip) in the Fortran applies `check_free_slip_bcs` each
   step (sheds ~half the energy on step 1). The port is periodic-only; use
   `BOUNDCOND=0` to compare. Free-slip BCs are NOT ported.

## Done & VALIDATED (continued)
- `forcing` — Kaneda et al. (2004) negative-viscosity forcing:
  - `data.py` `State`: `fscale[3]` (init 0) + `_ke_prev` (init 1.0)
  - `numerics.py` `_apply_forcing`: adds `fscale[c]*fu_hat[k]` for modes with
    kx≠0, ky≠0, kz≠0, |k|<kforcing; called in `compute_rhs` before projection
  - `input_output.py` `_update_fscale`: Kaneda energy feedback (tol=0.01,
    dfs=0.05), called each step in `print_progress` on all ranks; no extra
    broadcast needed since KE is already a global allreduce.  `fscale` printed
    as FHD in the progress box and logged in hydro.dat (column 10).
  Enable with `[force] forced=true variable_forcing=true kforcing=2.5` in
  config.toml. KE tracks KENTAR=0.5 within ~2% after ~100 steps (n=32 ABC).

## TODO (next)
The hydro port is feature-complete and runnable (decaying AND forced turbulence).
Possible follow-ups:
energy spectrum file per output frame (`output_spectra`), 2D slice output
(`output_slices`), dissipation-peak detection / `stop_at_disspeak`, gzip
compression on HDF5, collective parallel HDF5 (currently gather-to-root).

## Gotchas (already fixed; don't reintroduce)
1. RK4: zero `rks2` each step before stage 1 (stage 1 assigns, not accumulates).
2. ICs must be Hermitian — build from a real field via forward FFT.
3. Normalize all 3 velocity components by ONE shared scalar (else breaks div=0).
4. Patterson-Orszag dealiasing is correctly a no-op under 2/3 truncation.
5. Use plain np arrays everywhere, NOT mpi4py-fft DistArray (rejects `x[...]`).
6. Multi-rank: `mpiexec --mca btl self,vader -n N` (ignore TCP iface warnings).
7. ABC IC MUST superpose k=1..3 (cyclic component mixing) + a 10% stochastic
   perturbation. A single-mode ABC flow is Beltrami (omega=k u) so u x omega=0
   identically → the run is pure viscous diffusion, not hydro. Diagnose any
   "looks like diffusion" report by checking |u x omega|/|u| of the IC.
9. "Evolution looks like diffusion" in ParaView is a VIZ artifact, not physics:
   velocity magnitude is smooth (max/mean ~2.3) even for a correct turbulent
   field; structure lives in the VORTICITY (max/mean ~3.3). The Fortran's
   slice.*.h5 store vorticity 'w' + dissipation 'e', so comparing Python
   velocity to Fortran slices misleads. hdf5_io now writes a `/w` vorticity-
   magnitude scalar in every output frame (matches the Fortran to ~1e-6); color
   by `w` (or use a Q-criterion / Curl filter on `u`). At Re_lambda~27 (n=32)
   the flow is genuinely low-Re and only mildly structured regardless.
8. Stochastic IC spectra (initial_conditions.py): Fortran `random_field(kmax)`
   is a FLAT spectrum with a hard cutoff |k|<kmax AND only modes with all three
   wavenumber components nonzero (so kmax=2 → only the (+-1,+-1,+-1) modes, a
   large-scale perturbation). Fortran `erandom_field` is a power law,
   |coef|~k^(-11/6) (E(k)~k^(-11/3)). Do NOT model either as a Gaussian
   envelope at a "peak" wavenumber — that leaks spurious small-scale energy
   (e.g. an ABC IC that looks grainy at k=4..7 vs the Fortran's smooth field).

## How to resume
Open a session in this dir and say "resume ALIAKMON_py". I'll read MEMORY.md +
this file. Quick smoke test that everything still works:

    .venv/bin/python -c "from aliakmon_py.parameters import Config; \
      from aliakmon_py.data import State; from aliakmon_py import numerics as N, \
      initial_conditions as IC; from aliakmon_py.parameters import InitCond; \
      s=State(Config(n=32, initcond=InitCond.TAYLOR_GREEN_VORTEX)); \
      IC.set_initial_conditions(s); N.set_viscosity(s,0.02); \
      print('KE', N.kinetic_energy(s), 'div', N.incompressibility(s))"
