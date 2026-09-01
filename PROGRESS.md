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
- `data.py` — `State`: fields, mask (3 truncation criteria; an LES run
  overrides the configured one — see the LES bullet), phase factors
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

- `array viscosity` — the scalar kinematic viscosity is now a per-component
  field `nu_c(k)` on the local Fourier grid (`State.visc` is three real arrays
  shaped like `cmplx_shape`, not three scalars). `kernels.add_viscous` indexes
  it per mode; `mean_dissipation` sums `nu_c(k) k^2 |u_hat|^2` with nu *inside*
  the mode sum, which keeps it exactly equal to the energy the diffusion term
  removes so the `validation.py` budget still closes for any nu(k).
  `set_viscosity(state, nu, nu_mol=None)` takes a scalar, one shared array or a
  per-component list. Bitwise identical to the old scalar path for uniform nu.
  `State.nu_mol` holds the scalar molecular viscosity for the Taylor/Kolmogorov
  diagnostics — do NOT infer it from `min(visc)` once an LES model is on (the
  eddy viscosity has a non-zero k->0 plateau, so the minimum is
  `nu_mol + plateau`). Extends `compute_diffusive_terms`, which is scalar-only
  in the Fortran.

- `LES` — Chollet & Lesieur (1981) spectral eddy viscosity (NOT in the Fortran):
  - `numerics.chollet_lesieur_nu_t`:
    `nu_t(k) = A (1 + 34.5 exp(-3.03 k_c/k)) sqrt(E(k_c)/k_c)` with plateau
    `A = 0.441 C_K^(-3/2)` (= 0.267 at C_K=1.4, reproducing the published
    `0.267 + 9.21 exp(-3.03/x)` to 0.3%). Cusp is 2.667x the plateau at `k_c`.
  - `numerics.update_subgrid_model` (was `update_eddy_viscosity`, renamed when
    pDDLES arrived): sets `visc = nu_mol + nu_t`, called once per step from
    `aliakmon.main` *after* `timestep`, so nu_t is frozen across the RK stages
    and every diagnostic sees a nu_t consistent with the current field.
    `E(k_c)` costs one global reduction — do not call it per RK stage.
  - `Config.les_active` / `Config.diffusive`: the diffusion term and the
    dissipation diagnostics key off `diffusive`, so `viscous=false` + LES
    (pure eddy viscosity, no molecular part) still works.
  - **The numerical cutoff IS the LES cutoff.** Any LES model (functional or
    structural) makes `State._setup_mask` truncate *spherically* at `k_c`,
    whatever `[numerics] truncation` says — `data.State._effective_truncation`
    announces the override on root. A subgrid model is defined behind a sharp
    spectral filter, i.e. for a field holding every mode inside a sphere of
    radius `k_c` and nothing outside; the two-thirds (box) and polyhedral
    boundaries are neither, and a truncation *above* `k_c` leaves modes the
    model does not close. The test is inclusive (`|k| <= k_c`), matching
    `sgs._filter_mask` and pDDLES's own training filter. `k_c` comes from
    `sgs.les_cutoff`: `les_kc`, else `les_alpha`, else pDDLES's alpha, else
    `kmax` (which for a bare `les_model=1` reproduces the plain spherical mask).
  - **`les_alpha` names the cutoff as a fraction, on pDDLES's basis.**
    `k_c = les_alpha * sqrt(3)/2 * n` (`sgs.alpha_cutoff`), the grid CORNER as
    in `TurbDataset._les_filter_mask` — *not* the solver's
    `kmax = sqrt(2)/3 * n`. So `les_alpha = 0.1` and a checkpoint trained at
    `alpha = 0.1` are the same filter at the same `n` (verified: both give
    `k_c = 5.5426` and the same 418 modes at n=64), which is what makes a
    Chollet-Lesieur run comparable with a pDDLES one. Consequences of that
    choice: the usable range stops at `(sqrt(2)/3)/(sqrt(3)/2) = 0.5443`, where
    `k_c = kmax`; and `les_alpha = 0` means `kmax`, not zero. `les_cutoff` is
    the single choke point every cutoff passes through, so it rejects
    `les_kc` + `les_alpha` together (two names for one thing), `les_alpha` with
    pDDLES (whose alpha rides in the checkpoint), and any `k_c > kmax` — that
    last one guards the truncation change above, since the solver now truncates
    at `k_c` and would otherwise evolve modes past the dealiasing limit.
  Enable with `[les] les_model=1 les_ck=1.4 les_alpha=0.25`. Validated at
  n=32, Re_lambda=300 (kmax*eta=0.05, far too
  coarse for DNS): nu_t saturates near 10x nu_mol on the plateau and 28x at the
  cusp, and the energy budget closes to 0.1-0.3%. Expect a transient few-%
  budget residual while nu_t first ramps up — that is the O(dt) cost of
  freezing nu_t across a step, not an accounting error.

- `pDDLES` — **structural** subgrid model: builds the SGS stress tensor
  explicitly instead of an eddy viscosity (`aliakmon_py/sgs.py`, NOT in the
  Fortran). Predicts the full field from the resolved large scales with a
  trained PyTorch net, then filters:
      u*     = Predictor(U)                      (TorchScript checkpoint)
      tau_ij = G*(u*_i u*_j) - U_i U_j
      f_i    = -d(tau_ij)/dx_j   ->   f_hat_i = -i k_j tau_hat_ij
  `f` is added to the RHS *before* the pressure projection, which disposes of
  tau's isotropic part — no trace subtraction needed. Because `U` is
  solenoidal, the projection is orthogonal to it, so `eps_sgs` computed from
  the *unprojected* force is exact.
  - Model families now split: `Config.les_eddy_viscosity` (functional, folds
    `nu_t(k)` into `visc`) vs `Config.les_tensor` (structural, adds a force).
    `Config.diffusive` keys off the *eddy-viscosity* one only, so
    `viscous=false` + pDDLES is a genuinely diffusion-free run.
  - `numerics.sgs_dissipation`: `eps_sgs = -sum_k w Re(u_hat* . f_hat)`, added
    to the budget in `validation.py` so `dE/dt + eps_visc + eps_sgs ~ 0` still
    closes. May legitimately go NEGATIVE (backscatter) — that is the whole
    point of not using an eddy viscosity, and an eddy viscosity cannot do it.
  - The net is not MPI-decomposed: inference runs on the field gathered to
    root and is scattered back (like `hdf5_io`, simplicity over scalability).
    Verified: the SGS force checksum is bit-identical at 1, 2 and 4 ranks.
  - Stress is built once per step and frozen across the RK substages (same
    compromise as nu_t) — 4 net evaluations per step would otherwise dominate.
    Measured budget residual is cleanly O(dt): 1.71e-3 -> 8.66e-4 -> ... as dt
    halves. Do NOT read that residual as an accounting error.
  - `les_clip_backscatter` zeroes tau where `-tau_ij S_ij < 0`. Off by default
    so the trained model's backscatter is kept; turn it on if a run goes
    unstable. Measured: eps_sgs -5.5e-4 (net backscatter) unclipped vs +0.17
    clipped, i.e. the clip is what makes the closure net-dissipative.
  Enable with `[les] les_model=2 les_pddles_model="<Epoch_NNN>.pth"
  les_pddles_source="/home/giorgos/src/pDDLES"`.

  Loading the real pDDLES artifacts (checked against that tree, NOT guessed):
  - Weights are `.pth` **state_dicts**, not TorchScript: `trainer.save` writes
    `dict(epoch, model, optimizer, args)`. The checkpoint is SELF-DESCRIBING —
    the training `argparse.Namespace` rides along — so the architecture, its
    hyper-parameters and `alpha` are read from the file, not from config.toml.
    Load with `weights_only=False` (it holds a Namespace).
  - The architecture is imported from the pDDLES tree (`lib/arch/<arch>.py`)
    via `les_pddles_source`; ALIAKMON does not vendor it. `args.actfun` and
    `args.batchnorm` are pickled live objects, so the Namespace reconstructs
    the net exactly. Only `device`/`dev`/`noload`/`copy` are overridden.
  - **The scaler RIDES IN the checkpoint** and is NOT optional. pDDLES
    normalises outside the net (`trainer.train_one_epoch`):
        X_s = (X - X_mean)/X_std;  pred = model(X_s);  u* = y_std*pred + y_mean
    Since pDDLES commit 6777e04 ("Embed normalization constants in
    checkpoints, drop norm.pt") `trainer.save` writes
    `scaler=self.scaler.state_dict()` into the .pth, so the constants travel
    with the weights: `{X_mean,X_std,y_mean,y_std}` for NormScaler,
    `{X_min,X_max,y_min,y_max}` for MinmaxScaler, `{}` for DummyScaler, each
    per-component shape `(1,3,1,1,1)`. Skipping them feeds the net inputs far
    outside its training range.
    Those checkpoints also DROP `args.h5path` — nothing external is needed.
    ALIAKMON reads only the embedded state: the standalone `norm.pt` /
    `minmax.pt` files are no longer loaded, and a checkpoint with no `scaler`
    key is REJECTED (re-export it from a current pDDLES). `les_pddles_scaler`
    no longer takes a path — only `""`, `"auto"` or `"none"`.
  - `les_pddles_scaler = "auto"` is the stand-in when constants are absent:
    it standardises each component with the *field's own* mean and variance
    every call (verified: zero mean, unit variance to the float32 floor), and
    unscales with the same constants so an identity network round-trips.
    It follows the running solution rather than the training set, so it is a
    get-it-running/diagnostic path, not production. It warns once on root.
  - `k_c` is DERIVED from the checkpoint's `alpha`, so tau is built with the
    filter the net was trained against. pDDLES measures wavenumbers in
    cycles/sample (`torch.fft.fftfreq`), where the grid corner is `sqrt(3)/2`
    and `_les_filter_mask` keeps `|k|/n <= alpha*sqrt(3)/2`; in this solver's
    integer wavenumbers that is `k_c = alpha * sqrt(3)/2 * n`. Setting
    `les_kc` overrides it, which is usually a mistake. The solver truncates
    there too, so the evolved band is exactly the net's training band; that is
    why `State` loads the checkpoint (via `sgs.les_cutoff`) while it is still
    building the mask, which also makes a bad checkpoint fail before the
    banner rather than after the first output frame.
  - The net is resolution-specific (spectral coefficients indexed by a fixed
    wavenumber grid), so the run's `n` must equal the checkpoint's; enforced.
  torch is an optional dependency; nothing else in the port imports it.

- `32^3 pDDLES LES` — first run against a real checkpoint. Config + outputs in
  `runs/pddles32-fnet8/` (gitignored). torch 2.13.0+cpu is now installed in
  `.venv`. Predictor: `FNet32-8/weights/FNet32-8/Epoch_029.pth` (30 epochs,
  8 blocks, n=32, alpha=0.1, scaler='norm').
  - `alpha=0.1` => `k_c = alpha*sqrt(3)/2*n = 2.7713`, so the resolved band is
    only shells 0,1,2 (`k_max_active=2.449`, **51 modes** on the 32^3 grid).
    That band is now what the solver evolves automatically (the truncation is
    the sphere `|k| <= k_c`); `[numerics] truncation_kc`, which this run used
    to need set by hand, is left for cutting *below* `k_c`.
  - The mask alone moves — `state.kmax` stays at the scheme's nominal 15.085,
    because that is the grid's resolving power and is what sets nu in
    `setup_viscosity` and the `kmax*eta` diagnostic. Do NOT collapse the two:
    the banner's `kmax` and `k_max(active)` are different numbers on purpose,
    and it is `k_max(active)` that equals `k_c`.
  - Old checkpoints predate `precision`/`torch_dtype` in `pDDLES.py:main`;
    `sgs._backfill_args` re-derives the missing Namespace fields (never
    overriding what a checkpoint carries).
  - Result: 200 steps to t=10 in 65 s (~0.32 s/step, of which ~0.17 s is the
    network). Budget closes: mean 0.029%, worst 0.067%. div ~1e-16. KE decays
    0.500 -> 0.132, Re_lambda 31 -> 14.
  - **OPEN ISSUE:** `eps_sgs` is only 0.7-2.2% of the molecular dissipation
    and flips sign constantly (113 steps forward-scatter, 88 backscatter,
    range -9.3e-4 .. +7.6e-4) — i.e. the closure is nearly transfer-neutral.
    That is implausibly small for a cutoff that discards ~85% of the
    wavenumber range. Prime suspect is the stand-in scaler: measured directly,
    `rms(G*u*)` comes back ~25% BELOW `rms(U)`, so the reconstruction is well
    off in the resolved band alone. Get the real constants before drawing any
    conclusion about the model. UPDATE: those constants are now available —
    they ride in the checkpoint (see the scaler bullet above), so this is
    re-testable without hunting for a `norm.pt`. Not yet re-measured against a
    fully-trained embedded-scaler checkpoint.

## TODO (next)
The hydro port is feature-complete and runnable (decaying AND forced
turbulence; DNS, Chollet-Lesieur eddy-viscosity LES, or pDDLES tensor LES).
pDDLES HAS now been run against real trained checkpoints (torch 2.13.0+cpu is
installed in `.venv`): n=32 and n=64, serial and under MPI, budget closing to
0.1-2.5%, `eps_sgs` bitwise rank-independent at n=32 (1e-14 at n=64, inherited
from the distributed FFT). Still outstanding: every such run used a
lightly-trained or old-format checkpoint, so `eps_sgs` magnitude is not yet
trustworthy — see the OPEN ISSUE above.
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
10. Testing a subgrid model needs a BROADBAND field. Taylor-Green is
   spectrally compact (k=1 only, so u_i u_j lands at k=2, far inside the
   cutoff): `G*(u_i u_j) == u_i u_j`, tau is identically zero and every SGS
   test passes vacuously. The stochastic ICs are broadband but seed per rank,
   so serial and MPI results are NOT comparable. Use a deterministic analytic
   superposition spanning k=1..7 (see the pDDLES test harness).
11. An identity prediction (`u* == U`) must give ZERO SGS energy transfer —
   no predicted subgrid scales, no transfer. It is the model's consistency
   limit, and a useful check, but it cannot validate the pipeline (tau is
   non-zero in physical space; only `eps_sgs` vanishes, since the deviatoric
   trace subtraction leaves a pure gradient in the force that the solenoidal
   resolved field cannot see).
12. `tau_ij = G*(u*_i u*_j) - G*(u*_i) G*(u*_j)` — BOTH terms from the
   prediction (pDDLES `TurbDataset.subgrid_scale_tensor`). Substituting the
   solver's own `U` for `G*u*` looks equivalent, since the net is trained so
   `G*u* ~ U`, but the difference is exactly the reconstruction error and it
   enters tau as a spurious stress instead of cancelling.
13. A cutoff that is *reported* is not necessarily *applied*. `_filter_mask`
   once keyed off `cfg.les_kc` alone, so the alpha-derived `k_c` was computed,
   broadcast and printed in the progress box while the filter silently stayed
   at the full truncation — correct-looking banner, wrong tau, `eps_sgs` at
   round-off. The regression test used to be "a derived `k_c` keeps strictly
   fewer modes than `state.mask`"; that is now INVERTED, because the solver
   truncates at `k_c` as well, so `_filter_mask == state.mask` is the correct
   outcome. Test instead that both equal the sphere `|k| <= k_c`, i.e. that
   `k_max_active <= k_c` while `kmax` stays at the scheme's nominal value.
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
