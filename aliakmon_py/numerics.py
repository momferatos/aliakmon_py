"""Core pseudo-spectral solver (hydro subset of numerics.f90).

Implements the incompressible Navier-Stokes right-hand side in rotational
form, Patterson-Orszag dealiasing, the pressure projection, RK2/RK4 time
integration, the CFL condition and the basic diagnostics. MHD, radiation and
passive-scalar terms of the original ``right_hand_side`` are omitted.

State layout: velocity lives in Fourier space as ``state.fu`` (a list of three
component arrays). Every routine works on the local subdomain and uses
:mod:`aliakmon_py.kernels` for the hot per-gridpoint loops; MPI reductions
come from :mod:`aliakmon_py.backend`.
"""

from __future__ import annotations

import math

import numpy as np

from . import kernels as K
from .backend import (allreduce_array, allreduce_max, allreduce_min,
                      allreduce_sum)
from .data import State
from .parameters import (D, LBOX, NFIELDS, RMSUTAR, Config, Dealiasing,
                         Integration)

# RK4 stage weights (rkc in runge_kutta4).
_RKC = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)


# ----------------------------------------------------------------------
# Kaneda et al. (2004) negative-viscosity forcing  (forcing_rhs)
# ----------------------------------------------------------------------
def _apply_forcing(state: State, fin) -> None:
    """Add the Kaneda forcing term to state.rhs for the low-k band.

    Forced modes satisfy: kx≠0, ky≠0, kz≠0, and |k| < cfg.kforcing.
    The term is fscale[c] * fin[c] (negative viscosity in spectral space).
    state.fscale is updated each step via input_output._update_fscale.
    """
    kf = state.cfg.kforcing
    forced = (
        (state.KX != 0) & (state.KY != 0) & (state.KZ != 0)
        & (np.sqrt(state.K2) < kf)
    ).astype(np.float64)
    for c in range(3):
        state.rhs[c] += state.fscale[c] * forced * fin[c]


# ----------------------------------------------------------------------
# Right-hand side  (right_hand_side + compute_hydro + compute_diffusive_terms)
# ----------------------------------------------------------------------
def compute_rhs(state: State, fin) -> None:
    """Evaluate the RHS of d(fu)/dt for the velocity field ``fin``.

    ``fin`` is a list of three Fourier-space component arrays. The result is
    written into ``state.rhs`` (also three components). Side effect: refreshes
    ``state.maxvel`` and ``state.maxvort`` (used by the CFL condition).
    """
    T = state.T
    cfg = state.cfg
    dealias = cfg.dealiasing == Dealiasing.PATTERSON_ORSZAG

    u, w = state.u_real, state.w_real
    fw, fnl, fnls = state.fw, state.fnl, state.fnls
    nl = state.u_real  # reused as the real-space product buffer (see below)

    # Vorticity in Fourier space: omega = i k x u.
    K.spectral_curl(fin[0], fin[1], fin[2], state.kx, state.ky, state.kz,
                    fw[0], fw[1], fw[2])

    # Phase-shifted copies for dealiasing, transformed to physical space.
    if dealias:
        psu, du = state.psu_real, state.du_real
        for c in range(3):
            T.backward(fin[c] * state.phase, psu[c])
            T.backward(fw[c] * state.phase, du[c])

    # Non-shifted velocity and vorticity in physical space.
    for c in range(3):
        T.backward(fin[c], u[c])
        T.backward(fw[c], w[c])

    # Diagnostics for the CFL condition (global maxima).
    state.maxvel = allreduce_max(K.max_norm3(u[0], u[1], u[2]))
    state.maxvort = allreduce_max(K.max_norm3(w[0], w[1], w[2]))

    # Nonlinear term n = u x omega in physical space, then forward transform.
    # `nl` aliases `u`: cross product reads u/w and overwrites u in place,
    # which is safe because each output element depends only on the matching
    # input element.
    K.nonlinear_cross(u[0], u[1], u[2], w[0], w[1], w[2], nl[0], nl[1], nl[2])
    for c in range(3):
        T.forward(nl[c], fnl[c])

    if dealias:
        psu, du = state.psu_real, state.du_real
        K.nonlinear_cross(psu[0], psu[1], psu[2], du[0], du[1], du[2],
                          psu[0], psu[1], psu[2])
        for c in range(3):
            T.forward(psu[c], fnls[c])
        # Undo the shift and average: n = 0.5*(n + shift^-1(n_shifted)).
        for c in range(3):
            fnl[c][...] = 0.5 * (fnl[c] + fnls[c] * state.iphase)

    # Remove aliased modes.
    state.apply_truncation_mask(fnl)

    # Diffusive term and assemble RHS: rhs = nl - nu(k) k^2 fu + f_forcing.
    for c in range(3):
        state.rhs[c][...] = fnl[c]
        if cfg.viscous:
            K.add_viscous(state.rhs[c], fin[c], state.visc[c], state.K2)

    if cfg.forced:
        _apply_forcing(state, fin)

    # Project onto the divergence-free subspace (skip for Burgers).
    if not cfg.burgers:
        K.project_solenoidal(state.rhs[0], state.rhs[1], state.rhs[2],
                             state.kx, state.ky, state.kz)

    state.apply_truncation_mask(state.rhs)


# ----------------------------------------------------------------------
# Time integration  (timestep / runge_kutta2 / runge_kutta4)
# ----------------------------------------------------------------------
def timestep(state: State, dt: float) -> None:
    """Advance ``state.fu`` by one step using the configured RK method."""
    if state.cfg.integration_method == Integration.RUNGE_KUTTA2:
        _rk2(state, dt)
    elif state.cfg.integration_method == Integration.RUNGE_KUTTA4:
        _rk4(state, dt)
    else:
        raise ValueError("timestep: invalid integration method")


def _rk2(state: State, dt: float) -> None:
    fu, rhs, s1 = state.fu, state.rhs, state.rks1
    compute_rhs(state, fu)
    for c in range(3):
        s1[c][...] = fu[c] + 0.5 * dt * rhs[c]
    compute_rhs(state, s1)
    for c in range(3):
        fu[c][...] = fu[c] + dt * rhs[c]


def _rk4(state: State, dt: float) -> None:
    fu, rhs = state.fu, state.rhs
    s1, s2 = state.rks1, state.rks2  # s1: stage state, s2: running increment

    # Stage 1. Zero the running increment first: rk_stage accumulates into s2,
    # so it must start clean each timestep (the Fortran assigns on stage 1).
    compute_rhs(state, fu)
    for c in range(3):
        s2[c][...] = 0.0
        K.rk_stage(fu[c], rhs[c], s2[c], _RKC[0] * dt, 0.5 * dt, s1[c])
    # Stage 2
    compute_rhs(state, s1)
    for c in range(3):
        K.rk_stage(fu[c], rhs[c], s2[c], _RKC[1] * dt, 0.5 * dt, s1[c])
    # Stage 3
    compute_rhs(state, s1)
    for c in range(3):
        K.rk_stage(fu[c], rhs[c], s2[c], _RKC[2] * dt, 1.0 * dt, s1[c])
    # Stage 4 + final update
    compute_rhs(state, s1)
    for c in range(3):
        fu[c][...] = fu[c] + s2[c] + _RKC[3] * dt * rhs[c]


# ----------------------------------------------------------------------
# CFL condition  (cfl_condition)
# ----------------------------------------------------------------------
def cfl_dt(state: State) -> float:
    """Stable timestep from the current velocity field: CFL*dx/max|u|, <= 0.05.

    Requires ``state.maxvel`` to be current (set by the most recent
    :func:`compute_rhs`); on the very first call it is computed on demand.
    """
    maxvel = getattr(state, "maxvel", 0.0)
    if maxvel <= 0.0:
        for c in range(3):
            state.T.backward(state.fu[c], state.u_real[c])
        maxvel = allreduce_max(
            K.max_norm3(state.u_real[0], state.u_real[1], state.u_real[2]))
        state.maxvel = maxvel

    dx = LBOX / state.n
    dt = state.cfg.cfl * dx / max(maxvel, 1.0e-30)
    return min(dt, 0.05)


# ----------------------------------------------------------------------
# Diagnostics  (msvalue / dissipation / incompressibility)
# ----------------------------------------------------------------------
def mean_square_velocity(state: State, fin=None) -> float:
    """<|u|^2> averaged over the box (msvalue for the velocity field)."""
    fin = state.fu if fin is None else fin
    total = 0.0
    for c in range(3):
        ur = state.T.backward(fin[c], state.u_real[c])
        total += float(np.sum(ur * ur))
    total *= state.msfac
    return allreduce_sum(total)


def kinetic_energy(state: State, fin=None) -> float:
    """Total kinetic energy KE = 0.5 <|u|^2>."""
    return 0.5 * mean_square_velocity(state, fin)


def mean_dissipation(state: State, fin=None) -> float:
    """Mean viscous dissipation rate eps = <nu(k) |omega|^2>.

    Computed spectrally as the sum over modes of nu_c(k) * k^2 * |u_hat_c|^2,
    accounting for the Hermitian symmetry of the real transform (interior kz
    modes count twice). This is exactly the energy the diffusion term of
    :func:`compute_rhs` removes, so it stays consistent with the budget check
    in :mod:`aliakmon_py.validation` for any nu(k); with a uniform viscosity it
    collapses to the familiar eps = nu <|omega|^2> = 2 nu Z.
    """
    weight = _hermitian_weight(state)
    local = 0.0
    for c in range(NFIELDS):
        f = state.fu[c] if fin is None else fin[c]
        e = np.abs(f) ** 2
        local += float(np.sum(state.visc[c] * weight * state.K2 * e))
    return allreduce_sum(local)


def incompressibility(state: State, fin=None) -> float:
    """Global max |k . u| over active modes (should stay ~ machine epsilon)."""
    fin = state.fu if fin is None else fin
    div = state.KX * fin[0] + state.KY * fin[1] + state.KZ * fin[2]
    return allreduce_max(float(np.abs(div).max()))


def mean_kinetic_helicity(state: State, fin=None) -> float:
    """Mean kinetic helicity H = 0.5 <u . omega> (mean_kinetic_helicity).

    The vorticity is formed spectrally, both fields are brought to physical
    space and their pointwise dot product is box-averaged. The factor of one
    half matches the Fortran convention.
    """
    fin = state.fu if fin is None else fin
    K.spectral_curl(fin[0], fin[1], fin[2], state.kx, state.ky, state.kz,
                    state.fw[0], state.fw[1], state.fw[2])
    local = 0.0
    for c in range(3):
        u = state.T.backward(fin[c], state.u_real[c])
        w = state.T.backward(state.fw[c], state.w_real[c])
        local += float(np.sum(u * w))
    return 0.5 * state.msfac * allreduce_sum(local)


def _hermitian_weight(state: State) -> np.ndarray:
    """Per-mode multiplicity of the packed real transform (kz folding).

    The rfft keeps only ``kz >= 0``; every interior kz mode therefore stands
    in for its conjugate too and counts twice, while ``kz == 0`` and the
    Nyquist ``kz == N/2`` count once. Shaped to broadcast over a Fourier field.
    """
    n = state.n
    wz = np.where((state.kz == 0) | (state.kz == n // 2), 1.0, 2.0)
    return wz[None, None, :]


def energy_spectrum(state: State, fin=None):
    """Spherically-binned kinetic-energy spectrum ``E(k)`` (vector_spectrum).

    Returns ``(k, E)`` where ``k`` are integer shell wavenumbers and ``E[k]``
    is the kinetic energy in the shell ``round(|k|) == k``; ``sum(E)`` equals
    the total kinetic energy. Summed across ranks so the result is global.
    """
    fin = state.fu if fin is None else fin
    weight = _hermitian_weight(state)
    e = np.zeros_like(state.K2)
    for c in range(3):
        e += weight * (np.abs(fin[c]) ** 2)
    e *= 0.5  # KE density per mode

    nbins = state.n // 2 + 1
    kbin = np.minimum(np.rint(np.sqrt(state.K2)).astype(np.int64), nbins - 1)
    local = np.bincount(kbin.ravel(), weights=e.ravel(), minlength=nbins)
    spectrum = allreduce_array(local)
    return np.arange(nbins, dtype=np.float64), spectrum


def integral_length_scale(state: State, fin=None):
    """Integral length scale ``L`` and Taylor microscale ``lambda``.

    Both follow from the shell spectrum: ``L = (sum E(k)/k) / sum E(k)`` over
    non-zero shells and ``lambda = sqrt(sum E / sum k^2 E)``. The driver
    overrides ``lambda`` with the dissipation-based estimate; it is returned
    here for parity with the Fortran ``integral_length_scale``.
    """
    k, e = energy_spectrum(state, fin)
    total = float(e.sum())
    if total == 0.0:
        return 0.0, 0.0
    nz = k > 0
    L = float((e[nz] / k[nz]).sum()) / total
    diss = float((k ** 2 * e).sum())
    lam = math.sqrt(total / diss) if diss > 0.0 else 0.0
    return L, lam


# ----------------------------------------------------------------------
# Viscosity setup  (Reynolds-number handling from aliakmon.f90)
# ----------------------------------------------------------------------
def set_viscosity(state: State, nu) -> None:
    """Set the kinematic viscosity nu(k) on every velocity component.

    ``nu`` may be

    * a scalar — a uniform molecular viscosity (the classic DNS case);
    * a single array broadcastable to the local Fourier grid — one spectral
      viscosity profile shared by the three components;
    * a length-3 list/tuple of either — one profile per component.

    Values are *copied* into ``state.visc``, so callers keep ownership of their
    own buffers. ``state.nu_mol`` is refreshed to the global minimum over all
    components and modes: for a molecular viscosity carrying a non-negative
    eddy viscosity on top, that is the molecular floor the Taylor/Kolmogorov
    diagnostics are defined against, and it reduces to ``nu`` exactly in the
    scalar case.
    """
    if isinstance(nu, (list, tuple)):
        if len(nu) != NFIELDS:
            raise ValueError(
                f"set_viscosity: expected {NFIELDS} components, got {len(nu)}")
        parts = nu
    else:
        parts = (nu,) * NFIELDS

    # `[...] =` broadcasts scalars and shaped arrays alike, and raises on a
    # shape that does not fit the local Fourier grid.
    for c in range(NFIELDS):
        state.visc[c][...] = parts[c]

    local_min = min((float(v.min()) for v in state.visc if v.size),
                    default=math.inf)
    state.nu_mol = allreduce_min(local_min)


def setup_viscosity(state: State) -> dict:
    """Derive the kinematic viscosity from the run's target Reynolds number.

    Ports aliakmon.f90:163-185. Either the Taylor-microscale Reynolds number
    ``re`` is prescribed (and the Kolmogorov scale ``eta`` follows) or it is
    left at zero and inferred from the requested resolution ``kmax*eta``
    (``kmaxeta``). The viscosity that hits the target dissipation ``D`` at unit
    rms velocity is then ``nu = 15^{1/4} eta * RMSUTAR / sqrt(re)``.

    Returns the derived quantities used by the startup banner.
    """
    cfg = state.cfg
    kmax = state.kmax

    if cfg.re == 0.0:
        eta = cfg.kmaxeta / kmax
        re_tmp = ((D ** 0.25) * eta) ** (-4.0 / 3.0)
        re = 6.0 * math.sqrt(re_tmp)
    else:
        re = cfg.re
        re_tmp = (re / 6.0) ** 2
        eta = D ** (-0.25) * re_tmp ** (-0.75)

    nu = 15.0 ** 0.25 * (eta * RMSUTAR) / math.sqrt(re)
    if not cfg.viscous:
        nu = 0.0
    set_viscosity(state, nu)

    emean_target = nu ** 3 / eta ** 4 if eta > 0.0 else 0.0
    lambda_target = (math.sqrt(15.0 * nu * RMSUTAR ** 2 / emean_target)
                     if emean_target > 0.0 else 0.0)
    return dict(eta=eta, re=re, re_tmp=re_tmp, nu=nu, kmax=kmax,
                emean_target=emean_target, lambda_target=lambda_target)
