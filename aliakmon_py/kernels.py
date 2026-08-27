"""Numba pointwise kernels (the hot loops of numerics.f90).

Each kernel operates on a rank's *local* subdomain and is agnostic to how the
global domain is decomposed across MPI ranks: the caller passes the three
1-D wavenumber axes (already sliced to the local range) that correspond to the
three array dimensions. This mirrors the per-gridpoint ``do`` loops in
numerics.f90 (compute_hydro, curl, project, compute_diffusive_terms) without
the FFTW packed real/imaginary layout — here Fourier fields are native
complex128.

Velocity components are passed as three separate contiguous arrays
(``vx, vy, vz``) rather than a 4-D stack so Numba can parallelise cleanly.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# njit options shared by every kernel: thread the outer loop, allow FMA, and
# cache the compiled object between runs.
_OPTS = dict(parallel=True, fastmath=True, cache=True)


@njit(**_OPTS)
def nonlinear_cross(u1, u2, u3, w1, w2, w3, f1, f2, f3):
    """Real-space nonlinear term f = u x omega (compute_hydro inner loop).

    Writes the cross product of (u1,u2,u3) and (w1,w2,w3) into (f1,f2,f3).
    """
    a, b, c = u1.shape
    for i in prange(a):
        for j in range(b):
            for k in range(c):
                ux = u1[i, j, k]
                uy = u2[i, j, k]
                uz = u3[i, j, k]
                wx = w1[i, j, k]
                wy = w2[i, j, k]
                wz = w3[i, j, k]
                f1[i, j, k] = uy * wz - uz * wy
                f2[i, j, k] = uz * wx - ux * wz
                f3[i, j, k] = ux * wy - uy * wx


@njit(**_OPTS)
def max_norm3(u1, u2, u3):
    """Maximum Euclidean norm of a real vector field (for CFL / diagnostics)."""
    a, b, c = u1.shape
    # `m = max(m, s)` is the reduction idiom Numba recognises across prange.
    m = 0.0
    for i in prange(a):
        for j in range(b):
            for k in range(c):
                s = u1[i, j, k] ** 2 + u2[i, j, k] ** 2 + u3[i, j, k] ** 2
                m = max(m, s)
    return np.sqrt(m)


@njit(**_OPTS)
def spectral_curl(vx, vy, vz, kx, ky, kz, wx, wy, wz):
    """Spectral curl omega = i k x u (curl in numerics.f90).

    ``kx, ky, kz`` are the 1-D wavenumber axes for dimensions 0, 1, 2 of the
    complex fields. Results go into (wx, wy, wz).
    """
    a, b, c = vx.shape
    for i in prange(a):
        kxi = kx[i]
        for j in range(b):
            kyj = ky[j]
            for k in range(c):
                kzk = kz[k]
                ax = vx[i, j, k]
                ay = vy[i, j, k]
                az = vz[i, j, k]
                wx[i, j, k] = 1j * (kyj * az - kzk * ay)
                wy[i, j, k] = 1j * (kzk * ax - kxi * az)
                wz[i, j, k] = 1j * (kxi * ay - kyj * ax)


@njit(**_OPTS)
def project_solenoidal(vx, vy, vz, kx, ky, kz):
    """Project a vector field onto its divergence-free part, in place.

    Equivalent to ``project`` in numerics.f90: subtract the pressure gradient
    so that k . u = 0. With p = i(k.u)/k^2 and u += i k p the net update is
    u -= k (k.u)/k^2; the factors of i cancel.
    """
    a, b, c = vx.shape
    for i in prange(a):
        kxi = kx[i]
        for j in range(b):
            kyj = ky[j]
            for k in range(c):
                kzk = kz[k]
                ksq = kxi * kxi + kyj * kyj + kzk * kzk
                if ksq > 0.0:
                    div = (kxi * vx[i, j, k] + kyj * vy[i, j, k]
                           + kzk * vz[i, j, k]) / ksq
                    vx[i, j, k] -= kxi * div
                    vy[i, j, k] -= kyj * div
                    vz[i, j, k] -= kzk * div


@njit(**_OPTS)
def add_viscous(rhs, fu, visc, k2):
    """Add the viscous term to one component's RHS: rhs += -nu(k) k^2 fu.

    ``k2`` is the local squared-wavenumber array (same shape as ``fu``) and
    ``visc`` is this component's kinematic viscosity on the same grid, so the
    damping rate may vary from mode to mode (uniform molecular viscosity is
    the constant-array case). Generalises compute_diffusive_terms in
    numerics.f90, which carries a single scalar per component.
    """
    a, b, c = fu.shape
    for i in prange(a):
        for j in range(b):
            for k in range(c):
                rhs[i, j, k] += -visc[i, j, k] * k2[i, j, k] * fu[i, j, k]


@njit(**_OPTS)
def rk_stage(fu, rhs, accum, c_accum, half_or_full, stage_state):
    """One Runge-Kutta substage update for a single component.

    Mirrors the inner loops of runge_kutta4: accumulate the weighted RHS into
    ``accum`` (the running sum that becomes the final increment) and form the
    next stage state in ``stage_state``::

        accum       += c_accum     * rhs
        stage_state  = fu + half_or_full * rhs

    All arrays share the same shape; ``c_accum`` and ``half_or_full`` are
    scalars (dt-weighted coefficients supplied by the caller).
    """
    a, b, c = fu.shape
    for i in prange(a):
        for j in range(b):
            for k in range(c):
                r = rhs[i, j, k]
                accum[i, j, k] += c_accum * r
                stage_state[i, j, k] = fu[i, j, k] + half_or_full * r

