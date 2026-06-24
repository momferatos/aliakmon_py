"""Initial conditions (initial_conditions.f90, hydro subset).

Provides the velocity initial conditions used by the hydro solver: zero,
stochastic (flat / with spectrum), Orszag-Tang, ABC (Beltrami) flow and the
Taylor-Green vortex. MHD and passive-scalar pieces of the originals are
dropped.

Analytic fields are built in physical space on the local subdomain and
forward-transformed; every IC is then projected onto the divergence-free
subspace and truncated, so the velocity that enters the solver is solenoidal
and band-limited regardless of discretisation round-off.
"""

from __future__ import annotations

import numpy as np

from . import kernels as K
from . import numerics as N
from .data import State
from .parameters import InitCond


def _coords(state: State):
    """Broadcastable physical coordinate arrays (X, Y, Z) for this rank."""
    T = state.T
    return T.x[:, None, None], T.y[None, :, None], T.z[None, None, :]


def _from_real(state: State, comps) -> None:
    """Forward-transform three real components, then project and truncate."""
    for c in range(3):
        state.T.forward(np.ascontiguousarray(comps[c]), state.fu[c])
    state.apply_mask(state.fu)
    K.project_solenoidal(state.fu[0], state.fu[1], state.fu[2],
                         state.kx, state.ky, state.kz)
    state.apply_mask(state.fu)


def normalize_ms(state: State, target_ms: float = 1.0) -> None:
    """Rescale the velocity to a target mean-square <|u|^2> (rescale()).

    The same scalar factor multiplies every component, so the field stays
    divergence-free.
    """
    ms = N.mean_square_velocity(state)
    if ms > 0.0:
        factor = np.sqrt(target_ms / ms)
        for c in range(3):
            state.fu[c] *= factor


# ----------------------------------------------------------------------
# Analytic flows
# ----------------------------------------------------------------------
def zero_field(state: State) -> None:
    for c in range(3):
        state.fu[c][...] = 0.0


def taylor_green(state: State) -> None:
    """Classic Taylor-Green vortex (exactly divergence-free)."""
    X, Y, Z = _coords(state)
    shape = state.T.real_shape
    u0 = np.broadcast_to(np.sin(X) * np.cos(Y) * np.cos(Z), shape)
    u1 = np.broadcast_to(-np.cos(X) * np.sin(Y) * np.cos(Z), shape)
    u2 = np.zeros(shape)
    _from_real(state, [u0, u1, u2])


def abc_flow(state: State, a: float = 0.1, b: float = 0.3, c: float = 0.4,
             k1: int = 1, k2: int = 3, abc_rand: float = 0.1,
             seed: int | None = None) -> None:
    """Arnold-Beltrami-Childress initial condition (abc_flow, k=k1..k2).

    A *single* ABC mode is a Beltrami flow (omega = k u), so ``u x omega == 0``
    and the dynamics collapse to pure viscous diffusion. The original code
    avoids that two ways, both reproduced here:

    1. **Superpose** modes ``kk = k1..k2`` with a cyclic component permutation
       (the ``ii`` mixing in the Fortran), so the field is no longer a single
       curl eigenfunction and the nonlinear term is non-zero.
    2. **Add a stochastic small-scale perturbation** at ``abc_rand`` (10%) of
       unit-rms amplitude to seed the instability.

    The base ABC components for wavenumber ``kk`` are
    ``g1 = a sin(kk z) + c cos(kk y)``, ``g2 = b sin(kk x) + a cos(kk z)``,
    ``g3 = c sin(kk y) + b cos(kk x)``; the cyclic mixing assigns
    ``u = g1(k1)+g2(k1+1)+g3(k1+2)`` and rotations thereof per component.
    """
    # Stochastic perturbation first (random_field writes state.fu), rescaled to
    # unit mean-square like the Fortran `rescale(scratch)`. With kmax_cut=2 this
    # is a large-scale perturbation on the (+-1,+-1,+-1) modes, matching the
    # Fortran abc_flow's `random_field(nn, scratch, 2.0)`.
    random_field(state, kmax_cut=2.0, seed=seed)
    normalize_ms(state, 1.0)
    perturbation = [state.fu[c].copy() for c in range(3)]

    # Large-scale ABC superposition, built in physical space.
    X, Y, Z = _coords(state)
    shape = state.T.real_shape
    comps = [np.zeros(shape), np.zeros(shape), np.zeros(shape)]
    for kk in range(k1, k2 + 1):
        g = (a * np.sin(kk * Z) + c * np.cos(kk * Y),   # g1
             b * np.sin(kk * X) + a * np.cos(kk * Z),   # g2
             c * np.sin(kk * Y) + b * np.cos(kk * X))   # g3
        # Cyclic permutation: component m of mode kk takes g[(m + kk - k1) % 3].
        for m in range(3):
            comps[m] = comps[m] + np.broadcast_to(g[(m + kk - k1) % 3], shape)

    for c in range(3):
        state.T.forward(np.ascontiguousarray(comps[c]), state.fu[c])

    # Add the perturbation, then enforce solenoidality and truncate.
    for c in range(3):
        state.fu[c] += abc_rand * perturbation[c]
    state.apply_mask(state.fu)
    K.project_solenoidal(state.fu[0], state.fu[1], state.fu[2],
                         state.kx, state.ky, state.kz)
    state.apply_mask(state.fu)


def orszag_tang(state: State) -> None:
    """Orszag-Tang velocity field (the hydro part; divergence-free)."""
    X, Y, _ = _coords(state)
    shape = state.T.real_shape
    u0 = np.broadcast_to(-2.0 * np.sin(Y), shape)
    u1 = np.broadcast_to(2.0 * np.sin(X), shape)
    u2 = np.broadcast_to(np.sin(X) + np.sin(Y), shape)
    _from_real(state, [u0, u1, u2])


def _finalize_stochastic(state: State) -> None:
    """Mask, project solenoidal, mask again — shared tail of the generators."""
    state.apply_mask(state.fu)
    K.project_solenoidal(state.fu[0], state.fu[1], state.fu[2],
                         state.kx, state.ky, state.kz)
    state.apply_mask(state.fu)


def random_field(state: State, kmax_cut: float = 2.0,
                 seed: int | None = None) -> None:
    """Flat random spectrum below a hard cutoff (Fortran ``random_field``).

    Fills only the modes with ``|k| < kmax_cut`` *and* all three wavenumber
    components nonzero (the ``trk1/=0 .and. ...`` condition), with a flat
    (white) random spectrum. Built from real white noise so the field is
    Hermitian, then band-limited, projected solenoidal and truncated. With the
    default ``kmax_cut=2`` only the ``(+-1,+-1,+-1)`` modes survive, i.e. a
    purely large-scale perturbation.
    """
    rng = np.random.default_rng(seed)
    band = np.broadcast_to(
        (state.KX != 0) & (state.KY != 0) & (state.KZ != 0)
        & (np.sqrt(state.K2) < kmax_cut), state.T.cmplx_shape)
    for c in range(3):
        noise = rng.standard_normal(state.T.real_shape)
        state.T.forward(np.ascontiguousarray(noise), state.fu[c])
        state.fu[c] *= band
    _finalize_stochastic(state)


def erandom_field(state: State, seed: int | None = None) -> None:
    """Random field with a decaying power-law spectrum (Fortran ``erandom_field``).

    Random phases with coefficient magnitude ``~ k^(-11/6)`` (the Fortran's
    ``sqrt(k^-2 * k^-5/3)``), giving an energy spectrum ``E(k) ~ k^(-11/3)``.
    The mean mode is left at zero.
    """
    rng = np.random.default_rng(seed)
    kmag = np.sqrt(state.K2_nonzero)
    env = np.where(state.K2 == 0.0, 0.0, kmag ** (-11.0 / 6.0))
    for c in range(3):
        noise = rng.standard_normal(state.T.real_shape)
        state.T.forward(np.ascontiguousarray(noise), state.fu[c])
        state.fu[c] *= env
    _finalize_stochastic(state)


# ----------------------------------------------------------------------
# Dispatcher (set_initial_conditions)
# ----------------------------------------------------------------------
def set_initial_conditions(state: State) -> None:
    """Initialise ``state.fu`` according to ``cfg.initcond``."""
    cfg = state.cfg
    ic = cfg.initcond
    seed = 0 if cfg.seedrandom else None

    if ic == InitCond.ZERO:
        zero_field(state)
    elif ic == InitCond.STOCHASTIC_FLAT:
        random_field(state, kmax_cut=2.0, seed=seed)
    elif ic == InitCond.STOCHASTIC_WITH_SPECTRUM:
        erandom_field(state, seed=seed)
    elif ic == InitCond.ORSZAG_TANG_VORTEX:
        orszag_tang(state)
    elif ic == InitCond.ABC:
        abc_flow(state, seed=seed)
    elif ic == InitCond.TAYLOR_GREEN_VORTEX:
        taylor_green(state)
    else:
        raise ValueError(f"unsupported initial condition {ic!r}")
