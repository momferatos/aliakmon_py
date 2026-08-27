"""Field storage, wavenumbers and the truncation mask (data.f90).

Holds all distributed fields for a run plus the spectral bookkeeping: local
wavenumber axes (from :class:`~aliakmon_py.transforms.Transforms`), the
active-mode mask with its three truncation criteria (``twothirds_tr``,
``spherical_tr``, ``polyhedral_tr``) and the Patterson-Orszag phase-shift
factor used for dealiasing (``make_phases_array``).

Each field is stored as a list of three component arrays (one per velocity
component) so the Numba kernels receive plain contiguous arrays.
"""

from __future__ import annotations

import math

import numpy as np

from .backend import allreduce_max, allreduce_sum
from .parameters import NFIELDS, PI, Config, Truncation
from .transforms import Transforms

# TRFAC = sqrt(2)/3 (aliakmon.f90:73): spherical/polyhedral truncation radius
# factor and the nominal k_max for those schemes.
TRFAC = math.sqrt(2.0) / 3.0


# Solver fields are plain NumPy arrays shaped to this rank's local subdomain.
# mpi4py-fft's PFFT holds the decomposition/transpose plan, so forward/backward
# accept plain ndarrays of the right local shape (verified for serial and
# multi-rank). Plain arrays also avoid DistArray's indexing quirks and feed
# straight into the Numba kernels.
def _triple_real(T: Transforms):
    return [np.zeros(T.real_shape, dtype=np.float64) for _ in range(NFIELDS)]


def _triple_cmplx(T: Transforms):
    return [np.zeros(T.cmplx_shape, dtype=np.complex128) for _ in range(NFIELDS)]


def _triple_spectral_real(T: Transforms):
    """Three real-valued arrays laid out on the *Fourier* grid (e.g. nu(k))."""
    return [np.zeros(T.cmplx_shape, dtype=np.float64) for _ in range(NFIELDS)]


class State:
    """All GPU-free, MPI-distributed fields and spectral metadata for a run."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n = int(cfg.n)
        self.T = Transforms(self.n)

        # Local spectral geometry (views onto the Transforms object).
        self.kx, self.ky, self.kz = self.T.kx, self.T.ky, self.T.kz
        self.KX, self.KY, self.KZ = self.T.KX, self.T.KY, self.T.KZ
        self.K2 = self.T.K2
        self.K2_nonzero = self.K2.copy()
        # Guard the mean mode (present only on the rank owning kx=ky=kz=0).
        self.K2_nonzero[self.K2_nonzero == 0.0] = 1.0

        self._setup_mask()
        self._make_phases()

        # Primary evolved state: velocity in Fourier space, per component.
        self.fu = _triple_cmplx(self.T)

        # Per-component kinematic viscosity nu_c(k), one real array per
        # velocity component on the local Fourier grid (set once Re is known).
        # A uniform molecular viscosity is just a constant array; a spectral
        # eddy viscosity varies with |k|. `nu_mol` keeps the scalar molecular
        # floor that the Taylor/Kolmogorov diagnostics are defined against.
        self.visc = _triple_spectral_real(self.T)
        self.nu_mol = 0.0

        # Kaneda et al. (2004) negative-viscosity forcing state.
        # fscale is updated each step by input_output._update_fscale.
        # _ke_prev mirrors the Fortran `save :: kenprev = 1.0_rk`.
        self.fscale = np.zeros(NFIELDS, dtype=np.float64)
        self._ke_prev = 1.0

        # Mean-square normalisation in Fourier space (MSFAC, aliakmon.f90:160).
        self.msfac = 1.0 / float(self.n ** 3)

        # Work buffers reused every RHS evaluation / timestep (data.f90 module
        # arrays). Allocated once; the solver writes into them in place.
        self.u_real = _triple_real(self.T)      # velocity in physical space
        self.w_real = _triple_real(self.T)      # vorticity in physical space
        self.fnl = _triple_cmplx(self.T)        # nonlinear term / RHS accum
        self.rhs = _triple_cmplx(self.T)        # right-hand side
        self.rks1 = _triple_cmplx(self.T)       # RK stage state
        self.rks2 = _triple_cmplx(self.T)       # RK running increment
        self.fw = _triple_cmplx(self.T)         # vorticity in Fourier space
        # Dealiasing scratch (only touched when Patterson-Orszag is active).
        self.du_real = _triple_real(self.T)
        self.psu_real = _triple_real(self.T)
        self.fnls = _triple_cmplx(self.T)

    # ------------------------------------------------------------------
    # Active-mode mask (alloc_init loop + *_tr criteria in data.f90)
    # ------------------------------------------------------------------
    def _setup_mask(self) -> None:
        n = self.n
        akx, aky, akz = np.abs(self.KX), np.abs(self.KY), np.abs(self.KZ)
        trunc = self.cfg.truncation

        if trunc == Truncation.TWO_THIRDS:
            kcut = n // 3
            truncated = (akx > kcut) | (aky > kcut) | (akz > kcut)
            self.kmax = n // 3
        elif trunc == Truncation.SPHERICAL:
            kmax = TRFAC * n
            truncated = np.sqrt(self.K2) >= kmax
            self.kmax = kmax
        elif trunc == Truncation.POLYHEDRAL:
            half = n // 2
            twothirds = (2 * n) // 3
            truncated = (akx >= half) | (aky >= half) | (akz >= half)
            for a, b in ((self.KX, self.KY), (self.KY, self.KZ),
                         (self.KX, self.KZ)):
                truncated = truncated | (np.abs(a + b) >= twothirds)
                truncated = truncated | (np.abs(a - b) >= twothirds)
            self.kmax = TRFAC * n
        else:  # pragma: no cover - guarded by the Truncation enum
            raise ValueError(f"unknown truncation {trunc!r}")

        # Broadcast to the full local Fourier shape and store as 0/1 for
        # cheap in-place masking in the kernels.
        self.mask = np.broadcast_to(~truncated, self.T.cmplx_shape).copy()

        # Global maximum active wavevector modulus (k_max in alloc_init).
        local_active = np.sqrt(self.K2)[~truncated]
        local_max = float(local_active.max()) if local_active.size else 0.0
        self.k_max_active = allreduce_max(local_max)
        self.nmodes = int(allreduce_sum(int(self.mask.sum())))
    
    def apply_truncation_mask(self, fields) -> None:
        """Zero every truncated mode of each component field (truncate())."""
        for f in fields:
            f *= self.mask

    # ------------------------------------------------------------------
    # Patterson-Orszag phase factor (make_phases_array in data.f90)
    # ------------------------------------------------------------------
    def _make_phases(self) -> None:
        # Half-grid-cell shift: dx/2 = (2*pi/N)/2 = pi/N per unit wavenumber.
        hdx = PI / self.n
        arg = hdx * (self.KX + self.KY + self.KZ)
        phase = np.exp(1j * arg)
        # |phase| == 1, so the inverse shift is the complex conjugate.
        self.phase = np.broadcast_to(phase, self.T.cmplx_shape).copy()
        self.iphase = np.conjugate(self.phase)
