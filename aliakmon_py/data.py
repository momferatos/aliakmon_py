"""Field storage, wavenumbers and the truncation mask (data.f90).

Holds all distributed fields for a run plus the spectral bookkeeping: local
wavenumber axes (from :class:`~aliakmon_py.transforms.Transforms`), the
active-mode mask with its three truncation criteria (``twothirds_tr``,
``spherical_tr``, ``polyhedral_tr``) and the Patterson-Orszag phase-shift
factor used for dealiasing (``make_phases_array``).

One departure from the Fortran: an LES run always truncates spherically, at
the subgrid model's own cutoff ``k_c``, so that the numerical cutoff and the
LES cutoff are the same wavenumber. See :meth:`State._effective_truncation`.

Each field is stored as a list of three component arrays (one per velocity
component) so the Numba kernels receive plain contiguous arrays.
"""

from __future__ import annotations

import math

import numpy as np

from .backend import allreduce_max, allreduce_sum, on_root
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
        # Last subgrid-model report (numerics.update_eddy_viscosity); None
        # for a pure DNS run.
        self.les_info = None

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

        # Structural subgrid model scratch (aliakmon_py.sgs). None unless such
        # a model runs, so DNS and eddy-viscosity runs pay nothing for it.
        self.sgs_ubar = self.sgs_ustar = self.sgs_tau = None
        self.sgs_tau_hat = self.sgs_grad = self.sgs_cmplx = None
        self.fsgs = None
        if cfg.les_tensor:
            self._alloc_tensor_sgs()

    # ------------------------------------------------------------------
    # Structural subgrid-model buffers
    # ------------------------------------------------------------------
    def _alloc_tensor_sgs(self) -> None:
        """Allocate the work buffers a structural (tensor) LES model needs.

        Symmetric tensors are stored as six components in the order
        ``(11, 12, 13, 22, 23, 33)``, matching the kernel signatures; the
        velocity gradient ``sgs_grad[i][j]`` is ``dU_i/dx_j``.

        This is 12 extra real fields (21 with the backscatter clip) and 10
        complex ones per rank — roughly a doubling of the solver's footprint,
        which is the price of carrying a stress tensor rather than a scalar
        eddy viscosity.
        """
        T = self.T
        self.sgs_ubar = _triple_real(T)     # resolved velocity, physical space
        self.sgs_ustar = _triple_real(T)    # predicted full field u*
        self.sgs_tau = [np.zeros(T.real_shape, dtype=np.float64)
                        for _ in range(6)]
        self.sgs_tau_hat = [np.zeros(T.cmplx_shape, dtype=np.complex128)
                            for _ in range(6)]
        self.fsgs = _triple_cmplx(T)        # subgrid force, Fourier space
        self.sgs_cmplx = np.zeros(T.cmplx_shape, dtype=np.complex128)
        # The resolved strain is only needed to decide where to clip.
        if self.cfg.les_clip_backscatter:
            self.sgs_grad = [[np.zeros(T.real_shape, dtype=np.float64)
                              for _ in range(3)] for _ in range(3)]

    # ------------------------------------------------------------------
    # Active-mode mask (alloc_init loop + *_tr criteria in data.f90)
    # ------------------------------------------------------------------
    def _effective_truncation(self) -> Truncation:
        """The truncation scheme actually used; an LES run overrides the config.

        A subgrid model closes the equations behind a *sharp spectral filter*:
        it is defined for a field carrying every mode inside a sphere of radius
        ``k_c`` and nothing outside it. Neither the two-thirds mask (a box) nor
        the polyhedral one is such a sphere, so with an LES model on, the
        truncation is spherical whatever ``[numerics] truncation`` asks for.
        """
        cfg = self.cfg
        if not cfg.les_active or cfg.truncation == Truncation.SPHERICAL:
            return cfg.truncation
        on_root(f"LES {cfg.les_model.name}: [numerics] truncation = "
                f"{cfg.truncation.name} overridden with SPHERICAL — the "
                f"subgrid model is defined behind a sharp spectral filter, so "
                f"the resolved band must be the sphere |k| <= k_c.")
        return Truncation.SPHERICAL

    def _setup_mask(self) -> None:
        n = self.n
        akx, aky, akz = np.abs(self.KX), np.abs(self.KY), np.abs(self.KZ)
        trunc = self.truncation = self._effective_truncation()

        if trunc == Truncation.TWO_THIRDS:
            kcut = n // 3
            truncated = (akx > kcut) | (aky > kcut) | (akz > kcut)
            self.kmax = n // 3
        elif trunc == Truncation.SPHERICAL:
            # Nominal radius first: it is the grid's resolving power, and an
            # LES cutoff that falls back on it needs it already set.
            self.kmax = TRFAC * n
            kabs = np.sqrt(self.K2)
            if self.cfg.les_active:
                # Numerical cutoff == LES cutoff. The solver then evolves
                # exactly the band the subgrid model closes: no mode above
                # k_c, which the model does not represent and (for a trained
                # one) never saw carrying energy, and no mode below it
                # discarded. Imported here because sgs imports this module.
                from . import sgs as SGS
                # Inclusive, |k| <= k_c: this is the band the sharp filter in
                # sgs keeps, and the one pDDLES trained against.
                truncated = kabs > SGS.les_cutoff(self)
            else:
                truncated = kabs >= self.kmax
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

        # Optional explicit spectral cutoff, applied on top of the scheme's
        # own limit — for cutting the resolved band by hand in a DNS run, or
        # *below* k_c in an LES one. An LES needs nothing here to match its
        # model's filter: the spherical branch above already truncates at k_c.
        #
        # `self.kmax` is deliberately left at the scheme's nominal value: it
        # stands for the grid's resolving power, which is what sets the
        # molecular viscosity in `setup_viscosity` and the `kmax*eta`
        # diagnostic. Those must keep describing the underlying DNS
        # resolution, not the (much lower) LES cutoff.
        ktrunc = float(self.cfg.truncation_kc)
        if ktrunc > 0.0:
            truncated = truncated | (np.sqrt(self.K2) > ktrunc)

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
