"""Distributed FFT layer (replacement for the heFFTe wrapper, fft_heffte.f90).

Wraps ``mpi4py-fft``'s :class:`PFFT` to provide forward/backward real FFTs
with the same slab/pencil decomposition the Fortran code uses. It also derives
the *local* integer wavenumber axes for this rank from the FFT's local slice,
which the Numba kernels and the truncation mask consume.

Decomposition (for a single grid dimension ``N``):
  * real space    -> distributed on axis 0 (x); local ``(nx_local, N, N)``
  * Fourier space -> distributed on axis 1 (ky); local
    ``(N, ky_local, N//2 + 1)``

Because the box length is ``2*pi``, the physical wavenumber of a mode equals
its integer mode number, so the axes hold plain (float) integers.
"""

from __future__ import annotations

import numpy as np
from mpi4py_fft import PFFT, newDistArray

from .backend import COMM


class Transforms:
    """Per-run distributed real FFT plus local spectral bookkeeping."""

    def __init__(self, n: int):
        self.n = int(n)
        N = (self.n, self.n, self.n)
        # r2c over all three axes; grid=(-1,) lets mpi4py-fft pick the
        # decomposition (slab on a single axis, pencils on more ranks).
        self.fft = PFFT(COMM, N, axes=(0, 1, 2), dtype=float, grid=(-1,))

        self.real_shape = self.newreal().shape
        self.cmplx_shape = self.newcmplx().shape

        # Real-space global index ranges owned by this rank (distributed on x).
        # Used to build physical coordinates for analytic initial conditions.
        self.real_slice = self.fft.local_slice(False)
        dx = 2.0 * np.pi / self.n  # LBOX / N
        ax = np.arange(self.n) * dx
        self.x = np.ascontiguousarray(ax[self.real_slice[0]])
        self.y = np.ascontiguousarray(ax[self.real_slice[1]])
        self.z = np.ascontiguousarray(ax[self.real_slice[2]])

        # Global index ranges owned by this rank in Fourier space.
        sl = self.fft.local_slice(True)
        kx_full = (np.fft.fftfreq(self.n) * self.n).astype(np.float64)
        ky_full = (np.fft.fftfreq(self.n) * self.n).astype(np.float64)
        kz_half = (np.fft.rfftfreq(self.n) * self.n).astype(np.float64)

        # 1-D local wavenumber axes matching the three dims of a Fourier field.
        self.kx = np.ascontiguousarray(kx_full[sl[0]])
        self.ky = np.ascontiguousarray(ky_full[sl[1]])
        self.kz = np.ascontiguousarray(kz_half[sl[2]])

        # Broadcastable components and squared modulus over the local field.
        self.KX = self.kx[:, None, None]
        self.KY = self.ky[None, :, None]
        self.KZ = self.kz[None, None, :]
        self.K2 = (self.KX ** 2 + self.KY ** 2 + self.KZ ** 2)

    # -- array factories ------------------------------------------------
    def newreal(self) -> np.ndarray:
        """A zeroed real-space distributed array (local subdomain)."""
        a = newDistArray(self.fft, forward_output=False)
        a[...] = 0.0
        return a

    def newcmplx(self) -> np.ndarray:
        """A zeroed Fourier-space distributed array (local subdomain)."""
        a = newDistArray(self.fft, forward_output=True)
        a[...] = 0.0
        return a

    # -- transforms -----------------------------------------------------
    def forward(self, u: np.ndarray, uh: np.ndarray) -> np.ndarray:
        """Forward real FFT (normalised): physical ``u`` -> spectral ``uh``."""
        self.fft.forward(u, uh)
        return uh

    def backward(self, uh: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Inverse real FFT: spectral ``uh`` -> physical ``u``."""
        self.fft.backward(uh, u)
        return u
