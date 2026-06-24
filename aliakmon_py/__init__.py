"""ALIAKMON_py — a pseudo-spectral incompressible DNS solver.

Python port of the hydrodynamic core of ALIAKMON (modern Fortran). MHD and
radiative-transfer functionality from the original code are out of scope.
Fields are distributed across MPI ranks (slab/pencil) with NumPy arrays;
real FFTs go through ``mpi4py-fft`` and the pointwise kernels run under Numba.
"""

from .parameters import Config

__all__ = ["Config"]
__version__ = "0.0.1"
