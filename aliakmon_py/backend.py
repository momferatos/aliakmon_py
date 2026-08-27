"""Array backend and MPI parallel environment.

Compute model: distributed-memory CPU. Fields are decomposed across MPI
ranks (slab/pencil) exactly as in the Fortran code; transforms go through
``mpi4py-fft`` and the pointwise kernels run under Numba on each rank's local
subdomain.

This module is the Python analogue of the Fortran ``mpivars`` module. It
exposes the array library (``xp`` == NumPy) and a thin wrapper over the MPI
communicator with the reductions the diagnostics need.
"""

from __future__ import annotations

import numpy as xp  # the array library; named xp for brevity at call sites

try:
    from mpi4py import MPI

    _HAVE_MPI = True
except Exception:  # pragma: no cover - MPI should always be present here
    MPI = None
    _HAVE_MPI = False


class _SerialComm:
    """Minimal stand-in so the code runs without mpi4py (single rank)."""

    rank = 0
    size = 1

    def Barrier(self):  # noqa: N802 - match mpi4py API
        pass

    def allreduce(self, sendobj, op=None):
        return sendobj

    def bcast(self, obj, root=0):
        return obj


if _HAVE_MPI:
    COMM = MPI.COMM_WORLD
    _MAX = MPI.MAX
    _MIN = MPI.MIN
    _SUM = MPI.SUM
else:  # pragma: no cover
    COMM = _SerialComm()
    _MAX = "max"
    _MIN = "min"
    _SUM = "sum"

RANK = COMM.rank
SIZE = COMM.size
IS_ROOT = RANK == 0


def allreduce_max(value: float) -> float:
    """Global maximum of a scalar across all ranks."""
    return float(COMM.allreduce(value, op=_MAX))


def allreduce_min(value: float) -> float:
    """Global minimum of a scalar across all ranks."""
    return float(COMM.allreduce(value, op=_MIN))


def allreduce_sum(value: float) -> float:
    """Global sum of a scalar across all ranks (e.g. for mean quantities)."""
    return float(COMM.allreduce(value, op=_SUM))


def allreduce_array(arr):
    """Element-wise global sum of a NumPy array across all ranks.

    Returns ``arr`` unchanged in the serial (single-rank) case; otherwise uses
    the buffer-protocol ``Allreduce`` so binned diagnostics (energy spectra)
    can be assembled from every rank's local contribution.
    """
    if not _HAVE_MPI or SIZE == 1:
        return arr
    out = xp.empty_like(arr)
    COMM.Allreduce(arr, out, op=_SUM)
    return out


def on_root(msg: str) -> None:
    """Print only from the root rank, to avoid SIZE-fold duplicated output."""
    if IS_ROOT:
        print(msg, flush=True)


def backend_name() -> str:
    return f"numpy + MPI ({SIZE} rank{'s' if SIZE != 1 else ''})"
