"""HDF5 field output and XDMF descriptors (I/O subset of hdf5.f90).

Writes the velocity field to ``output.NNNNNN.h5`` as a single vector dataset
``/u`` of shape ``(N, N, N, 3)`` plus the scalar metadata ``/time`` and
``/nfile``, mirroring ``write_hdf5_compute_file``. A companion
``output.NNNNNN.xmf`` XDMF file makes the result loadable in ParaView/VisIt
(port of ``write_xdmf_file``). The restart reader ``read_field`` is the inverse
of the writer.

Layout: ``/u`` is component-last with the spatial axes ordered ``(z, y, x)`` --
element ``u[gz, gy, gx, c]`` is velocity component ``c`` at grid point
``(gx, gy, gz)``. This is byte-compatible with the Fortran's HDF5 output (the
Fortran writes a column-major ``(3, nx, ny, nz)`` array, which the HDF5 Fortran
layer stores as C-order ``(nz, ny, nx, 3)``), so Python and Fortran files
interchange directly and the XDMF ``Dimensions="N N N 3"`` is correct in any
ParaView reader. ``read_field`` accepts either code's files unchanged.

Parallel decomposition: each rank owns the subdomain ``state.T.real_slice`` of
the global box. Output is assembled on the root rank by gathering those
subdomains; restart broadcasts the global field and every rank keeps its own
slice. This favours simplicity over scalability — adequate for the CPU/dev
runs this port targets — rather than collective parallel HDF5.
"""

from __future__ import annotations

import numpy as np

try:
    import h5py
except Exception as exc:  # pragma: no cover - h5py is a hard dependency here
    raise ImportError("hdf5_io requires the 'h5py' package") from exc

from . import kernels as K
from .backend import COMM, IS_ROOT, SIZE
from .data import State


# ----------------------------------------------------------------------
# Distributed <-> global helpers
# ----------------------------------------------------------------------
def _gather_field(state: State, local: np.ndarray):
    """Collect a rank-local real field into the global array on the root rank.

    Returns the full ``(N, N, N)`` array on root and ``None`` elsewhere. In the
    serial case the local array already spans the whole box.
    """
    local = np.ascontiguousarray(local)
    if SIZE == 1:
        return np.array(local)

    parcels = COMM.gather((state.T.real_slice, local), root=0)
    if not IS_ROOT:
        return None
    full = np.empty((state.n, state.n, state.n), dtype=np.float64)
    for sl, block in parcels:
        full[sl] = block
    return full


def _velocity_to_physical(state: State):
    """Inverse-transform the spectral velocity into three fresh real arrays."""
    return [state.T.backward(state.fu[c], np.empty(state.T.real_shape))
            for c in range(3)]


def _vorticity_magnitude(state: State) -> np.ndarray:
    """Physical-space ``|omega|`` from the current spectral velocity.

    Turbulent structure shows up in the vorticity, not the (smooth) velocity
    magnitude, so each output frame carries this scalar for visualisation. Uses
    fresh buffers to avoid touching the solver's work arrays.
    """
    fw = [np.empty(state.T.cmplx_shape, dtype=np.complex128) for _ in range(3)]
    K.spectral_curl(state.fu[0], state.fu[1], state.fu[2],
                    state.kx, state.ky, state.kz, fw[0], fw[1], fw[2])
    w = [state.T.backward(fw[c], np.empty(state.T.real_shape)) for c in range(3)]
    return np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def write_field(state: State, time: float, nfile: int,
                prefix: str = "output") -> str | None:
    """Write the velocity field to ``<prefix>.NNNNNN.h5`` plus its XDMF file.

    The HDF5 file is created only on the root rank. Returns the file name on
    root and ``None`` on the other ranks.
    """
    comps = _velocity_to_physical(state)
    gathered = [_gather_field(state, comps[c]) for c in range(3)]
    wmag = _gather_field(state, _vorticity_magnitude(state))

    fname = None
    if IS_ROOT:
        # Component-last with spatial axes (z, y, x) to match the Fortran HDF5
        # layout (see module docstring): gathered[c] is [gx, gy, gz].
        stacked = np.stack(gathered, axis=-1)            # (Nx, Ny, Nz, 3)
        data = np.ascontiguousarray(stacked.transpose(2, 1, 0, 3))  # (Nz,Ny,Nx,3)
        wdata = np.ascontiguousarray(wmag.transpose(2, 1, 0))       # (Nz,Ny,Nx)
        fname = f"{prefix}.{nfile:06d}.h5"
        with h5py.File(fname, "w") as fh:
            fh.create_dataset("u", data=data.astype(np.float64))
            fh.create_dataset("w", data=wdata.astype(np.float64))
            fh.create_dataset("time", data=np.array([time], dtype=np.float64))
            fh.create_dataset("nfile", data=np.array([nfile], dtype=np.int32))
        _write_xdmf(state.n, fname, nfile, prefix)

    if SIZE > 1:
        COMM.Barrier()
    return fname


def _write_xdmf(n: int, h5name: str, nfile: int, prefix: str) -> None:
    """Emit the XDMF wrapper describing the ``/u`` vector dataset."""
    xmf = f"{prefix}.{nfile:06d}.xmf"
    text = f"""<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
  <Domain>
    <Grid Name="Grid">
      <Geometry Origin="" Type="ORIGIN_DXDYDZ">
        <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="8">0 0 0</DataItem>
        <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="8">1 1 1</DataItem>
      </Geometry>
      <Topology NumberOfElements="{n} {n} {n}" Type="3DCoRectMesh"/>
      <Attribute Center="Node" Name="u" Type="Vector">
        <DataItem DataType="Float" Precision="8" Dimensions="{n} {n} {n} 3" Format="HDF">{h5name}:/u</DataItem>
      </Attribute>
      <Attribute Center="Node" Name="w" Type="Scalar">
        <DataItem DataType="Float" Precision="8" Dimensions="{n} {n} {n}" Format="HDF">{h5name}:/w</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    with open(xmf, "w") as fh:
        fh.write(text)


# ----------------------------------------------------------------------
# Reading (restart)
# ----------------------------------------------------------------------
def read_field(state: State, filename: str) -> float:
    """Load a velocity field from ``filename`` into ``state.fu`` (spectral).

    Reads the ``(N, N, N, 3)`` component-last ``/u`` dataset (the layout written
    by :func:`write_field` and by the Fortran code, so either is accepted). The
    global field is read on root, broadcast, and each rank forward-transforms
    its own ``real_slice``. Returns the stored simulation time (zero if the file
    predates the timestamp convention).
    """
    if IS_ROOT:
        with h5py.File(filename, "r") as fh:
            data = np.asarray(fh["u"][...], dtype=np.float64)  # (Nz,Ny,Nx,3)
            time = float(fh["time"][0]) if "time" in fh else 0.0
    else:
        data = None
        time = 0.0

    if SIZE > 1:
        data = COMM.bcast(data, root=0)
        time = COMM.bcast(time, root=0)

    sl = state.T.real_slice
    for c in range(3):
        comp = data[..., c].transpose(2, 1, 0)  # [gz,gy,gx] -> [gx,gy,gz]
        local = np.ascontiguousarray(comp[sl])
        state.T.forward(local, state.fu[c])
    state.apply_mask(state.fu)
    return time
