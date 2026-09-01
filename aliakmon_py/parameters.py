"""Constants, enumerations and the run configuration.

Port of the Fortran ``types`` and ``parameters`` modules (parameters.f90),
restricted to the hydrodynamic solver. MHD, radiation, particle and
passive-scalar parameters are intentionally omitted.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, fields
from enum import IntEnum
from pathlib import Path

# TOML reader: stdlib tomllib on 3.11+, tomli backport otherwise.
if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover - depends on interpreter version
    import tomli as _toml


# --------------------------------------------------------------------------
# Enumerations (mirrors the integer parameter constants in parameters.f90)
# --------------------------------------------------------------------------
class InitCond(IntEnum):
    ZERO = 0
    STOCHASTIC_FLAT = 1
    STOCHASTIC_WITH_SPECTRUM = 2
    ORSZAG_TANG_VORTEX = 3
    ABC = 4
    TAYLOR_GREEN_VORTEX = 5
    RADSPHERE = 6  # radiation IC; unused in the hydro port


class BoundCond(IntEnum):
    PERIODIC = 0
    FREESLIP = 1


class Integration(IntEnum):
    EULER = 0
    RUNGE_KUTTA2 = 1
    RUNGE_KUTTA4 = 2


class Truncation(IntEnum):
    TWO_THIRDS = 0
    SPHERICAL = 1
    POLYHEDRAL = 2


class Dealiasing(IntEnum):
    NONE = 0
    PATTERSON_ORSZAG = 1


class Forcing(IntEnum):
    STOCHASTIC = 0
    KANEDA = 1


class LESModel(IntEnum):
    """Subgrid model closing the filtered equations (none = pure DNS).

    Two families, which enter the solver at completely different points:

    * *functional* models represent the subgrid stress by an eddy viscosity
      ``nu_t(k)``, which is folded into ``State.visc`` and rides along in the
      existing diffusion term;
    * *structural* models build the subgrid stress tensor ``tau_ij`` itself and
      enter as the force ``-d(tau_ij)/dx_j`` added to the right-hand side. No
      eddy viscosity is involved. See :mod:`aliakmon_py.sgs`.
    """

    NONE = 0
    CHOLLET_LESIEUR = 1     # functional: spectral eddy viscosity
    PDDLES = 2              # structural: predicted-field deconvolution LES


# Which LES models supply an eddy viscosity, and which build a stress tensor.
EDDY_VISCOSITY_MODELS = frozenset({LESModel.CHOLLET_LESIEUR})
TENSOR_MODELS = frozenset({LESModel.PDDLES})


# --------------------------------------------------------------------------
# Physical / mathematical constants (parameters.f90 module-level parameters)
# --------------------------------------------------------------------------
PI = math.pi
LBOX = 2.0 * PI          # length of the simulation box
SMALL = 1.0e-16          # guard against division by zero
KENTAR = 0.5             # target kinetic energy
D = 0.4                  # parameter for target Reynolds-number estimate
RMSUTAR = math.sqrt(1.0 / 3.0)  # target rms velocity (sqrt(2/3 * KENTAR))

# Field-component indices (hydro: three velocity components, 0-based here
# versus the 1-based nu1/nu2/nu3 of the Fortran code).
NU1, NU2, NU3 = 0, 1, 2
NFIELDS = 3


def dim1(n: int) -> int:
    """X-dimension size of the packed real-FFT array.

    Mirrors ``dim1`` in parameters.f90: ``2*(floor(n/2)+1)``. With CuPy's
    ``rfftn`` the equivalent complex extent on the transformed axis is
    ``n // 2 + 1``; this helper is kept for parity / buffer sizing.
    """
    return 2 * (n // 2 + 1)


# --------------------------------------------------------------------------
# Run configuration loaded from config.toml
# --------------------------------------------------------------------------
@dataclass
class Config:
    """Flat view of the hydro-relevant namelist parameters.

    Field names match the lowercase TOML keys. Enum-valued integers are
    stored as their IntEnum type for readability at the call site.
    """

    # [general]
    n: int = 64
    timesteps: int = 0
    tmax: float = 10.0
    cfl: float = 0.8
    re: float = 0.0
    kmaxeta: float = 2.0
    stop_at_disspeak: bool = False
    valid: bool = False

    # [hydro]
    viscous: bool = True
    burgers: bool = False

    # [force]
    forced: bool = False
    variable_forcing: bool = True
    kforcing: float = 2.5

    # [les]
    les_model: LESModel = LESModel.NONE
    les_ck: float = 1.4
    # The filter cutoff, either as the wavenumber itself or as the fraction of
    # the grid corner that pDDLES's `alpha` also measures (see sgs).
    les_kc: float = 0.0
    les_alpha: float = 0.0
    # pDDLES only; ignored by the eddy-viscosity models.
    les_pddles_model: str = ""
    les_pddles_source: str = ""
    les_pddles_scaler: str = ""
    les_pddles_device: str = "cpu"
    les_clip_backscatter: bool = False

    # [numerics]
    integration_method: Integration = Integration.RUNGE_KUTTA4
    truncation: Truncation = Truncation.POLYHEDRAL
    truncation_kc: float = 0.0
    dealiasing: Dealiasing = Dealiasing.PATTERSON_ORSZAG

    # [initialconditions]
    initcond: InitCond = InitCond.ABC
    kinitcond: float = 2.5
    seedrandom: bool = False
    boundcond: BoundCond = BoundCond.FREESLIP

    # [inputoutput]
    outputfiles: bool = True
    input_field: bool = False
    input_field_filename: str = "output.888888.h5"
    nfilestart: int = 0
    hdf5frate: float = 10.0
    compression_level: int = 0
    hdf5_overwrite: bool = False

    # Maps a config field to the enum it should be coerced into.
    _ENUMS = {
        "integration_method": Integration,
        "truncation": Truncation,
        "dealiasing": Dealiasing,
        "initcond": InitCond,
        "boundcond": BoundCond,
        "les_model": LESModel,
    }

    @property
    def les_active(self) -> bool:
        """True when any subgrid model is running."""
        return self.les_model != LESModel.NONE

    @property
    def les_eddy_viscosity(self) -> bool:
        """True for a functional model that contributes ``nu_t(k)``."""
        return self.les_model in EDDY_VISCOSITY_MODELS

    @property
    def les_tensor(self) -> bool:
        """True for a structural model that builds ``tau_ij`` explicitly."""
        return self.les_model in TENSOR_MODELS

    @property
    def diffusive(self) -> bool:
        """True when the RHS carries a ``nu(k) k^2`` diffusion term at all.

        An eddy-viscosity LES supplies a viscosity even with ``viscous = false``
        (no molecular part), so the diffusive term must not be skipped then. A
        structural model never touches ``nu(k)`` — it adds a separate subgrid
        force — so there ``viscous = false`` really does mean no diffusion.
        """
        return self.viscous or self.les_eddy_viscosity

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load a Config from a TOML file laid out like aliakmon.nml."""
        with open(path, "rb") as fh:
            raw = _toml.load(fh)

        # Flatten the [section] tables into a single namespace, matching how
        # the Fortran namelist exposes every key globally.
        flat: dict[str, object] = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)

        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in flat.items() if k in known}

        unknown = set(flat) - known
        if unknown:
            # Silently ignored keys hide typos; surface them instead.
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        for name, enum_cls in cls._ENUMS.items():
            if name in kwargs:
                kwargs[name] = enum_cls(kwargs[name])

        return cls(**kwargs)
