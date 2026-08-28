"""Structural subgrid-scale models: the stress tensor built explicitly.

The functional model in :mod:`aliakmon_py.numerics` (Chollet-Lesieur) never
forms the subgrid stress: it represents it by an eddy viscosity ``nu_t(k)``
folded into the existing diffusion term. The model here does the opposite — it
constructs

    tau_ij = filtered(u_i u_j) - U_i U_j

and feeds the filtered momentum equation the corresponding force

    f_i = -d(tau_ij)/dx_j        ->        f_hat_i = -i k_j tau_hat_ij ,

which is added to the right-hand side. No viscosity of any kind is involved,
so ``tau_ij`` is free to be non-aligned with the resolved strain and to
backscatter — the two things an eddy viscosity structurally cannot do.

**pDDLES** (predicted-field deconvolution LES) closes this by *predicting* the
full field from the resolved large scales with a trained network and then
filtering the prediction:

    u*     = Predictor(U)                     (a trained PyTorch model)
    tau_ij = G*(u*_i u*_j) - G*(u*_i) G*(u*_j)

where ``U`` is the resolved velocity the solver evolves, ``u*`` the predicted
full field, and ``G`` the sharp spectral LES filter at ``k_c``. Both terms are
built from the *prediction*, matching ``TurbDataset.subgrid_scale_tensor`` in
the pDDLES tree; the stored tensor is made deviatoric there too. Substituting
the solver's own ``U`` for ``G*u*`` would look equivalent — the network is
trained so that ``G*u* ~ U`` — but the difference is exactly the
reconstruction error, and it would enter ``tau`` as a spurious stress instead
of cancelling.

Two consequences of a *trained* predictor shape the implementation:

* **The network is not MPI-decomposed.** Inference therefore runs on the
  assembled global field on the root rank, and the prediction is scattered back
  to the ranks' subdomains. Like :mod:`aliakmon_py.hdf5_io`, this favours
  simplicity over scalability.
* **Inference is expensive.** The stress is built once per timestep and frozen
  across the RK substages — the same compromise :func:`numerics.update_eddy_viscosity`
  already makes for ``nu_t``, and an O(dt) one. It also makes the subgrid
  dissipation diagnostic free, since the force used by the step is still in
  ``state.fsgs`` when the diagnostics run.

Checkpoint contract
-------------------
``les_pddles_model`` points at a pDDLES ``.pth`` checkpoint — the dict
``{epoch, model, optimizer, args}`` that ``trainer.save`` writes. It is
self-describing: the training ``argparse.Namespace`` travels with the weights,
so the architecture, its hyper-parameters and the filter cutoff ``alpha`` are
all read from the file rather than restated in ``config.toml``. Only the
device and the data-loading flags are overridden.

Three things follow, and each is a way to get silently wrong answers:

* **The architecture is imported from the pDDLES tree**, ``lib/arch/<arch>.py``
  — point ``les_pddles_source`` at a checkout. ALIAKMON does not vendor it.
* **The scaler is not in the checkpoint.** pDDLES normalises outside the
  network (``trainer.train_one_epoch``), with constants fitted over the
  training set and saved beside the data as ``norm.pt`` / ``minmax.pt``. They
  are loaded here and applied around the forward pass; without them the
  network sees inputs far outside its training range. ``les_pddles_scaler``
  overrides the path if the data has moved, or takes the literal ``"auto"``
  to stand in for a missing file by standardising each component with the
  field's own mean and variance. ``"auto"`` tracks the running solution
  rather than the training set, so it is for getting a run moving and for
  diagnostics, not for production.
* **The filter must match training.** ``k_c`` is derived from the checkpoint's
  ``alpha`` as ``alpha * sqrt(3)/2 * n``, so ``tau`` is built with the filter
  the network was trained against. Setting ``les_kc`` overrides this, which is
  usually a mistake.

The network itself takes the physical-space velocity as float32
``(1, 3, N, N, N)`` with axes ``[batch, component, ix, iy, iz]`` on the
periodic ``[0, 2*pi)^3`` box, matching the transpose ``TurbDataset`` applies
when it loads ALIAKMON's own HDF5 output. It is resolution-specific: its
spectral coefficients are indexed by a fixed wavenumber grid, so the run's
``n`` must equal the checkpoint's.
"""

from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np

from . import kernels as K
from .backend import COMM, IS_ROOT, SIZE, on_root
from .data import State
from .parameters import PI, LESModel


# ----------------------------------------------------------------------
# Filter geometry
# ----------------------------------------------------------------------
# pDDLES measures wavenumbers in cycles/sample (torch.fft.fftfreq), where the
# grid corner sits at sqrt(3)/2. Its LES filter keeps |k|/n <= alpha*sqrt(3)/2,
# so in the integer-wavenumber units of this solver (box length 2*pi) the
# cutoff is alpha * sqrt(3)/2 * n. See TurbDataset._les_filter_mask.
_GRID_KMAX_FACTOR = math.sqrt(3.0) / 2.0


def cutoff_wavenumber(state: State) -> float:
    """LES filter cutoff ``k_c``, in integer wavenumbers.

    Precedence: an explicit ``les_kc``, else the value derived from the
    checkpoint's ``alpha`` by :func:`_sync_cutoff`, else the truncation
    ``kmax``. The middle case is the one that matters for pDDLES — tau has to
    be built with the same filter the network was trained against.
    """
    kc = float(state.cfg.les_kc)
    if kc > 0.0:
        return kc
    derived = getattr(state, "_sgs_kc", None)
    if derived is not None:
        return float(derived)
    return float(state.kmax)


def _sync_cutoff(state: State) -> None:
    """Derive ``k_c`` from the checkpoint's ``alpha`` and share it, once.

    Only the root rank holds the network, so the derived cutoff is broadcast:
    every rank must build the same filter mask. Must run before
    :func:`_filter_mask` caches anything.
    """
    if state.cfg.les_kc > 0.0 or getattr(state, "_sgs_kc", None) is not None:
        return
    kc = None
    if IS_ROOT:
        alpha = float(getattr(_load_predictor(state).args, "alpha", 0.0))
        kc = (alpha * _GRID_KMAX_FACTOR * state.n if alpha > 0.0
              else float(state.kmax))
    if SIZE > 1:
        kc = COMM.bcast(kc, root=0)
    state._sgs_kc = kc

    # The network was trained on inputs that were identically zero above k_c.
    # If the solver's resolved band reaches past it, every step hands the net
    # energy at wavenumbers it only ever saw empty, and tau there is not
    # meaningfully predicted. Set [numerics] truncation_kc to match.
    if state.k_max_active > kc * (1.0 + 1.0e-9):
        on_root(
            f"pDDLES WARNING: the solver resolves up to |k| = "
            f"{state.k_max_active:.3f} but the network was trained with the "
            f"LES filter at k_c = {kc:.3f}. Its input is out of distribution "
            f"above k_c. Set [numerics] truncation_kc = {kc:.4f} to evolve "
            f"the band the model was trained for.")


def filter_width(state: State) -> float:
    """Effective width ``Delta = pi / k_c`` of the sharp spectral filter."""
    kc = cutoff_wavenumber(state)
    return PI / kc if kc > 0.0 else 0.0


def _filter_mask(state: State) -> np.ndarray:
    """Sharp spectral filter ``G(k) = 1`` for ``|k| <= k_c``, cached on state.

    When ``les_kc`` is unset this is exactly the solver's truncation mask, so
    the same array is reused rather than rebuilt.
    """
    g = getattr(state, "_sgs_filter", None)
    if g is None:
        # A specific cutoff -- configured, or derived from the checkpoint's
        # alpha by _sync_cutoff -- means a sharp sphere at that k_c. Keying
        # this off cfg.les_kc alone silently ignored the derived cutoff and
        # filtered at the truncation instead. Without any k_c, "filter at the
        # truncation" means the solver's own mask, whose polyhedral boundary
        # is not a sphere and so cannot be expressed as a |k| <= k_c test.
        has_kc = (state.cfg.les_kc > 0.0
                  or getattr(state, "_sgs_kc", None) is not None)
        if has_kc:
            g = (np.sqrt(state.K2) <= cutoff_wavenumber(state)) & state.mask
        else:
            g = state.mask
        g = np.broadcast_to(g, state.T.cmplx_shape).copy()
        state._sgs_filter = g
    return g


def _ik_axes(state: State):
    """``(i*kx, i*ky, i*kz)`` broadcastable over a Fourier field, cached."""
    ik = getattr(state, "_sgs_ik", None)
    if ik is None:
        ik = (1j * state.KX, 1j * state.KY, 1j * state.KZ)
        state._sgs_ik = ik
    return ik


# ----------------------------------------------------------------------
# Trained predictor (pDDLES)
# ----------------------------------------------------------------------
# Sentinel for the stand-in scaler: normalise using the field's own moments
# instead of constants fitted over a training set. See _standardise below.
_AUTO_SCALER = object()


class _Predictor:
    """A trained pDDLES network plus the scaler it was trained with.

    Holds the pieces of the reference inference path (``trainer.train_one_epoch``
    in the pDDLES tree), which is *not* just a forward pass::

        X_scaled = (X - X_mean) / X_std
        pred     = model(X_scaled)
        u_star   = y_std * pred + y_mean

    Skipping the scaler feeds the network inputs several standard deviations
    from anything it saw in training, so it must be loaded, not defaulted --
    or, failing that, stood in for (``les_pddles_scaler = "auto"``).
    """

    def __init__(self, model, scaler, args, device):
        self.model = model
        self.scaler = scaler   # (x_mean, x_std, y_mean, y_std), None, or _AUTO
        self.args = args
        self.device = device

    def __call__(self, u_global: np.ndarray) -> np.ndarray:
        import torch

        x_np = np.ascontiguousarray(u_global, dtype=np.float32)
        auto = self.scaler is _AUTO_SCALER
        if auto:
            x_np, bias, scale = _standardise(x_np)

        x = torch.from_numpy(x_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.scaler is not None and not auto:
                x_mean, x_std, y_mean, y_std = self.scaler
                x = (x - x_mean) / x_std
            y = self.model(x)
            if self.scaler is not None and not auto:
                y = y_std * y + y_mean
        y = y.detach().to("cpu").numpy()

        if auto:
            # Undo with the same constants, so an identity network in scaled
            # space returns the input unchanged -- which keeps the model's
            # consistency limit (no predicted subgrid scales => no transfer).
            y = y * scale[None] + bias[None]
        return y


def _standardise(u: np.ndarray):
    """Scale each velocity component to zero mean and unit variance.

    ``u`` is ``(3, N, N, N)``. Returns ``(scaled, mean, std)`` with the moments
    shaped ``(3, 1, 1, 1)`` so they broadcast back over the field. Per
    component, matching how ``NormScaler`` reduces over ``(batch, x, y, z)``
    and keeps the channel axis.

    A component that is uniform has zero variance and cannot be given unit
    variance; its scale is left at 1 so the field passes through unchanged
    rather than producing NaNs.
    """
    mean = u.mean(axis=(1, 2, 3), keepdims=True)
    std = u.std(axis=(1, 2, 3), keepdims=True)
    std = np.where(std > 0.0, std, 1.0).astype(u.dtype)
    mean = mean.astype(u.dtype)
    return (u - mean) / std, mean, std


def _resolve_scaler(torch, ckpt_args, state, device):
    """Load the normalisation constants the checkpoint was trained with.

    ``args.scaler`` names the scheme; the fitted constants live beside the
    training data (``args.h5path``) in ``norm.pt`` / ``minmax.pt``, not in the
    checkpoint. ``les_pddles_scaler`` overrides that path for a checkpoint
    whose training data has since moved.
    """
    kind = getattr(ckpt_args, "scaler", "norm")
    if kind == "none":
        return None

    # Stand-in for a norm.pt that is not available: standardise with the
    # field's own per-component moments at every call. Loud, because it is not
    # what the network was trained against -- the fitted constants are
    # training-set statistics, while these follow the running solution.
    if state.cfg.les_pddles_scaler == "auto":
        on_root("pDDLES: [les] les_pddles_scaler = 'auto' — standardising "
                "with the field's own moments instead of the fitted "
                f"{kind!r} constants. Diagnostic use only; load the real "
                "norm.pt for production runs.")
        return _AUTO_SCALER

    fname = {"norm": "norm.pt", "minmax": "minmax.pt"}.get(kind)
    if fname is None:
        raise ValueError(f"pDDLES: unsupported scaler {kind!r} in checkpoint")

    path = state.cfg.les_pddles_scaler
    if not path:
        h5path = getattr(ckpt_args, "h5path", "") or ""
        path = os.path.join(str(h5path), fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"pDDLES: the checkpoint was trained with the {kind!r} scaler but "
            f"its constants were not found at {path!r}. That file is written "
            f"next to the training data, not into the checkpoint — copy it "
            f"across and point [les] les_pddles_scaler at it. For a run "
            f"without it, set les_pddles_scaler = \"auto\" to standardise "
            f"with the field's own moments (diagnostic only).")

    blob = torch.load(path, map_location=device, weights_only=False)
    vals = blob["vals"]
    if kind == "norm":
        # stack of X_mean, X_std, y_mean, y_std, each shaped (1, 3, 1, 1, 1).
        return tuple(vals[i].to(device) for i in range(4))
    # minmax stores four scalars; recast to the same (bias, range) algebra.
    x_min, x_max, y_min, y_max = (vals[i].to(device) for i in range(4))
    return (x_min, x_max - x_min, y_min, y_max - y_min)


def _backfill_args(args, torch) -> None:
    """Supply derived fields that older checkpoints' Namespaces predate.

    ``pDDLES.py:main`` computes several attributes *after* ``parse_args`` and
    before building the model — ``dimensions``, the conv/batchnorm classes,
    the dtypes. Only what existed when a checkpoint was written got pickled
    into it, so a network saved before an attribute was introduced cannot be
    rebuilt by today's ``lib/arch`` without it (the FNet32/WNet32 checkpoints
    predate ``precision``/``torch_dtype``, for instance).

    Fills in only what is *missing*, reproducing main()'s own defaults, so a
    checkpoint that carries a value always keeps it.
    """
    import torch.nn as nn

    if not hasattr(args, "dimensions"):
        args.dimensions = 3
    if not hasattr(args, "hdf5_key"):
        args.hdf5_key = "scl" if getattr(args, "scalar", False) else "u"
    two_d = args.dimensions == 2
    if not hasattr(args, "conv"):
        args.conv = nn.Conv2d if two_d else nn.Conv3d
    if not hasattr(args, "batchnorm"):
        args.batchnorm = nn.BatchNorm2d if two_d else nn.BatchNorm3d
    if not hasattr(args, "precision"):
        args.precision = "single"
    single = args.precision == "single"
    if not hasattr(args, "numpy_dtype"):
        args.numpy_dtype = np.float32 if single else np.float64
    if not hasattr(args, "torch_dtype"):
        args.torch_dtype = torch.float32 if single else torch.float64


def _load_predictor(state: State) -> _Predictor:
    """Build and cache the pDDLES predictor. Root rank only.

    Deferred until first use so that neither ``torch`` nor the pDDLES source
    tree is needed by DNS or eddy-viscosity runs.

    The checkpoint is self-describing: ``torch.save`` stored the training
    ``argparse.Namespace`` alongside the weights, so the architecture, its
    hyper-parameters and the filter cutoff ``alpha`` all come from the file.
    Only the device and the data-loading flags are overridden.
    """
    predictor = getattr(state, "_sgs_predictor", None)
    if predictor is not None:
        return predictor

    # Configuration is validated before torch is imported, so the common
    # mistakes report themselves rather than surfacing as an import error.
    cfg = state.cfg
    path = cfg.les_pddles_model
    if not path:
        raise ValueError(
            "pDDLES requires a trained predictor: set [les] les_pddles_model "
            "to a pDDLES .pth checkpoint (see aliakmon_py.sgs)")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"pDDLES predictor not found: {path!r} ([les] les_pddles_model)")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pDDLES needs PyTorch; install it (pip install torch) or choose "
            "another les_model") from exc

    # The architecture lives in the pDDLES tree (lib/arch/<arch>.py), which is
    # not a package ALIAKMON depends on, so it is imported by path.
    src = cfg.les_pddles_source
    if src and src not in sys.path:
        sys.path.insert(0, src)

    device = cfg.les_pddles_device
    # weights_only=False: the payload carries an argparse.Namespace, matching
    # how the pDDLES trainer reloads its own checkpoints.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise ValueError(
            f"pDDLES: {path!r} is not a pDDLES checkpoint (expected 'args' and "
            f"'model' keys, found {sorted(ckpt)[:6]})")
    args = ckpt["args"]

    # Run on this machine, from memory, with no data loading or staging.
    args.device = device
    args.dev = "cpu" if str(device).startswith("cpu") else "gpu"
    args.noload = True
    args.copy = False
    _backfill_args(args, torch)

    if int(getattr(args, "n", state.n)) != state.n:
        raise ValueError(
            f"pDDLES: checkpoint was trained at n={args.n} but this run has "
            f"n={state.n}. The network is resolution-specific (its spectral "
            f"coefficients are indexed by a fixed wavenumber grid).")

    try:
        arch = importlib.import_module(f"lib.arch.{args.arch}")
    except ImportError as exc:
        raise ImportError(
            f"pDDLES: cannot import lib.arch.{args.arch}; point [les] "
            f"les_pddles_source at a checkout of the pDDLES source tree "
            f"(currently {src!r})") from exc

    model = arch.get_model(args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    predictor = _Predictor(model, _resolve_scaler(torch, args, state, device),
                           args, device)
    state._sgs_predictor = predictor
    return predictor


def _predict(state: State, u_global: np.ndarray) -> np.ndarray:
    """Run the trained predictor on the global field. Root rank only.

    ``u_global`` is ``(3, N, N, N)`` float64; the return is the same shape.
    The round trip through the network's float32 arithmetic caps the subgrid
    term at ~1e-7 relative precision, which is far below its modelling error.
    """
    predictor = _load_predictor(state)
    y = predictor(u_global)

    n = state.n
    expected = (1, 3, n, n, n)
    if y.shape != expected:
        raise ValueError(
            f"pDDLES predictor returned shape {tuple(y.shape)}, expected "
            f"{expected} — see the checkpoint contract in aliakmon_py.sgs")
    return y[0].astype(np.float64)


# ----------------------------------------------------------------------
# Gather / scatter around the (non-decomposed) predictor
# ----------------------------------------------------------------------
def _gather_velocity(state: State, u_local):
    """Assemble the local subdomains into a global ``(3, N, N, N)`` on root.

    Returns ``(global_field, slices)`` on root and ``(None, None)`` elsewhere;
    ``slices`` is in rank order so the prediction can be scattered straight
    back with :func:`_scatter_prediction`.
    """
    block = np.ascontiguousarray(np.stack(u_local))
    if SIZE == 1:
        return block, [state.T.real_slice]

    parcels = COMM.gather((state.T.real_slice, block), root=0)
    if not IS_ROOT:
        return None, None
    n = state.n
    full = np.empty((3, n, n, n), dtype=np.float64)
    slices = []
    for sl, chunk in parcels:
        full[(slice(None),) + tuple(sl)] = chunk
        slices.append(sl)
    return full, slices


def _scatter_prediction(state: State, u_global, slices, out) -> None:
    """Send each rank its own subdomain of the predicted field into ``out``."""
    if SIZE == 1:
        for c in range(3):
            out[c][...] = u_global[c]
        return

    chunks = None
    if IS_ROOT:
        chunks = [np.ascontiguousarray(u_global[(slice(None),) + tuple(sl)])
                  for sl in slices]
    mine = COMM.scatter(chunks, root=0)
    for c in range(3):
        out[c][...] = mine[c]


# ----------------------------------------------------------------------
# The stress itself
# ----------------------------------------------------------------------
def _velocity_gradients(state: State, fin) -> None:
    """Fill ``state.sgs_grad[i][j]`` with dU_i/dx_j (nine backward transforms)."""
    T = state.T
    tmp = state.sgs_cmplx
    ik = _ik_axes(state)
    for i in range(3):
        for j in range(3):
            np.multiply(fin[i], ik[j], out=tmp)
            T.backward(tmp, state.sgs_grad[i][j])


def subgrid_stress(state: State, fin) -> None:
    """Build ``tau_ij`` for the field ``fin`` into ``state.sgs_tau``.

    Physical-space result, six components in ``(11, 12, 13, 22, 23, 33)``
    order. Costs one
    predictor evaluation (with its gather/scatter) plus twelve transforms, and
    nine more when the backscatter clip is on.
    """
    if state.cfg.les_model != LESModel.PDDLES:  # pragma: no cover - enum-guarded
        raise ValueError(f"not a structural LES model: {state.cfg.les_model!r}")

    # Fix the filter cutoff from the checkpoint before any mask is built.
    _sync_cutoff(state)

    T = state.T
    ubar, ustar, tau = state.sgs_ubar, state.sgs_ustar, state.sgs_tau
    g, tmp = _filter_mask(state), state.sgs_cmplx

    # Resolved velocity in physical space, then the predicted full field.
    for c in range(3):
        T.backward(fin[c], ubar[c])
    u_global, slices = _gather_velocity(state, ubar)
    predicted = _predict(state, u_global) if IS_ROOT else None
    _scatter_prediction(state, predicted, slices, ustar)

    # tau_ij = G*(u*_i u*_j) - G*(u*_i) G*(u*_j), both terms built from the
    # prediction. This is TurbDataset.subgrid_scale_tensor in the pDDLES
    # reference. Using the solver's own U for the second term would look
    # equivalent -- the net is trained so G*u* ~ U -- but it is not: the
    # difference is exactly the reconstruction error, and it would leak into
    # tau as a spurious stress rather than cancelling.
    K.outer_product(ustar[0], ustar[1], ustar[2],
                    tau[0], tau[1], tau[2], tau[3], tau[4], tau[5])
    for m in range(6):
        T.forward(tau[m], tmp)
        tmp *= g
        T.backward(tmp, tau[m])

    # Filter the prediction in place, then subtract the outer product of it.
    for c in range(3):
        T.forward(ustar[c], tmp)
        tmp *= g
        T.backward(tmp, ustar[c])
    K.subtract_outer_product(ustar[0], ustar[1], ustar[2],
                             tau[0], tau[1], tau[2], tau[3], tau[4], tau[5])
    K.make_deviatoric(tau[0], tau[3], tau[5])

    if state.cfg.les_clip_backscatter:
        _velocity_gradients(state, fin)
        a = state.sgs_grad
        K.clip_backscatter(a[0][0], a[0][1], a[0][2],
                           a[1][0], a[1][1], a[1][2],
                           a[2][0], a[2][1], a[2][2],
                           tau[0], tau[1], tau[2], tau[3], tau[4], tau[5])


def subgrid_force(state: State, fin=None) -> None:
    """Refresh ``state.fsgs`` with ``f_i = -d(tau_ij)/dx_j`` for ``fin``.

    Called once per timestep by :func:`numerics.update_subgrid_model`; the
    result is then held fixed through the RK substages of that step.

    The stress is truncated before being differentiated, so the force carries
    no content the solver would discard anyway. Its isotropic part needs no
    special treatment: ``d(tau_kk/3)/dx_i`` is a pure gradient and the pressure
    projection in :func:`numerics.compute_rhs` removes it for free.
    """
    fin = state.fu if fin is None else fin
    subgrid_stress(state, fin)

    T, tau_hat = state.T, state.sgs_tau_hat
    for m in range(6):
        T.forward(state.sgs_tau[m], tau_hat[m])
        tau_hat[m] *= state.mask
    K.tensor_divergence(tau_hat[0], tau_hat[1], tau_hat[2],
                        tau_hat[3], tau_hat[4], tau_hat[5],
                        state.kx, state.ky, state.kz,
                        state.fsgs[0], state.fsgs[1], state.fsgs[2])


def sgs_info(state: State) -> dict:
    """Static description of the active structural model, for reporting."""
    return dict(kind="tensor", model=state.cfg.les_model.name,
                kc=cutoff_wavenumber(state), delta=filter_width(state),
                clipped=state.cfg.les_clip_backscatter)
