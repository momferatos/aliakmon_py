"""Run-time conservation checks (validation.f90, hydro subset).

Provides the energy budget test the Fortran ``dissipation_test`` performs:

* **Viscous runs** — the total kinetic energy ``E`` is sampled every step and
  its central-difference rate ``dE/dt`` is compared with the instantaneous
  dissipation ``eps``. For decaying turbulence ``dE/dt = -eps``, so the
  residual ``err = dE/dt + eps`` should hover near zero (the relative error is
  reported against ``dE/dt``).
* **Inviscid runs** — energy is conserved, so the residual is the drift of the
  total energy from its initial value; the mean kinetic helicity drift is
  reported alongside it.

The MHD, resistive and ambipolar branches of the original are omitted.
"""

from __future__ import annotations

from .backend import on_root
from .data import State
from . import numerics as N


class DissipationTest:
    """Stateful energy-budget validator, evaluated once per timestep.

    Holds the rolling history needed for the central difference. Call
    :meth:`update` after each completed step with the step size ``dt``; it
    prints the residual on the root rank and returns it (or ``None`` until
    enough samples have accumulated).
    """

    def __init__(self, state: State):
        self.state = state
        self.viscous = bool(state.cfg.viscous)
        # Rolling samples of (kinetic energy, dissipation rate).
        self._energy: list[float] = []
        self._eps: list[float] = []
        self.te_initial = N.kinetic_energy(state)
        self.mkh_initial = N.mean_kinetic_helicity(state)

    def update(self, dt: float):
        """Record the current state and report the conservation residual."""
        ke = N.kinetic_energy(self.state)
        eps = N.mean_dissipation(self.state) if self.viscous else 0.0
        self._energy.append(ke)
        self._eps.append(eps)

        if self.viscous:
            return self._viscous_residual(dt)
        return self._inviscid_residual(ke)

    def _viscous_residual(self, dt: float):
        # Need three samples for a centred dE/dt about the middle one.
        if len(self._energy) < 3:
            return None
        e_prev, _, e_next = self._energy[-3:]
        eps_mid = self._eps[-2]
        dedt = (e_next - e_prev) / (2.0 * dt)
        err = dedt + eps_mid
        rel = (err / dedt * 100.0) if dedt != 0.0 else 0.0
        on_root(f"E test: {err:11.3e} {rel:9.3f}%")
        return err

    def _inviscid_residual(self, ke: float):
        err = ke - self.te_initial
        rel = (err / self.te_initial * 100.0) if self.te_initial != 0.0 else 0.0
        mkh = N.mean_kinetic_helicity(self.state)
        mkh_err = mkh - self.mkh_initial
        on_root(f"TE: {err:11.3e} {rel:9.3f}%   "
                f"MKH: {mkh_err:11.3e}")
        return err
