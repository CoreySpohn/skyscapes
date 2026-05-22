"""Parametric atmosphere -- stub for PICASO/ExoJAX radiative-transfer adapters.

A concrete subclass (with its RT engine stored as ``eqx.field(static=True)``)
will land in a later plan. Until then, ``reflected_spectrum`` raises so the
class is usable for type dispatch but not for actual contrast evaluation.
"""

from __future__ import annotations

from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from .base import AbstractAtmosphere


class ParametricAtmosphere(AbstractAtmosphere):
    """Stub for future radiative-transfer atmospheres.

    Attributes:
        Rp_Rearth: Planetary radii, shape ``(K,)``.
    """

    Rp_Rearth: Array

    def reflected_spectrum(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
    ) -> Array:
        """Not implemented -- waiting on RT adapter (PICASO/ExoJAX)."""
        raise NotImplementedError(
            "ParametricAtmosphere is a stub. A concrete RT-backed subclass "
            "will be introduced in a later plan."
        )

    def __repr__(self) -> str:
        """One-line summary marking this as a placeholder atmosphere."""
        rp = fmt_scalar_or_array(self.Rp_Rearth)
        return f"ParametricAtmosphere(Rp={rp} R_earth, stub)"
