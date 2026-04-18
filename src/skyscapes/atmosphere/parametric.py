"""Parametric atmosphere -- stub for PICASO/ExoJAX radiative-transfer adapters.

A concrete subclass (with its RT engine stored as ``eqx.field(static=True)``)
will land in a later plan. Until then, ``reflected_spectrum`` raises so the
class is usable for type dispatch but not for actual contrast evaluation.
"""

from __future__ import annotations

from jaxtyping import Array

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
