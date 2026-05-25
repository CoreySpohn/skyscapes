"""Grey Lambertian physical model -- analytic phase, no wavelength dependence."""

import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from .base import AbstractPhysicalModel


def _lambert_phase(phase_angle_rad: Array) -> Array:
    """Classical Lambert phase: Phi(b) = (sin(b) + (pi - b) * cos(b)) / pi."""
    sin_b = jnp.sin(phase_angle_rad)
    cos_b = jnp.cos(phase_angle_rad)
    return (sin_b + (jnp.pi - phase_angle_rad) * cos_b) / jnp.pi


class LambertianPhysicalModel(AbstractPhysicalModel):
    """Lambertian reflector.

    Attributes:
        Ag: Geometric albedo, shape ``(K,)``.
    """

    Ag: Array

    def contrast(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Contrast = Ag * Phi(beta) * (Rp/r)**2. Wavelength-independent.

        ``phase_angle_rad`` and ``dist_AU`` are shape ``(K, T)``.
        ``wavelength_nm`` is part of the interface but ignored in the
        grey case. ``Rp_Rearth`` is shape ``(K,)``, supplied by the
        host ``Planet``.
        """
        Rp_AU = (Rp_Rearth * Rearth2AU)[:, None]
        Ag = self.Ag[:, None]
        phase = _lambert_phase(phase_angle_rad)
        return Ag * phase * (Rp_AU / dist_AU) ** 2

    def __repr__(self) -> str:
        """One-line summary of geometric albedo."""
        ag = fmt_scalar_or_array(self.Ag)
        return f"LambertianPhysicalModel(Ag={ag})"
