"""Grey Lambertian atmosphere -- analytic phase, no wavelength dependence."""

import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from jaxtyping import Array

from .base import AbstractAtmosphere


def _lambert_phase(phase_angle_rad: Array) -> Array:
    """Classical Lambert phase function, Phi(beta) = (sin(beta) + (pi - beta) * cos(beta)) / pi."""
    sin_b = jnp.sin(phase_angle_rad)
    cos_b = jnp.cos(phase_angle_rad)
    return (sin_b + (jnp.pi - phase_angle_rad) * cos_b) / jnp.pi


class LambertianAtmosphere(AbstractAtmosphere):
    """Lambertian reflector.

    Attributes:
        Rp_Rearth: Planetary radius, shape ``(K,)``.
        Ag: Geometric albedo, shape ``(K,)``.
    """

    Rp_Rearth: Array
    Ag: Array

    def reflected_spectrum(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
    ) -> Array:
        """Contrast = Ag * Phi(beta) * (Rp/r)**2.  Wavelength-independent.

        ``phase_angle_rad`` and ``dist_AU`` are shape ``(K, T)``.
        ``wavelength_nm`` is part of the interface but ignored in the
        grey case.
        """
        Rp_AU = (self.Rp_Rearth * Rearth2AU)[:, None]
        Ag = self.Ag[:, None]
        phase = _lambert_phase(phase_angle_rad)
        return Ag * phase * (Rp_AU / dist_AU) ** 2
