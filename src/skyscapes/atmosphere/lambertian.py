"""Grey Lambertian atmosphere — analytic phase, no wavelength dependence."""

import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from jaxtyping import Array

from .base import AbstractAtmosphere


def _lambert_phase(phase_angle_rad: Array) -> Array:
    """Classical Lambert phase function, Φ(β) = (sinβ + (π-β)cosβ) / π."""
    sin_b = jnp.sin(phase_angle_rad)
    cos_b = jnp.cos(phase_angle_rad)
    return (sin_b + (jnp.pi - phase_angle_rad) * cos_b) / jnp.pi


def _broadcast_over_leading(x: Array, target: Array) -> Array:
    """Reshape a (K,)-shaped x by appending singleton axes to match target's rank.

    ``target.ndim`` is static at trace time, so the reshape tuple is also static.
    """
    n_extra = target.ndim - x.ndim
    if n_extra <= 0:
        return x
    return x.reshape(x.shape + (1,) * n_extra)


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
        """Contrast = Ag · Φ(β) · (Rp/r)^2.  Wavelength-independent.

        ``phase_angle_rad`` and ``dist_AU`` may be shape ``(K,)`` or
        ``(K, T)``; the output matches their shape. ``wavelength_nm``
        is part of the interface but ignored in the grey case.
        """
        Rp_AU = self.Rp_Rearth * Rearth2AU
        phase = _lambert_phase(phase_angle_rad)
        Ag = _broadcast_over_leading(self.Ag, phase)
        Rp_b = _broadcast_over_leading(Rp_AU, phase)
        return Ag * phase * (Rp_b / dist_AU) ** 2
