"""Gray Lambertian physical model -- analytic phase, no wavelength dependence."""

import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from .base import AbstractPhysicalModel


def lambert_phase(phase_angle_rad: Array) -> Array:
    """Lambert phase on [0, pi], clipped at zero to suppress endpoint roundoff.

    Phi(b) = (sin(b) + (pi - b) cos(b)) / pi. Clipping keeps reflected
    light nonnegative at new phase; it leaves interior values unchanged.
    """
    sin_b = jnp.sin(phase_angle_rad)
    cos_b = jnp.cos(phase_angle_rad)
    return jnp.maximum((sin_b + (jnp.pi - phase_angle_rad) * cos_b) / jnp.pi, 0.0)


def reflected_contrast(phase_angle_rad, dist_AU, Ag, Rp_Rearth):
    """Return Ag Phi(beta) (Rp/r)^2 for broadcast-compatible input arrays.

    Geometry comes from orbix. Radius is in Earth radii, distance in AU,
    and Ag is geometric albedo (possibly wavelength dependent).
    """
    return Ag * lambert_phase(phase_angle_rad) * (Rp_Rearth * Rearth2AU / dist_AU) ** 2


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
        gray case. ``Rp_Rearth`` is shape ``(K,)``, supplied by the
        host ``Planet``.
        """
        return reflected_contrast(
            phase_angle_rad, dist_AU, self.Ag[:, None], Rp_Rearth[:, None]
        )

    def __repr__(self) -> str:
        """One-line summary of geometric albedo."""
        ag = fmt_scalar_or_array(self.Ag)
        return f"LambertianPhysicalModel(Ag={ag})"
