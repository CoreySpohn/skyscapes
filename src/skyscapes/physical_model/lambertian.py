"""Gray Lambertian physical model -- analytic phase, no wavelength dependence."""

import interpax
import jax
import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import (
    nm_to_wavenumber_cm,
    spherical_to_geometric_albedo,
)
from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from .base import AbstractPhysicalModel


def lambert_phase(phase_angle_rad: Array) -> Array:
    """Lambert phase on [0, pi], clipped at zero to suppress endpoint roundoff.

    Phi(b) = (sin(b) + (pi - b) cos(b)) / pi. Near new phase, a series in
    the distance from pi preserves the cubic crescent as the terms cancel.
    Clipping keeps reflected light nonnegative at new phase.

    Args:
        phase_angle_rad: Star-planet-observer phase angle [rad], in [0, pi].

    Returns:
        Dimensionless Lambert phase function, same shape as the input; one at
        zero phase angle.
    """
    sin_b = jnp.sin(phase_angle_rad)
    cos_b = jnp.cos(phase_angle_rad)
    direct = sin_b + (jnp.pi - phase_angle_rad) * cos_b
    # Trigonometric argument reduction preserves the distance from pi even
    # when pi itself is not representable at the input dtype.
    delta = jnp.arctan2(sin_b, -cos_b)
    d2 = delta * delta
    # sin(delta) - delta*cos(delta), through degree 13. At delta <= 1/4
    # the first omitted term is smaller than one float64 epsilon of the sum.
    series = (
        delta
        * d2
        * (
            1 / 3
            + d2
            * (
                -1 / 30
                + d2
                * (1 / 840 + d2 * (-1 / 45360 + d2 * (1 / 3991680 - d2 / 518918400)))
            )
        )
    )
    near_new = (phase_angle_rad >= jnp.pi - 0.25) & (phase_angle_rad <= jnp.pi)
    numerator = jnp.where(near_new, series, direct)
    return jnp.where(
        phase_angle_rad == jnp.pi, 0.0, jnp.maximum(numerator / jnp.pi, 0.0)
    )


def reflected_contrast(phase_angle_rad, star_planet_distance_AU, Ag, Rp_Rearth):
    """Return Ag Phi(beta) (Rp/r)^2 for broadcast-compatible input arrays.

    Geometry comes from orbix. Radius is in Earth radii, distance in AU,
    and Ag is geometric albedo (possibly wavelength dependent).

    Args:
        phase_angle_rad: Star-planet-observer phase angle [rad].
        star_planet_distance_AU: Star-planet distance [AU].
        Ag: Geometric albedo, dimensionless; may depend on wavelength.
        Rp_Rearth: Planet radius [Earth radii].

    Returns:
        Contrast (planet-to-star flux ratio), broadcast shape of the inputs.
    """
    return (
        Ag
        * lambert_phase(phase_angle_rad)
        * (Rp_Rearth * Rearth2AU / star_planet_distance_AU) ** 2
    )


def project_reflected(
    reflectivity: Array,
    nu_grid: Array,
    *,
    phase_angle_rad: Array,
    star_planet_distance_AU: Array,
    wavelength_nm: Array,
    radius_Rearth: Array,
) -> Array:
    """Reflected-light contrast cube from a plane-parallel reflectivity.

    Interpolates ``reflectivity`` onto the wavelengths (linear, with linear
    extrapolation, so values outside the grid can be negative), converts the
    spherical albedo to geometric albedo with the Lambertian-sphere factor
    2/3 (Seager 2010, eq 3.36), and applies Lambert phase and ``(Rp/d)^2``.

    Args:
        reflectivity: Plane-parallel reflectivity ``R(nu)``, shape ``(K, n_nu)``.
        nu_grid: Wavenumber grid [cm^-1], shape ``(n_nu,)``.
        phase_angle_rad: Star-planet-observer phase angle, shape ``(K, T)``.
        star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
        wavelength_nm: Wavelengths [nm], shape ``(W,)``.
        radius_Rearth: Planet radius [Earth radii], shape ``(K,)``.

    Returns:
        Contrast, shape ``(W, K, T)``.
    """
    target_nu = nm_to_wavenumber_cm(wavelength_nm)

    def interp_one_planet(spectrum):
        return interpax.interp1d(
            target_nu, nu_grid, spectrum, method="linear", extrap=True
        )

    albedo = spherical_to_geometric_albedo(
        jax.vmap(interp_one_planet)(reflectivity)
    )  # (K, W)
    return reflected_contrast(
        phase_angle_rad[None, :, :],
        star_planet_distance_AU[None, :, :],
        albedo.T[:, :, None],
        radius_Rearth[None, :, None],
    )


class LambertianPhysicalModel(AbstractPhysicalModel):
    """Lambertian reflector.

    Attributes:
        Ag: Geometric albedo, shape ``(K,)``.
    """

    Ag: Array

    def contrast(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Contrast = Ag * Phi(beta) * (Rp/r)**2. Wavelength-independent.

        ``phase_angle_rad`` and ``star_planet_distance_AU`` are shape ``(K, T)``.
        ``wavelength_nm`` is part of the interface but ignored in the
        gray case. ``Rp_Rearth`` is shape ``(K,)``, supplied by the
        host ``Planet``.

        Args:
            phase_angle_rad: Star-planet-observer phase angle [rad], shape ``(K, T)``.
            star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
            wavelength_nm: Wavelength [nm]; ignored.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.

        Returns:
            Contrast (planet-to-star flux ratio), shape ``(K, T)``.
        """
        return reflected_contrast(
            phase_angle_rad,
            star_planet_distance_AU,
            self.Ag[:, None],
            Rp_Rearth[:, None],
        )

    def __repr__(self) -> str:
        """One-line summary of geometric albedo.

        Returns:
            One-line summary string.
        """
        ag = fmt_scalar_or_array(self.Ag)
        return f"LambertianPhysicalModel(Ag={ag})"
