"""Blackbody thermal emission: a planet that shines by its own temperature."""

from math import pi, sqrt

import jax.numpy as jnp
from hwoutils.constants import Rearth2m, c, h, k_B, nm2m, nm2um, pc2m, um2m
from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from .base import AbstractPhysicalModel


class BlackbodyPhysicalModel(AbstractPhysicalModel):
    """Planet emitting as a uniform blackbody disk, independent of phase.

    The planet's photon flux density at the observer is
    ``pi B(T) (Rp / d)^2`` with ``B`` the Planck photon radiance (Rybicki &
    Lightman 1979, Radiative Processes in Astrophysics); the contrast is
    that flux over the host star's own ``spec_flux_density``, so any star
    model serves as the denominator.

    Attributes:
        teff_K: Planet effective temperature [K], shape ``(K,)``.
    """

    teff_K: Array

    def spec_flux_density(
        self, wavelength_nm: Array, Rp_Rearth: Array, dist_pc: Array
    ) -> Array:
        """Planet photon flux density at the observer [ph/s/m^2/nm].

        Args:
            wavelength_nm: Wavelength [nm], scalar.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            dist_pc: Distance to the system [pc].

        Returns:
            Flux density, shape ``(K,)``.
        """
        wavelength_um = jnp.asarray(wavelength_nm) * nm2um
        exponent = (h * c / (k_B * um2m)) / (wavelength_um * self.teff_K)
        # Fold radiance, dilution and per-nm conversion into one moderate scale.
        # Separate huge/tiny factors lose float32 second derivatives under JIT.
        amplitude = sqrt(2.0 * pi * c * nm2m) * Rearth2m / (um2m**2 * pc2m)
        scale = (amplitude * Rp_Rearth / dist_pc) ** 2
        return scale / (wavelength_um**4 * jnp.expm1(exponent))

    def contrast(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Require the host star: a thermal contrast depends on its flux."""
        raise ValueError(
            "Use contrast_at_time(..., time_jd=..., star=...) for thermal emission"
        )

    def contrast_at_time(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
        *,
        time_jd: Array,
        star=None,
    ) -> Array:
        """Planet flux over the star's flux at one wavelength.

        Args:
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``; sets the
                output shape only.
            dist_AU: Star-planet distance [AU]; ignored.
            wavelength_nm: Wavelength [nm], scalar.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_jd: Observation Julian Dates, shape ``(T,)``.
            star: The host ``AbstractStar``.

        Returns:
            Contrast, shape ``(K, T)``.
        """
        # scene.Planet imports physical_model, so defer this circular dependency.
        from ..scene.star import BlackbodyStar

        if star is None:
            raise ValueError("BlackbodyPhysicalModel needs the host star")
        if isinstance(star, BlackbodyStar):
            wavelength_um = jnp.asarray(wavelength_nm) * nm2um
            c2 = h * c / (k_B * um2m)
            x_star = c2 / (wavelength_um * star.teff_K)
            x_planet = c2 / (wavelength_um * self.teff_K)
            area = (Rp_Rearth * Rearth2m / star.radius_m) ** 2
            ratio = (area * jnp.expm1(x_star) / jnp.expm1(x_planet))[:, None]
        else:
            planet = self.spec_flux_density(wavelength_nm, Rp_Rearth, star.dist_pc)
            ratio = planet[:, None] / star.spec_flux_density(wavelength_nm, time_jd)
        return ratio * jnp.ones_like(phase_angle_rad)

    def __repr__(self) -> str:
        """One-line summary of the effective temperature."""
        return f"BlackbodyPhysicalModel(Teff={fmt_scalar_or_array(self.teff_K)} K)"
