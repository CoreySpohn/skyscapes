"""Blackbody thermal emission: a planet that shines by its own temperature."""

import jax.numpy as jnp
from hwoutils.constants import Rearth2m, c, h, k_B, nm2m, pc2m
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
        wavelength_m = jnp.asarray(wavelength_nm) * nm2m
        exponent = h * c / (wavelength_m * k_B * self.teff_K)
        photon_radiance = 2.0 * c / wavelength_m**4 / jnp.expm1(exponent)
        dilution = (Rp_Rearth * Rearth2m / (dist_pc * pc2m)) ** 2
        return jnp.pi * photon_radiance * dilution * nm2m  # per m to per nm

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
        if star is None:
            raise ValueError("BlackbodyPhysicalModel needs the host star")
        planet = self.spec_flux_density(wavelength_nm, Rp_Rearth, star.dist_pc)
        ratio = planet[:, None] / star.spec_flux_density(wavelength_nm, time_jd)
        return ratio * jnp.ones_like(phase_angle_rad)

    def __repr__(self) -> str:
        """One-line summary of the effective temperature."""
        return f"BlackbodyPhysicalModel(Teff={fmt_scalar_or_array(self.teff_K)} K)"
