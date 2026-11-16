"""Blackbody thermal emission: a planet that shines by its own temperature."""

import jax.numpy as jnp
from hwoutils.constants import Rearth2m, c, h, k_B, nm2um, um2m
from jaxtyping import Array

from .._planck import _planck_photon_flux
from .._repr import fmt_scalar_or_array
from .base import AbstractPhysicalModel


class BlackbodyPhysicalModel(AbstractPhysicalModel):
    """Planet emitting as a uniform blackbody disk, independent of phase.

    The planet's photon flux density at the observer is
    ``pi B(T) (Rp / d)^2`` with ``B`` the Planck photon radiance (Rybicki &
    Lightman 1979, Radiative Processes in Astrophysics); the contrast is
    that flux over the host star's own ``photon_flux_density``, so any star
    model serves as the denominator.

    Attributes:
        teff_K: Planet effective temperature [K], shape ``(K,)``.
    """

    teff_K: Array

    def _photon_flux_density(
        self,
        wavelength_nm: Array,
        Rp_Rearth: Array,
        *,
        dist_pc: Array,
        phase_angle_rad: Array | None = None,
        dist_AU: Array | None = None,
        time_d: Array | None = None,
        star=None,
    ) -> Array:
        """Observer photon flux [ph/s/m^2/nm].

        ``time_d`` is days from the scene epoch.

        The radius is supplied by the planet and ``dist_pc`` by the system.
        Steady emission ignores phase, orbital distance, epoch and host flux.
        Returns shape ``(K,)`` directly, or ``(K, T)`` when
        ``phase_angle_rad`` supplies the time axis.
        """
        flux = _planck_photon_flux(
            wavelength_nm,
            self.teff_K,
            Rp_Rearth,
            dist_pc,
            radius_unit_m=Rearth2m,
            stable_wien=phase_angle_rad is not None,
        )
        if phase_angle_rad is None:
            return flux
        return flux[:, None] * jnp.ones_like(phase_angle_rad)

    def contrast(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Require the host star: a thermal contrast depends on its flux."""
        raise ValueError(
            "Use contrast_at_time(..., time_d=..., star=...) for thermal emission"
        )

    def contrast_at_time(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
        *,
        time_d: Array,
        star=None,
        dist_pc: Array | None = None,
    ) -> Array:
        """Planet flux over the star's flux at one wavelength.

        Args:
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``; sets the
                output shape only.
            dist_AU: Star-planet distance [AU]; ignored.
            wavelength_nm: Wavelength [nm], scalar.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            dist_pc: Observer distance supplied by the system [pc].
            star: The host ``AbstractStar``.

        Returns:
            Contrast, shape ``(K, T)``.
        """
        # scene.Planet imports physical_model, so defer this circular dependency.
        from ..scene.star import BlackbodyStar

        if star is None:
            raise ValueError("BlackbodyPhysicalModel needs the host star")
        if isinstance(star, BlackbodyStar):
            # Cancel the shared dilution before division to retain numerical range.
            wavelength_um = jnp.asarray(wavelength_nm) * nm2um
            c2 = h * c / (k_B * um2m)
            x_star = c2 / (wavelength_um * star.teff_K)
            x_planet = c2 / (wavelength_um * self.teff_K)
            area = (Rp_Rearth * Rearth2m / star.radius_m) ** 2
            ratio = (area * jnp.expm1(x_star) / jnp.expm1(x_planet))[:, None]
            return ratio * jnp.ones_like(phase_angle_rad)
        flux = self._photon_flux_density(
            wavelength_nm,
            phase_angle_rad=phase_angle_rad,
            dist_AU=dist_AU,
            Rp_Rearth=Rp_Rearth,
            time_d=time_d,
            star=star,
            dist_pc=dist_pc,
        )
        return flux / star.photon_flux_density(wavelength_nm, time_d, dist_pc=dist_pc)

    def __repr__(self) -> str:
        """One-line summary of the effective temperature."""
        return f"BlackbodyPhysicalModel(Teff={fmt_scalar_or_array(self.teff_K)} K)"
