"""Additive radiative components sharing one planet's geometry and radius."""

import numpy as np
from jaxtyping import Array
from optixstuff.spectral import spectral_breaks

from .base import AbstractPhysicalModel


class SumPhysicalModel(AbstractPhysicalModel):
    """Sum photon flux from a nonempty tuple of physical models.

    All components receive the same radius, orbit geometry, epochs and host
    star. Their parameters remain differentiable PyTree leaves. Contrast is
    the summed observer flux divided by the stellar flux.

    Attributes:
        components: Physical models describing the same ``K`` planets.
    """

    components: tuple[AbstractPhysicalModel, ...]

    def spectral_breaks_nm(self) -> np.ndarray:
        """Union of all component spectral breaks [nm].

        Returns:
            1-D array of the union of the component wavelength breaks [nm].
        """
        return spectral_breaks(*self.components)

    def __check_init__(self):
        """Reject an empty sum before tracing or evaluation."""
        if not self.components:
            raise ValueError("SumPhysicalModel needs at least one component")

    def _photon_flux_density(
        self,
        wavelength_nm: Array,
        *,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        Rp_Rearth: Array,
        time_d: Array,
        star,
        dist_pc: Array,
    ) -> Array:
        """Summed observer photon flux [ph/s/m^2/nm], shape ``(K, T)``.

        ``time_d`` is days from the scene epoch.
        """
        return sum(
            component._photon_flux_density(
                wavelength_nm,
                phase_angle_rad=phase_angle_rad,
                star_planet_distance_AU=star_planet_distance_AU,
                Rp_Rearth=Rp_Rearth,
                time_d=time_d,
                star=star,
                dist_pc=dist_pc,
            )
            for component in self.components
        )

    def contrast(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Require the host star and epochs for the component fluxes.

        Args:
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``; unused.
            star_planet_distance_AU: Star-planet distance [AU]; unused.
            wavelength_nm: Wavelength [nm]; unused.
            Rp_Rearth: Planet radius [Earth radii]; unused.

        Returns:
            Nothing; this method always raises ``ValueError``.
        """
        raise ValueError("Use contrast_at_time(..., time_d=..., star=...) for a sum")

    def contrast_at_time(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
        *,
        time_d: Array,
        star=None,
        dist_pc: Array | None = None,
    ) -> Array:
        """Summed photon flux over stellar flux, shape ``(K, T)``.

        ``time_d`` is days from the scene epoch.

        Args:
            phase_angle_rad: Star-planet-observer phase angle [rad], shape ``(K, T)``.
            star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
            wavelength_nm: Scalar wavelength [nm].
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            star: Host ``AbstractStar``; required.
            dist_pc: Distance of the system from the observer [pc].

        Returns:
            Contrast (planet-to-star flux ratio), shape ``(K, T)``.
        """
        if star is None:
            raise ValueError("SumPhysicalModel needs the host star")
        flux = self._photon_flux_density(
            wavelength_nm,
            phase_angle_rad=phase_angle_rad,
            star_planet_distance_AU=star_planet_distance_AU,
            Rp_Rearth=Rp_Rearth,
            time_d=time_d,
            star=star,
            dist_pc=dist_pc,
        )
        return flux / star.photon_flux_density(wavelength_nm, time_d, dist_pc=dist_pc)
