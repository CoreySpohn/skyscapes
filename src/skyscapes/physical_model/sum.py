"""Additive radiative components sharing one planet's geometry and radius."""

from jaxtyping import Array

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

    def __check_init__(self):
        """Reject an empty sum before tracing or evaluation."""
        if not self.components:
            raise ValueError("SumPhysicalModel needs at least one component")

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        *,
        phase_angle_rad: Array,
        dist_AU: Array,
        Rp_Rearth: Array,
        time_jd: Array,
        star,
    ) -> Array:
        """Summed observer photon flux [ph/s/m^2/nm], shape ``(K, T)``."""
        return sum(
            component.spec_flux_density(
                wavelength_nm,
                phase_angle_rad=phase_angle_rad,
                dist_AU=dist_AU,
                Rp_Rearth=Rp_Rearth,
                time_jd=time_jd,
                star=star,
            )
            for component in self.components
        )

    def contrast(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Require the host star and epochs for the component fluxes."""
        raise ValueError("Use contrast_at_time(..., time_jd=..., star=...) for a sum")

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
        """Summed photon flux over stellar flux, shape ``(K, T)``."""
        if star is None:
            raise ValueError("SumPhysicalModel needs the host star")
        flux = self.spec_flux_density(
            wavelength_nm,
            phase_angle_rad=phase_angle_rad,
            dist_AU=dist_AU,
            Rp_Rearth=Rp_Rearth,
            time_jd=time_jd,
            star=star,
        )
        return flux / star.spec_flux_density(wavelength_nm, time_jd)
