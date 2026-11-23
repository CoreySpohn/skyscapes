"""Blackbody thermal emission: a planet that shines by its own temperature."""

import jax.numpy as jnp
from hwoutils.constants import Rearth2m, c, h, k_B, nm2um, pc2m, um2m
from hwoutils.planck import planck_photon_radiance
from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from .base import AbstractPhysicalModel

# Second radiation constant in micrometer kelvin, as in hwoutils.planck.
_EXPONENT_UM_K = h * c / (k_B * um2m)


def _log_occupation(wavelength_nm, temperature_k):
    """Log of the Planck occupation factor exp(-x) / (1 - exp(-x)).

    ``x = hc / (lambda k T)``. Two Planck radiances at one wavelength differ
    only by this factor, so their ratio is the exponential of a difference of
    these logs, which stays representable in float32 at any temperature.
    """
    exponent = _EXPONENT_UM_K / (
        jnp.asarray(wavelength_nm) * nm2um * jnp.asarray(temperature_k)
    )
    return -exponent - jnp.log(-jnp.expm1(-exponent))


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
        star_planet_distance_AU: Array | None = None,
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
        flux = (
            planck_photon_radiance(wavelength_nm, self.teff_K)
            * jnp.pi
            * (Rp_Rearth * Rearth2m / (dist_pc * pc2m)) ** 2
        )
        if phase_angle_rad is None:
            return flux
        return flux[:, None] * jnp.ones_like(phase_angle_rad)

    def node_photon_flux_density(
        self,
        quadrature,
        resolution,
        time_d: Array,
        *,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        Rp_Rearth: Array,
        star,
        star_nodes: Array,
        dist_pc: Array,
    ) -> Array:
        """Thermal flux at the nodes [ph s^-1 m^-2 nm^-1]; the star does not enter.

        A Planck spectrum is smooth, so its samples at the nodes integrate it
        against a response with the rule's own order.

        Args:
            quadrature: Band rule whose nodes are read.
            resolution: Unused; a smooth spectrum needs no operator.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``; sets the shape.
            star_planet_distance_AU: Star-planet distance [AU]; ignored.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            star: The host star; ignored.
            star_nodes: The host star at the nodes; ignored.
            dist_pc: Observer distance supplied by the system [pc].

        Returns:
            Photon flux density, shape ``(W, Q, K, T)``.
        """
        return self._flux_at(
            quadrature.nodes_nm,
            time_d,
            phase_angle_rad=phase_angle_rad,
            star_planet_distance_AU=star_planet_distance_AU,
            Rp_Rearth=Rp_Rearth,
            star=star,
            dist_pc=dist_pc,
        )

    def contrast(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Require the host star: a thermal contrast depends on its flux.

        Args:
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``; unused.
            star_planet_distance_AU: Star-planet distance [AU]; unused.
            wavelength_nm: Wavelength [nm]; unused.
            Rp_Rearth: Planet radius [Earth radii]; unused.

        Returns:
            Nothing; this method always raises ``ValueError``.
        """
        raise ValueError(
            "Use contrast_at_time(..., time_d=..., star=...) for thermal emission"
        )

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
        """Planet flux over the star's flux at one wavelength.

        Args:
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``; sets the
                output shape only.
            star_planet_distance_AU: Star-planet distance [AU]; ignored.
            wavelength_nm: Wavelength [nm], scalar.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            star: The host ``AbstractStar``.
            dist_pc: Observer distance supplied by the system [pc].

        Returns:
            Contrast, shape ``(K, T)``.
        """
        # scene.Planet imports physical_model, so defer this circular dependency.
        from ..scene.star import BlackbodyStar

        if star is None:
            raise ValueError("BlackbodyPhysicalModel needs the host star")
        if isinstance(star, BlackbodyStar):
            # Cancel the shared dilution and Planck prefactor before division,
            # so the ratio and its adjoints stay representable in float32.
            area = (Rp_Rearth * Rearth2m / star.radius_m) ** 2
            log_ratio = _log_occupation(wavelength_nm, self.teff_K) - _log_occupation(
                wavelength_nm, star.teff_K
            )
            # The denominator is the star's scaled flux, so its scale divides.
            ratio = (area * jnp.exp(log_ratio) / star.flux_scale)[:, None]
            return ratio * jnp.ones_like(phase_angle_rad)
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

    def __repr__(self) -> str:
        """One-line summary of the effective temperature.

        Returns:
            One-line summary string.
        """
        return f"BlackbodyPhysicalModel(Teff={fmt_scalar_or_array(self.teff_K)} K)"
