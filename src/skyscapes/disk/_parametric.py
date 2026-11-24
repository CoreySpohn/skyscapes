"""Shared photon rendering of typed parametric scattering profiles."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import pc2AU
from hwoutils.conversions import arcsec_to_au
from optixstuff.spectral import Band, Knots
from zodi import AbstractDustAmplitude, SurfaceBrightnessDust

from .amplitude import calibration_band, check_amplitude, density_level
from .base import AbstractDisk


class AbstractParametricDisk(AbstractDisk):
    """Profile owning its amplitude; the scene owns the orientation.

    The inclination and position angle are not leaves of the disk: every radiance
    read takes them as keyword arguments, and a scene passes its midplane angles.

    The declared unit profile is extinction per AU. zero_point_jy optionally
    overrides the calibration passband's monochromatic flux zero point.
    ``calibration_band`` is the wavelength interval a surface-brightness
    amplitude is calibrated over: the zero-magnitude source and, under the
    bolometric luminosity rule, the host star are averaged across it.

    Attributes:
        profile_id: Class-level identifier of the profile family; a
            ``DensityDust`` amplitude must carry the same identifier.
        wavelengths_nm: Static wavelength grid of the tabulated scattering
            parameters [nm], a ``Knots``.
        amplitude: Typed density or calibrated surface-brightness amplitude.
        zero_point_jy: Optional monochromatic flux zero point of the calibration
            passband [Jy]. ``None`` uses the passband default. Static.
        calibration_band: Optional ``optixstuff.spectral.Band`` the calibration
            averages over, centred on the calibration passband's center. ``None``
            uses a 506 to 594 nm top hat for Johnson V, and a monochromatic zero
            point at the center for any other passband (which the bolometric
            rule refuses). Static.
    """

    profile_id: eqx.AbstractClassVar[str]
    wavelengths_nm: eqx.AbstractVar[Knots]
    amplitude: eqx.AbstractVar[AbstractDustAmplitude]
    zero_point_jy: float | None = eqx.field(static=True, default=None, kw_only=True)
    calibration_band: Band | None = eqx.field(static=True, default=None, kw_only=True)

    def calibration_bands(self) -> tuple:
        """Return the calibration band when the amplitude uses the bolometric rule.

        Returns:
            A one-element tuple of ``optixstuff.spectral.Band``, or an empty tuple
            for a density amplitude or the band luminosity rule, whose level does
            not read the host star.
        """
        if (
            isinstance(self.amplitude, SurfaceBrightnessDust)
            and self.amplitude.luminosity_rule == "bolometric"
        ):
            return (calibration_band(self.amplitude, self),)
        return ()

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the tabulated scattering parameters [nm].

        Returns:
            1-D array of the wavelengths of the tabulated scattering parameters [nm].
        """
        return self.wavelengths_nm.nm

    def __check_init__(self):
        """Validate the declared disk inputs."""
        check_amplitude(self.amplitude, self)

    @abstractmethod
    def _scattered_integral(
        self, wavelength_nm, east_AU, north_AU, inclination_deg, position_angle_deg
    ):
        """Unit-profile scattering integral at broadcastable sky coordinates."""

    def _photon_radiance_at(
        self,
        wavelength_nm,
        time_d,
        east_arcsec,
        north_arcsec,
        *,
        star,
        dist_pc,
        inclination_deg,
        position_angle_deg,
    ):
        """Photon radiance in photons/s/m^2/nm/sr from the parametric profile.

        Calibration uses the star at epoch zero; rendering follows its photon
        spectrum at time_d. A steady host is distance-independent in radiance
        at a fixed physical radius.
        """
        east, north = jnp.broadcast_arrays(east_arcsec, north_arcsec)
        east_AU, north_AU = arcsec_to_au(east, dist_pc), arcsec_to_au(north, dist_pc)
        wave = jnp.asarray(wavelength_nm, dtype=float)
        level = density_level(self.amplitude, self, star=star, dist_pc=dist_pc)
        integral = jax.vmap(
            lambda w: self._scattered_integral(
                w, east_AU, north_AU, inclination_deg, position_angle_deg
            )
        )(wave.reshape(-1)).reshape(wave.shape + east.shape)
        flux = star.photon_flux_density(wave, time_d, dist_pc=dist_pc)
        flux = flux.reshape(wave.shape + (1,) * east.ndim)
        return level * (dist_pc * pc2AU) ** 2 * flux * integral
