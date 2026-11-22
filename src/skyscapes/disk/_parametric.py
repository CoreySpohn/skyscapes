"""Shared photon rendering of typed parametric scattering profiles."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import pc2AU
from hwoutils.conversions import arcsec_to_au
from zodi import AbstractDustAmplitude

from .amplitude import check_amplitude, density_level
from .base import AbstractDisk


class AbstractParametricDisk(AbstractDisk):
    """Profile owning its amplitude and astronomical orientation.

    The declared unit profile is extinction per AU. zero_point_jy optionally
    overrides the calibration passband's monochromatic flux zero point.

    Attributes:
        profile_id: Class-level identifier of the profile family; a
            ``DensityDust`` amplitude must carry the same identifier.
        wavelengths_nm: 1-D wavelength grid of the tabulated scattering
            parameters [nm].
        amplitude: Typed density or calibrated surface-brightness amplitude.
        inclination_deg: Inclination of the disk plane [deg]; zero is face-on.
        position_angle_deg: Position angle of the major axis, north through east
            [deg].
        zero_point_jy: Optional monochromatic flux zero point of the calibration
            passband [Jy]. ``None`` uses the passband default. Static.
    """

    profile_id: eqx.AbstractClassVar[str]
    wavelengths_nm: eqx.AbstractVar[jax.Array]
    amplitude: eqx.AbstractVar[AbstractDustAmplitude]
    inclination_deg: eqx.AbstractVar[jax.Array]
    position_angle_deg: eqx.AbstractVar[jax.Array]
    zero_point_jy: float | None = eqx.field(static=True, default=None, kw_only=True)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the tabulated scattering parameters [nm].

        Returns:
            1-D array of the wavelengths of the tabulated scattering parameters [nm].
        """
        return np.asarray(self.wavelengths_nm)

    def __check_init__(self):
        """Validate the declared disk inputs."""
        check_amplitude(self.amplitude, self)

    @abstractmethod
    def _scattered_integral(
        self, wavelength_nm, east_AU, north_AU, inclination_deg, position_angle_deg
    ):
        """Unit-profile scattering integral at broadcastable sky coordinates."""

    def _photon_radiance_at(
        self, wavelength_nm, time_d, east_arcsec, north_arcsec, *, star, dist_pc
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
                w, east_AU, north_AU, self.inclination_deg, self.position_angle_deg
            )
        )(wave.reshape(-1)).reshape(wave.shape + east.shape)
        flux = star.photon_flux_density(wave, time_d, dist_pc=dist_pc)
        flux = flux.reshape(wave.shape + (1,) * east.ndim)
        return level * (dist_pc * pc2AU) ** 2 * flux * integral
