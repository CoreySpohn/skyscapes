"""Shared photon rendering of typed parametric scattering profiles."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import pc2AU
from zodi import AbstractDustAmplitude

from .amplitude import check_amplitude, density_level
from .base import AbstractDisk, check_sky_grid, radiance_image


class AbstractParametricDisk(AbstractDisk):
    """Profile owning its amplitude and astronomical orientation.

    The declared unit profile is extinction per AU. zero_point_jy optionally
    overrides the calibration passband's monochromatic flux zero point.
    """

    profile_id: eqx.AbstractClassVar[str]
    amplitude: eqx.AbstractVar[AbstractDustAmplitude]
    inclination_deg: eqx.AbstractVar[jax.Array]
    position_angle_deg: eqx.AbstractVar[jax.Array]
    zero_point_jy: float | None = eqx.field(static=True, default=None, kw_only=True)

    def __check_init__(self):
        """Validate the declared disk inputs."""
        check_amplitude(self.amplitude, self)

    @abstractmethod
    def _scattered_integral(
        self, wavelength_nm, east_AU, north_AU, inclination_deg, position_angle_deg
    ):
        """Unit-profile scattering integral on explicit 2D sky coordinates."""

    def photon_radiance(self, wavelength_nm, time_d, *, star, dist_pc, grid):
        """Sample photon radiance; arbitrary wavelength axes precede (y,x).

        Calibration uses the star at epoch zero; rendering follows its photon
        spectrum at time_d. A steady host is distance-independent in radiance
        at a fixed physical radius. Orientation comes only from grid metadata.
        """
        check_sky_grid(grid)
        east, north = grid.orientation.detector_to_sky(*grid.coordinates())
        wave = jnp.asarray(wavelength_nm, dtype=float)
        level = density_level(self.amplitude, self, star=star, dist_pc=dist_pc)
        integral = jax.vmap(
            lambda w: self._scattered_integral(
                w,
                east * dist_pc,
                north * dist_pc,
                self.inclination_deg,
                self.position_angle_deg,
            )
        )(wave.reshape(-1)).reshape(wave.shape + grid.shape)
        flux = star.photon_flux_density(wave, time_d, dist_pc=dist_pc)
        return radiance_image(
            level * (dist_pc * pc2AU) ** 2 * flux[..., None, None] * integral, grid
        )
