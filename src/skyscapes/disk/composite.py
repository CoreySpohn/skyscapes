"""Sum of scattered-light components on a caller's sky grid."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from optixstuff.spectral import spectral_breaks

from .base import AbstractDisk


class CompositeDisk(AbstractDisk):
    """Sum radiance from components that may have different native grids.

    Each component applies its own ``flux_scale``; the composite's scales the sum.

    Attributes:
        components: Disks whose radiances are summed; at least one.
        flux_scale: Dimensionless scalar factor on the summed radiance, default one.
    """

    components: tuple[AbstractDisk, ...]
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Union of all component spectral breaks [nm].

        Returns:
            1-D array of the union of the component wavelength breaks [nm].
        """
        return spectral_breaks(*self.components)

    def calibration_bands(self) -> tuple:
        """Calibration bands of all components, each distinct band once.

        Returns:
            The union of the component calibration bands, in component order.
        """
        return tuple(
            dict.fromkeys(b for c in self.components for b in c.calibration_bands())
        )

    def __check_init__(self):
        """Validate the declared disk inputs."""
        if not self.components:
            raise ValueError("CompositeDisk requires at least one component.")

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
        """Sum component photon radiance in photons/s/m^2/nm/sr.

        Every component is read at the same orientation.
        """
        return sum(
            c.photon_radiance_at(
                wavelength_nm,
                time_d,
                east_arcsec,
                north_arcsec,
                star=star,
                dist_pc=dist_pc,
                inclination_deg=inclination_deg,
                position_angle_deg=position_angle_deg,
            )
            for c in self.components
        )
