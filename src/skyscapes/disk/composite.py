"""Sum of scattered-light components on a caller's sky grid."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from optixstuff.spectral import spectral_breaks

from .base import AbstractDisk


def orient_disk(disk, inclination_deg, position_angle_deg):
    """Return a parametric disk at the system midplane; stored cubes stay fixed.

    Args:
        disk: Disk to orient; composites are oriented component by component.
        inclination_deg: Midplane inclination [deg]; zero is face-on.
        position_angle_deg: Midplane position angle, north through east [deg].

    Returns:
        The disk with parametric components set to the given orientation. Other
        disks, such as stored cubes, are returned unchanged.
    """
    from ._parametric import AbstractParametricDisk

    if isinstance(disk, CompositeDisk):
        return eqx.tree_at(
            lambda d: d.components,
            disk,
            tuple(
                orient_disk(c, inclination_deg, position_angle_deg)
                for c in disk.components
            ),
        )
    if isinstance(disk, AbstractParametricDisk):
        return eqx.tree_at(
            lambda d: (d.inclination_deg, d.position_angle_deg),
            disk,
            (inclination_deg, position_angle_deg),
        )
    return disk


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
        self, wavelength_nm, time_d, east_arcsec, north_arcsec, *, star, dist_pc
    ):
        """Sum component photon radiance in photons/s/m^2/nm/sr."""
        return sum(
            c.photon_radiance_at(
                wavelength_nm,
                time_d,
                east_arcsec,
                north_arcsec,
                star=star,
                dist_pc=dist_pc,
            )
            for c in self.components
        )
