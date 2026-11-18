"""Sum of scattered-light components on a caller's sky grid."""

import equinox as eqx

from .base import AbstractDisk


def orient_disk(disk, inclination_deg, position_angle_deg):
    """Return a parametric disk at the system midplane; stored cubes stay fixed."""
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
    """Sum radiance from components that may have different native grids."""

    components: tuple[AbstractDisk, ...]

    def __check_init__(self):
        """Validate the declared disk inputs."""
        if not self.components:
            raise ValueError("CompositeDisk requires at least one component.")

    def photon_radiance_at(
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
