"""Sum of scattered-light components on a caller's sky grid."""

import equinox as eqx

from .base import AbstractDisk, radiance_image


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
    """Components may have different native grids; each renders on the target."""

    components: tuple[AbstractDisk, ...]

    def __check_init__(self):
        """Validate the declared disk inputs."""
        if not self.components:
            raise ValueError("CompositeDisk requires at least one component.")

    def photon_radiance(self, wavelength_nm, time_d, *, star, dist_pc, grid):
        """Sum photon radiance densities on exactly the requested grid."""
        data = sum(
            c.photon_radiance(
                wavelength_nm, time_d, star=star, dist_pc=dist_pc, grid=grid
            ).data
            for c in self.components
        )
        return radiance_image(data, grid)
