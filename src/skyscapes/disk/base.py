"""Scattered photon radiance on caller-owned sky grids."""

from abc import abstractmethod

import equinox as eqx
from hwoutils.grids import Image, PixelGrid
from jaxtyping import Array


def check_sky_grid(grid):
    """Require angular sampling with an explicit detector-to-sky orientation."""
    if grid.unit != "arcsec" or grid.orientation is None:
        raise ValueError("disk grid must be arcsec with a SkyOrientation")


def radiance_image(data, grid):
    """Name wavelength axes and the radiance unit, without pixel integration."""
    return Image(
        data,
        grid,
        tuple(f"wavelength_{i}" for i in range(data.ndim - 2)),
        measure="per_unit_area",
        quantity="ph s^-1 m^-2 nm^-1 sr^-1",
    )


class AbstractDisk(eqx.Module):
    """Extended scattered light queried at sky coordinates.

    Values are photon radiance per steradian, independent of pixel area.
    Consumers integrate a pixel by multiplying by
    ``grid.pixel_area * hwoutils.constants.arcsec2rad**2``.
    Times are scalar days from the scene epoch; wavelengths have any shape.
    """

    def photon_radiance(
        self, wavelength_nm, time_d, *, star, dist_pc, grid: PixelGrid
    ) -> Image:
        """Return an Image with shape ``wavelength_nm.shape + grid.shape``.

        ``grid`` must have unit arcsec and a SkyOrientation. The image has
        measure per_unit_area and quantity ph s^-1 m^-2 nm^-1 sr^-1.
        """
        check_sky_grid(grid)
        east, north = grid.orientation.detector_to_sky(*grid.coordinates())
        data = self.photon_radiance_at(
            wavelength_nm, time_d, east, north, star=star, dist_pc=dist_pc
        )
        return radiance_image(data, grid)

    @abstractmethod
    def photon_radiance_at(
        self, wavelength_nm, time_d, east_arcsec, north_arcsec, *, star, dist_pc
    ) -> Array:
        """Photon radiance in photons/s/m^2/nm/sr at the supplied sky offsets.

        Wavelength axes precede the broadcast shape of east_arcsec and
        north_arcsec. time_d is scalar days from the scene epoch.
        """
