"""Scattered photon radiance on caller-owned sky grids."""

from abc import abstractmethod
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from optixstuff.grids import Image, Measure, PixelGrid

from .._dtype import _working_dtype
from .._flux_scale import check_flux_scale


def check_sky_grid(grid):
    """Require angular sampling with an explicit detector-to-sky orientation.

    Args:
        grid: Sky grid to validate; it must have unit ``"arcsec"`` and a
            ``SkyOrientation``.
    """
    if grid.unit != "arcsec" or grid.orientation is None:
        raise ValueError("disk grid must be arcsec with a SkyOrientation")


def _sky_coordinates(grid, dtype):
    """Read the caller's sky grid at the radiance reader's working dtype."""
    check_sky_grid(grid)
    rows, columns = np.ogrid[: grid.shape[0], : grid.shape[1]]
    x = jnp.asarray(columns - grid.center_x, dtype=dtype)
    y = jnp.asarray(rows - grid.center_y, dtype=dtype)
    # The host offsets are already centered. Let PixelGrid apply its basis.
    centered = replace(grid, center_x=0.0, center_y=0.0)
    return grid.orientation.detector_to_sky(*centered.pixel_to_unit(x, y))


def radiance_image(data, grid):
    """Name wavelength axes and the radiance unit, without pixel integration.

    Args:
        data: Photon radiance [ph s^-1 m^-2 nm^-1 sr^-1], shape
            ``wavelength_shape + grid.shape``.
        grid: Sky grid whose pixels the last two axes of ``data`` sample.

    Returns:
        Image of ``data`` on ``grid`` with a per-steradian density measure.
    """
    return Image(
        data,
        grid,
        ("wavelength",)
        if data.ndim == 3
        else tuple(f"wavelength_{i}" for i in range(data.ndim - 2)),
        measure=Measure.density("sr"),
        quantity="ph s^-1 m^-2 nm^-1",
    )


class AbstractDisk(eqx.Module):
    """Extended scattered light queried at sky coordinates.

    Values are photon radiance per steradian, independent of pixel area.
    Call ``Image.integrated()`` to integrate over the sky pixels.
    Times are scalar days from the scene epoch; wavelengths have any shape.
    The disk orientation is not a leaf of the disk: a scene owns it
    (``Scene.midplane_inc_deg``, ``Scene.midplane_position_angle_deg``) and every
    read takes ``inclination_deg`` and ``position_angle_deg`` as keyword arguments.

    ``flux_scale`` is a dimensionless factor, default one, on the disk's own
    radiance, applied once in :meth:`photon_radiance_at`; a concrete disk
    implements :meth:`_photon_radiance_at` and never sees it. A disk defined
    relative to the star also follows ``star.flux_scale`` (it reads the star's
    flux). The image is linear in it, and jointly linear in the planet and disk
    scales at a fixed star scale. A parametric disk's scale is exactly
    degenerate with its ``amplitude.level``. It is checked finite, not positive.

    Attributes:
        flux_scale: Dimensionless scalar factor on the disk's own radiance, applied
            once in :meth:`photon_radiance_at`. Concrete disks default it to one.
    """

    flux_scale: eqx.AbstractVar[Array]

    def __check_init__(self):
        """Require a finite ``flux_scale``."""
        check_flux_scale(self.flux_scale)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks in the disk scattering spectrum [nm].

        Returns:
            1-D array of wavelength breaks [nm]; empty here, and overridden by disks
            with a tabulated spectrum.
        """
        return np.empty(0)

    def calibration_bands(self) -> tuple:
        """Bands over which the disk's calibration averages the host star, on the host.

        Returns:
            A tuple of ``optixstuff.spectral.Band``; empty here, and overridden by
            a parametric disk calibrated under the bolometric luminosity rule.
        """
        return ()

    def photon_radiance(
        self,
        wavelength_nm,
        time_d,
        *,
        star,
        dist_pc,
        inclination_deg,
        position_angle_deg,
        grid: PixelGrid,
    ) -> Image:
        """Return an Image with shape ``wavelength_nm.shape + grid.shape``.

        ``grid`` must have unit arcsec and a SkyOrientation. The image has
        a per-steradian density measure and quantity ph s^-1 m^-2 nm^-1.

        Args:
            wavelength_nm: Wavelengths [nm], any shape.
            time_d: Scalar time in days from the scene epoch.
            star: Host star, supplying the photon flux that the disk scatters.
            dist_pc: Distance of the system from the observer [pc].
            inclination_deg: Disk midplane inclination [deg]; zero is face-on.
            position_angle_deg: Disk major-axis position angle, north through
                east [deg]. A stored cube or an azimuthally symmetric disk
                accepts both and ignores them.
            grid: Sky grid on which to evaluate the radiance.

        Returns:
            Photon radiance [ph s^-1 m^-2 nm^-1 sr^-1] as an image of shape
            ``wavelength_nm.shape + grid.shape``.
        """
        dtype = _working_dtype(
            wavelength_nm,
            time_d,
            dist_pc,
            inclination_deg,
            position_angle_deg,
            *jax.tree.leaves((self, star)),
        )
        east, north = _sky_coordinates(grid, dtype)
        data = self.photon_radiance_at(
            wavelength_nm,
            time_d,
            east,
            north,
            star=star,
            dist_pc=dist_pc,
            inclination_deg=inclination_deg,
            position_angle_deg=position_angle_deg,
        )
        return radiance_image(data, grid)

    def photon_radiance_at(
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
    ) -> Array:
        """Photon radiance in photons/s/m^2/nm/sr at the supplied sky offsets.

        Wavelength axes precede the broadcast shape of east_arcsec and
        north_arcsec. time_d is scalar days from the scene epoch. The result is
        ``flux_scale`` times the model's own radiance.

        Args:
            wavelength_nm: Wavelengths [nm], any shape.
            time_d: Scalar time in days from the scene epoch.
            east_arcsec: Sky offsets of the query points toward east of the star
                [arcsec]; broadcast against ``north_arcsec``.
            north_arcsec: Sky offsets of the query points toward north of the star
                [arcsec].
            star: Host star, supplying the photon flux that the disk scatters.
            dist_pc: Distance of the system from the observer [pc].
            inclination_deg: Disk midplane inclination [deg]; zero is face-on.
            position_angle_deg: Disk major-axis position angle, north through
                east [deg]. A stored cube or an azimuthally symmetric disk
                accepts both and ignores them.

        Returns:
            Photon radiance [ph s^-1 m^-2 nm^-1 sr^-1], shape
            ``wavelength_nm.shape + broadcast(east_arcsec, north_arcsec).shape``.
        """
        return self.flux_scale * self._photon_radiance_at(
            wavelength_nm,
            time_d,
            east_arcsec,
            north_arcsec,
            star=star,
            dist_pc=dist_pc,
            inclination_deg=inclination_deg,
            position_angle_deg=position_angle_deg,
        )

    @abstractmethod
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
    ) -> Array:
        """Unscaled photon radiance; the hook a concrete disk implements."""
