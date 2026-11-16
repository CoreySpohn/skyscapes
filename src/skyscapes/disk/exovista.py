"""Stored host-relative pixel contrasts with explicit native sky geometry."""

import equinox as eqx
import interpax
import jax.numpy as jnp
from hwoutils.constants import arcsec2rad
from hwoutils.grids import PixelGrid, resample
from jaxtyping import Array

from .base import AbstractDisk, check_sky_grid, radiance_image


class ExovistaDisk(AbstractDisk):
    """Wavelength-interpolated contrast cube on a native PixelGrid.

    contrast_cube holds F_pixel/Phi_star, shape (n_wave, ny, nx). Since
    F_pixel = L * Omega_native and Omega_native = native_grid.pixel_area
    * arcsec2rad**2 sr, L = contrast_cube * Phi_star / Omega_native.
    Resampling L as per_unit_area preserves density, with no target pixel
    area factor. Geometry is baked into the cube; system angles are ignored.
    """

    native_grid: PixelGrid = eqx.field(static=True)
    wavelengths_nm: Array
    contrast_cube: Array
    _contrast_interp: interpax.CubicSpline

    def __init__(self, native_grid, wavelengths_nm, contrast_cube):
        """Store the native geometry and construct the spectral interpolant."""
        self.native_grid = native_grid
        self.wavelengths_nm = wavelengths_nm
        self.contrast_cube = contrast_cube
        self._contrast_interp = interpax.CubicSpline(
            wavelengths_nm, contrast_cube, axis=0
        )

    def __check_init__(self):
        """Validate the declared disk inputs."""
        check_sky_grid(self.native_grid)
        if self.contrast_cube.shape != (
            len(self.wavelengths_nm),
            *self.native_grid.shape,
        ):
            raise ValueError("contrast_cube must match wavelengths and native_grid")

    def photon_radiance(self, wavelength_nm, time_d, *, star, dist_pc, grid):
        """Convert native pixel contrasts to radiance, then resample density."""
        check_sky_grid(grid)
        wave = jnp.asarray(wavelength_nm, dtype=float)
        contrast = self._contrast_interp(wave.reshape(-1)).reshape(
            wave.shape + self.native_grid.shape
        )
        flux = star.photon_flux_density(wave, time_d, dist_pc=dist_pc)
        data = (
            contrast
            * flux[..., None, None]
            / (self.native_grid.pixel_area * arcsec2rad**2)
        )
        native = radiance_image(data, self.native_grid)
        return native if grid == self.native_grid else resample(native, grid)
