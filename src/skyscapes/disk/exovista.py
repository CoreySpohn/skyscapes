"""Stored host-relative pixel contrasts with explicit native sky geometry."""

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.map_coordinates import map_coordinates
from jaxtyping import Array
from optixstuff.grids import PixelGrid

from .base import AbstractDisk, check_sky_grid


class ExovistaDisk(AbstractDisk):
    """Wavelength-interpolated contrast cube on a native PixelGrid.

    contrast_cube holds F_pixel/Phi_star, shape (n_wave, ny, nx). Since
    F_pixel = L * Omega_native and Omega_native = native_grid.cell_area("sr");
    L = contrast_cube * Phi_star / Omega_native.
    Spatial sampling is bilinear, with zero extension of the native samples.
    There is no target pixel area factor. Geometry is baked into the cube;
    system angles are ignored.
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

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the stored contrast cube [nm]."""
        return np.asarray(self.wavelengths_nm)

    def __check_init__(self):
        """Validate the declared disk inputs."""
        check_sky_grid(self.native_grid)
        if self.contrast_cube.shape != (
            len(self.wavelengths_nm),
            *self.native_grid.shape,
        ):
            raise ValueError("contrast_cube must match wavelengths and native_grid")

    def photon_radiance_at(
        self, wavelength_nm, time_d, east_arcsec, north_arcsec, *, star, dist_pc
    ):
        """Photon radiance in photons/s/m^2/nm/sr from the stored contrast cube."""
        east, north = jnp.broadcast_arrays(east_arcsec, north_arcsec)
        grid = self.native_grid
        pixels = grid.unit_to_pixel(*grid.orientation.sky_to_detector(east, north))
        spline = self._contrast_interp

        def at_wavelength(wave):
            # A wavelength needs only the four planes from its spline segment.
            index = jnp.clip(
                jnp.searchsorted(spline.x, wave, side="right") - 1,
                0,
                spline.x.size - 2,
            )
            knots = jax.lax.dynamic_slice_in_dim(spline.x, index, 2)
            planes = spline.c[:, index].astype(jnp.result_type(spline.c, wave))

            def sample(plane):
                return map_coordinates(plane, (pixels.y, pixels.x), order=1)

            # Spatial interpolation commutes with the spectral polynomial.
            coefficients = jax.vmap(sample)(planes)
            sampled = interpax.PPoly.construct_fast(
                coefficients[:, None], knots, spline.extrapolate
            )
            return sampled(wave)

        wave = jnp.asarray(wavelength_nm, dtype=float)
        contrast = jax.vmap(at_wavelength)(wave.reshape(-1))
        contrast = contrast.reshape(wave.shape + east.shape)
        flux = star.photon_flux_density(wave, time_d, dist_pc=dist_pc)
        flux = flux.reshape(wave.shape + (1,) * east.ndim)
        return contrast * flux / grid.cell_area("sr")
