"""Disk wavelength interpolation uses consistent floating-point dtypes."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.constants import arcsec2rad, um2nm
from optixstuff.spectral import Knots

from skyscapes.io.exovista import _load_disk
from skyscapes.scene import FlatStar


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("wavelength_dtype", [">f4", ">f8"])
def test_load_disk_interpolates_without_dtype_warnings(enable_x64, wavelength_dtype):
    # Four wavelengths exercise the general cubic-spline scatter path.
    # The final plane is auxiliary data, not a wavelength channel.
    wavelengths_um = np.array([0.4, 0.5, 0.6, 0.7], dtype=wavelength_dtype)
    cube = np.array([1e-8, 4e-8, 9e-8, 16e-8, -1.0], dtype=">f8")[:, None, None]
    with (
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(wavelengths_um),
                fits.ImageHDU(cube, header=fits.Header({"PXSCLMAS": 10.0})),
            ]
        ) as hdul,
        jax.enable_x64(enable_x64),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        disk = _load_disk(hdul, fits_ext=2)
        brightness = (
            disk.photon_radiance(
                jnp.asarray(550.0),
                0.0,
                star=FlatStar(1.0),
                grid=disk.native_grid,
                dist_pc=10.0,
            ).data
            * disk.native_grid.pixel_area
            * arcsec2rad**2
        )
        dtype = jnp.float64 if enable_x64 else jnp.float32
        # The knots are static host float64 in both precisions: the stored
        # microns widened, then scaled (exact digest).
        widened = np.asarray(wavelengths_um, dtype=np.float64) * um2nm
        assert disk.knots.digest == Knots(widened).digest
        assert disk.contrast_cube.dtype == dtype
        assert brightness.dtype == dtype
        np.testing.assert_array_equal(disk.contrast_cube, cube[:-1].astype(np.float32))
        # The samples follow ((wavelength_nm - 300) / 100)**2 * 1e-8.
        np.testing.assert_allclose(brightness, [[6.25e-8]], rtol=5e-7, atol=0)
        assert disk.native_grid.step == 0.01
