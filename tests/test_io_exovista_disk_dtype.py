"""Disk wavelength interpolation uses consistent floating-point dtypes."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from skyscapes.io.exovista import _load_disk


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
        brightness = disk.surface_brightness(jnp.asarray(550.0), 0.0, 0.0, 0.0)
        dtype = jnp.float64 if enable_x64 else jnp.float32
        assert disk.wavelengths_nm.dtype == dtype
        assert disk.contrast_cube.dtype == dtype
        assert brightness.dtype == dtype
        np.testing.assert_array_equal(disk.contrast_cube, cube[:-1].astype(np.float32))
        # The samples follow ((wavelength_nm - 300) / 100)**2 * 1e-8.
        np.testing.assert_allclose(brightness, [[6.25e-8]], rtol=5e-7, atol=0)
        assert disk.pixel_scale_arcsec == 0.01
