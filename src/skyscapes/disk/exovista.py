"""ExovistaDisk: port of legacy Disk -- wavelength-interpolated contrast cube."""

from __future__ import annotations

import interpax
from jaxtyping import Array

from .base import AbstractDisk


class ExovistaDisk(AbstractDisk):
    """Wavelength-interpolated 3D contrast cube loaded from ExoVista FITS.

    Attributes:
        pixel_scale_arcsec: Pixel scale [arcsec/pixel].
        wavelengths_nm: 1-D wavelength grid [nm], shape ``(n_wl,)``.
        contrast_cube: Contrast cube, shape ``(n_wl, ny, nx)``.
    """

    pixel_scale_arcsec: float
    wavelengths_nm: Array
    contrast_cube: Array
    _contrast_interp: interpax.CubicSpline

    def __init__(
        self,
        pixel_scale_arcsec: float,
        wavelengths_nm: Array,
        contrast_cube: Array,
    ):
        """Store geometry and pre-build the wavelength cubic spline."""
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.wavelengths_nm = wavelengths_nm
        self.contrast_cube = contrast_cube
        self._contrast_interp = interpax.CubicSpline(
            wavelengths_nm, contrast_cube, axis=0
        )

    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_jd: Array,
        incl_deg: Array,
        pa_deg: Array,
    ) -> Array:
        """Contrast map at the requested wavelength, shape ``(ny, nx)``.

        ``time_jd``, ``incl_deg``, and ``pa_deg`` are part of the
        AbstractDisk interface but ignored here: the cube is a single
        time snapshot with disk geometry already baked in by the loader.
        """
        return self._contrast_interp(wavelength_nm)

    def spatial_extent(self) -> tuple[float, float]:
        """Return ``(width_arcsec, height_arcsec)``."""
        ny, nx = self.contrast_cube.shape[-2:]
        return (nx * self.pixel_scale_arcsec, ny * self.pixel_scale_arcsec)
