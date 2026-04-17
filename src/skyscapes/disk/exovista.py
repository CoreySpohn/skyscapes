"""ExovistaDisk: port of legacy Disk — wavelength-interpolated contrast cube."""

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
        inclination_deg: Disk midplane inclination [deg]. Metadata only —
            no geometric projection is applied by ``surface_brightness``.
            The contrast cube is already rendered in the on-sky frame by
            ExoVista, so these fields exist to record the midplane that
            orbit-carrying consumers (e.g. EXOSIMS) need when rotating
            planets from midplane-frame elements into sky coords.
        position_angle_deg: Disk midplane PA [deg]. Metadata only.
    """

    pixel_scale_arcsec: float
    wavelengths_nm: Array
    contrast_cube: Array
    inclination_deg: float
    position_angle_deg: float
    _contrast_interp: interpax.CubicSpline

    def __init__(
        self,
        pixel_scale_arcsec: float,
        wavelengths_nm: Array,
        contrast_cube: Array,
        inclination_deg: float = 0.0,
        position_angle_deg: float = 0.0,
    ):
        """Store geometry and pre-build the wavelength cubic spline."""
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.wavelengths_nm = wavelengths_nm
        self.contrast_cube = contrast_cube
        self.inclination_deg = inclination_deg
        self.position_angle_deg = position_angle_deg
        self._contrast_interp = interpax.CubicSpline(
            wavelengths_nm, contrast_cube, axis=0
        )

    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Contrast map at the requested wavelength, shape ``(ny, nx)``.

        ``time_jd`` is part of the AbstractDisk interface but ExovistaDisk
        is static in time (the cube is a single snapshot).
        """
        return self._contrast_interp(wavelength_nm)

    def spatial_extent(self) -> tuple[float, float]:
        """Return ``(width_arcsec, height_arcsec)``."""
        ny, nx = self.contrast_cube.shape[-2:]
        return (nx * self.pixel_scale_arcsec, ny * self.pixel_scale_arcsec)
