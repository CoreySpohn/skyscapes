"""Uniform disk: constant contrast across a rectangular grid."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractDisk


class UniformDisk(AbstractDisk):
    """Flat, constant-brightness disk.

    Attributes:
        contrast: Scalar flux ratio applied to every pixel.
        pixel_scale_arcsec: Pixel scale [arcsec/pixel].
        shape: ``(ny, nx)`` pixel grid dimensions, static.
        inclination_deg: Disk midplane inclination [deg]. Metadata only —
            no geometric projection is applied by ``surface_brightness``.
            Plan-5 follow-up will add a system-level frame conversion.
        position_angle_deg: Disk midplane PA [deg]. Metadata only.
    """

    contrast: Array
    pixel_scale_arcsec: float
    shape: tuple[int, int] = eqx.field(static=True)
    inclination_deg: float = 0.0
    position_angle_deg: float = 0.0

    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Constant contrast map, shape ``(ny, nx)``.

        ``wavelength_nm`` and ``time_jd`` are part of the AbstractDisk
        interface but the uniform disk is wavelength- and time-independent.
        """
        return jnp.full(self.shape, self.contrast)

    def spatial_extent(self) -> tuple[float, float]:
        """Return ``(width_arcsec, height_arcsec)``."""
        ny, nx = self.shape
        return (nx * self.pixel_scale_arcsec, ny * self.pixel_scale_arcsec)
