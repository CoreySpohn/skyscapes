"""Parametric disk: power-law radial surface brightness."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractDisk


class ParametricDisk(AbstractDisk):
    """Single-power-law radial disk, grey spectrum.

    Surface brightness is ``contrast_0 * (r / r0)^alpha`` for radial
    distance ``r`` in arcsec measured from the grid center.

    Attributes:
        contrast_0: Reference contrast at ``r = r0_arcsec``.
        r0_arcsec: Reference radius [arcsec].
        alpha: Power-law exponent (dimensionless).
        pixel_scale_arcsec: Pixel scale [arcsec/pixel].
        shape: ``(ny, nx)`` pixel grid dimensions, static.
        inclination_deg: Disk midplane inclination [deg]. Metadata only —
            no geometric projection is applied by ``surface_brightness``.
            Plan-5 follow-up will add a system-level frame conversion.
        position_angle_deg: Disk midplane PA [deg]. Metadata only.
    """

    contrast_0: Array
    r0_arcsec: Array
    alpha: Array
    pixel_scale_arcsec: float
    shape: tuple[int, int] = eqx.field(static=True)
    inclination_deg: float = 0.0
    position_angle_deg: float = 0.0

    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Power-law contrast map, shape ``(ny, nx)``.

        ``wavelength_nm`` and ``time_jd`` are part of the AbstractDisk
        interface but the grey parametric disk is wavelength- and
        time-independent.
        """
        ny, nx = self.shape
        y = (jnp.arange(ny) - (ny - 1) / 2.0) * self.pixel_scale_arcsec
        x = (jnp.arange(nx) - (nx - 1) / 2.0) * self.pixel_scale_arcsec
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        r = jnp.sqrt(xx**2 + yy**2)
        eps = jnp.finfo(r.dtype).tiny
        return self.contrast_0 * ((r + eps) / self.r0_arcsec) ** self.alpha

    def spatial_extent(self) -> tuple[float, float]:
        """Return ``(width_arcsec, height_arcsec)``."""
        ny, nx = self.shape
        return (nx * self.pixel_scale_arcsec, ny * self.pixel_scale_arcsec)
