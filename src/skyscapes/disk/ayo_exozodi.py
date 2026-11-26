"""The AYO analytic exozodi as a disk model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import arcsec_to_au
from hwoutils.precision import like
from jaxtyping import Array
from zodi.exozodi import MAG_1ZODI_V_ARCSEC2, MV_SUN, exozodi_flux_ratio_v
from zodi.units import intensity_from_ab_mag_arcsec2, power_to_photon_intensity

from .._dtype import _working_dtype
from .base import AbstractDisk


class AYOExozodi(AbstractDisk):
    """Scalar AYO exozodi with flat AB colour and inverse-square illumination.

    At 1 AU, Fexozodi = level_zodis * 10**(-0.4 * brightness), per arcsec^2,
    where brightness is surface_brightness_mag_arcsec2 for one zodi.
    Its denominator is the AB magnitude-zero flux density F0(lambda), from
    hwoutils.constants.AB_ZERO_POINT_JY, not the host-star flux. Radiance is
    F0 * Fexozodi divided by arcsec2rad**2 and separation_au**2. There is no
    host colour, luminosity, or inclination correction. The analytic AB zero point has
    no table knots; photon colour is 1/lambda (constant F_nu).

    One 22-mag/arcsec^2 zodi is 10^-8.8 relative to a magnitude-zero source.
    The alternative approximately 10^-6.87 ratio uses a solar absolute-V
    magnitude 4.83 source at 10 pc as denominator. Face-on and inclined
    definitions also differ by a factor 0.49; no such conversion is applied
    here. These dialects must not be substituted without conversion.

    The radiance depends only on the projected physical separation
    ``dist_pc * hypot(east, north)`` from the star, not the orbital radius, so
    the disk is face-on symmetric and ignores time, the star and the scene
    midplane angles. The scalar law is singular at zero separation: sample it
    away from the star pixel.

    Attributes:
        level_zodis: Exozodi level, in zodis.
        surface_brightness_mag_arcsec2: AB surface brightness of one zodi at
            1 AU.
        flux_scale: Dimensionless factor on the radiance, default one; see
            :class:`~skyscapes.disk.AbstractDisk`.
    """

    level_zodis: float | Array = 3.0
    surface_brightness_mag_arcsec2: float | Array = MAG_1ZODI_V_ARCSEC2
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)

    def spectral_breaks_nm(self) -> np.ndarray:
        """The analytic flat-AB spectrum has no wavelength breaks.

        Returns:
            Empty 1-D array of wavelength breaks [nm].
        """
        return np.empty(0)

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
        """Photons/s/m^2/nm/sr, wavelength axes first, then the sky shape.

        The analytic model is azimuthally symmetric: ``inclination_deg`` and
        ``position_angle_deg`` are accepted and ignored.
        """
        dtype = _working_dtype(
            wavelength_nm,
            east_arcsec,
            north_arcsec,
            dist_pc,
            self.flux_scale,
            self.level_zodis,
            self.surface_brightness_mag_arcsec2,
        )
        wave = jnp.asarray(wavelength_nm, dtype=dtype)
        east = jnp.asarray(east_arcsec, dtype=dtype)
        north = jnp.asarray(north_arcsec, dtype=dtype)
        separation = jnp.hypot(east, north)
        wave = wave.reshape(wave.shape + (1,) * separation.ndim)
        magnitude = self.surface_brightness_mag_arcsec2
        if hasattr(magnitude, "dtype"):
            magnitude = like(magnitude, wave)
        # The zodi law evaluates a plain host magnitude's attenuation before
        # multiplying by device arrays. Preserve that host calculation.
        ratio = exozodi_flux_ratio_v(
            like(self.level_zodis, wave),
            like(MV_SUN, wave),
            arcsec_to_au(separation, like(dist_pc, wave)),
            mag_1zodi=magnitude,
        )
        intensity = intensity_from_ab_mag_arcsec2(like(0.0, wave), wave) * ratio
        return power_to_photon_intensity(intensity, wave) / 1000.0
