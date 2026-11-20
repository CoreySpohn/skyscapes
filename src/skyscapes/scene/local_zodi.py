"""Local (solar-system) zodiacal photon radiance, with physics delegated to zodi."""

from __future__ import annotations

from abc import abstractmethod
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import zodi
from hwoutils.conversions import um_to_nm
from jax import Array
from orbix.observatory import ecliptic_pointing
from orbix.system import SkyFrame
from zodi.tables import TABLE19_WAVELENGTH_UM
from zodi.units import intensity_from_ab_mag_arcsec2, power_to_photon_intensity


class AbstractLocalZodi(eqx.Module):
    """Local background radiance in ph s^-1 m^-2 nm^-1 sr^-1.

    The model is given where and when the scene is seen, the scene's
    :class:`~orbix.system.SkyFrame` and the time in days from its epoch, and
    derives whatever it needs from them. Every implementation preserves the
    wavelength array's shape.
    """

    @abstractmethod
    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the model's spectral breakpoints in nm as a host array."""

    @abstractmethod
    def photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Evaluate photon radiance at wavelengths in nm, without integration.

        Args:
            wavelength_nm: Wavelengths [nm].
            sky_frame: The scene's sky frame: position, distance and epoch. A
                model that depends on the sightline raises a ``ValueError``
                when the frame has no position (an unplaced scene).
            time_d: Observation time, days from ``sky_frame.epoch``.
        """


@final
class AYOZodi(AbstractLocalZodi):
    """Fixed 22 AB mag/arcsec^2 at 550 nm with Table 19 spectral color.

    The sky frame and time are accepted and ignored. The reference magnitude
    parameter sets the fixed apparent brightness; zodi owns the color model.
    """

    surface_brightness_mag_arcsec2: float = 22.0
    reference_wavelength_nm: float = 550.0

    def spectral_breaks_nm(self) -> np.ndarray:
        """Table 19 color breakpoints [nm]; the spectrum is piecewise in them."""
        return um_to_nm(TABLE19_WAVELENGTH_UM)

    def photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Photon radiance per sr; the sightline is ignored, wavelength-shaped."""
        wave = jnp.asarray(wavelength_nm)
        mag = zodi.ayo_default_ab_mag(
            wave,
            mag_v=self.surface_brightness_mag_arcsec2,
            reference_wavelength_nm=self.reference_wavelength_nm,
        )
        return (
            power_to_photon_intensity(intensity_from_ab_mag_arcsec2(mag, wave), wave)
            / 1000.0
        )


@final
class LeinertZodi(AbstractLocalZodi):
    """Sightline-dependent Leinert radiance anchored on absolute Table 19.

    The ecliptic latitude and the helioecliptic longitude (target minus Sun)
    follow from the frame's position and the time through
    :func:`orbix.observatory.ecliptic_pointing`. The absolute scale comes from
    ``zodi.specific_intensity`` with ``anchor='table19'``; there is no
    adjustable reference magnitude. At (longitude=90, latitude=0) the V-band
    brightness is approximately 22 AB mag/arcsec^2; AYO's rounded 22-mag
    normalization differs slightly from the absolute table value.
    """

    def spectral_breaks_nm(self) -> np.ndarray:
        """Table 19 color breakpoints [nm]; the spectrum is piecewise in them."""
        return um_to_nm(TABLE19_WAVELENGTH_UM)

    def photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Photon radiance per sr, converted from zodi's per-micron output."""
        if sky_frame.ra_deg is None:
            raise ValueError(
                "LeinertZodi depends on the sightline, but the scene has no sky "
                "position; build it with Scene.from_classical(..., ra_deg=, "
                "dec_deg=) or use a sightline-independent model such as AYOZodi"
            )
        latitude_deg, longitude_deg = ecliptic_pointing(
            sky_frame.ra_deg, sky_frame.dec_deg, sky_frame.epoch, time_d
        )
        return (
            zodi.specific_intensity(
                longitude_deg,
                latitude_deg,
                jnp.asarray(wavelength_nm),
                photon_units=True,
                anchor="table19",
            )
            / 1000.0
        )


@final
class PrecomputedZodi(AbstractLocalZodi):
    """Linearly interpolated photon radiance supplied in per-steradian units.

    The sky frame and time are accepted and ignored. Values outside the
    supplied wavelength support are NaN.
    """

    wavelengths_nm: Array = eqx.field(converter=jnp.asarray)
    radiance_phot_per_s_m2_nm_sr: Array = eqx.field(converter=jnp.asarray)

    def __init__(self, wavelengths_nm: Array, radiance_phot_per_s_m2_nm_sr: Array):
        """Store a photon radiance table in ph s^-1 m^-2 nm^-1 sr^-1."""
        self.wavelengths_nm = jnp.asarray(wavelengths_nm)
        self.radiance_phot_per_s_m2_nm_sr = jnp.asarray(radiance_phot_per_s_m2_nm_sr)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the tabulated radiance wavelengths in nm on the host."""
        return np.asarray(self.wavelengths_nm)

    def photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Photon radiance per sr; the sightline is ignored, wavelength-shaped."""
        wave = jnp.asarray(wavelength_nm)
        return jnp.interp(
            wave,
            self.wavelengths_nm,
            self.radiance_phot_per_s_m2_nm_sr,
            left=jnp.nan,
            right=jnp.nan,
        )
