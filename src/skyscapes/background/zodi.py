"""Local zodiacal photon radiance, with physics delegated to zodi."""

from abc import abstractmethod
from typing import final

import equinox as eqx
import interpax
import jax.numpy as jnp
import zodi
from hwoutils.constants import arcsec2rad, nm2um, um2nm
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy
from jaxtyping import Array
from zodi.tables import TABLE19_WAVELENGTH_UM


class AbstractLocalZodi(eqx.Module):
    """Local background radiance in ph s^-1 m^-2 nm^-1 sr^-1.

    Every implementation preserves the wavelength array's shape. Pointing
    angles are degrees; solar_longitude_deg is target minus solar longitude
    (helio-ecliptic longitude difference), as in the zodi tables.
    """

    @abstractmethod
    def photon_radiance(
        self,
        wavelength_nm: Array,
        *,
        ecliptic_latitude_deg: Array,
        solar_longitude_deg: Array,
    ) -> Array:
        """Evaluate photon radiance at wavelengths in nm, without integration."""


@final
class AYOZodi(AbstractLocalZodi):
    """Fixed 22 AB mag/arcsec^2 at 550 nm with Table 19 spectral color.

    Both pointing keywords are accepted and ignored. The reference magnitude
    parameter sets the fixed apparent brightness; zodi owns the color model.
    """

    surface_brightness_mag_arcsec2: float = 22.0
    reference_wavelength_nm: float = 550.0

    @property
    def quadrature_knots_nm(self):
        """Table 19 colour breakpoints [nm]; the spectrum is piecewise in them."""
        return TABLE19_WAVELENGTH_UM * um2nm

    def photon_radiance(
        self,
        wavelength_nm: Array,
        *,
        ecliptic_latitude_deg: Array,
        solar_longitude_deg: Array,
    ) -> Array:
        """Photon radiance per sr; pointing is ignored, wavelength-shaped."""
        wave = jnp.asarray(wavelength_nm)
        mag = zodi.ayo_default_ab_mag(
            wave,
            mag_v=self.surface_brightness_mag_arcsec2,
            reference_wavelength_nm=self.reference_wavelength_nm,
        )
        return jy_to_photons_per_nm_per_m2(mag_to_flux_jy(mag), wave) / arcsec2rad**2


@final
class LeinertZodi(AbstractLocalZodi):
    """Pointing-dependent Leinert radiance anchored on absolute Table 19.

    The absolute scale comes from zodi.specific_intensity, anchor='table19'.
    There is no adjustable reference magnitude. At (longitude=90, latitude=0)
    the V-band brightness is approximately 22 AB mag/arcsec^2; AYO's rounded
    22-mag normalization differs slightly from the absolute table value.
    """

    @property
    def quadrature_knots_nm(self):
        """Table 19 colour breakpoints [nm]; the spectrum is piecewise in them."""
        return TABLE19_WAVELENGTH_UM * um2nm

    def photon_radiance(
        self,
        wavelength_nm: Array,
        *,
        ecliptic_latitude_deg: Array,
        solar_longitude_deg: Array,
    ) -> Array:
        """Photon radiance per sr, converted from zodi's per-micron output."""
        return (
            zodi.specific_intensity(
                jnp.asarray(solar_longitude_deg),
                jnp.asarray(ecliptic_latitude_deg),
                jnp.asarray(wavelength_nm),
                photon_units=True,
                anchor="table19",
            )
            * nm2um
        )


@final
class PrecomputedZodi(AbstractLocalZodi):
    """Linearly interpolated photon radiance supplied in per-steradian units.

    Both pointing keywords are accepted and ignored. Values outside the
    supplied wavelength support are NaN, following interpax's default.
    """

    wavelengths_nm: Array = eqx.field(converter=jnp.asarray)
    radiance_phot_per_s_m2_nm_sr: Array = eqx.field(converter=jnp.asarray)
    _radiance_interp: interpax.Interpolator1D

    def __init__(self, wavelengths_nm: Array, radiance_phot_per_s_m2_nm_sr: Array):
        """Store a photon radiance table in ph s^-1 m^-2 nm^-1 sr^-1."""
        self.wavelengths_nm = jnp.asarray(wavelengths_nm)
        self.radiance_phot_per_s_m2_nm_sr = jnp.asarray(radiance_phot_per_s_m2_nm_sr)
        self._radiance_interp = interpax.Interpolator1D(
            self.wavelengths_nm,
            self.radiance_phot_per_s_m2_nm_sr,
            method="linear",
        )

    def photon_radiance(
        self,
        wavelength_nm: Array,
        *,
        ecliptic_latitude_deg: Array,
        solar_longitude_deg: Array,
    ) -> Array:
        """Photon radiance per sr; pointing is ignored, wavelength-shaped."""
        wave = jnp.asarray(wavelength_nm)
        return self._radiance_interp(wave.reshape(-1)).reshape(wave.shape)
