"""Zodiacal-light background sources.

Three concrete variants:

- :class:`AYOZodi` -- fixed V-band surface brightness with the
  Leinert wavelength-dependent color correction. Matches the AYO/EXOSIMS
  ETC Calibration Task Group convention (135 deg solar longitude).
- :class:`LeinertZodi` -- full Leinert et al. (1998) tables for
  position- and wavelength-dependence.
- :class:`PrecomputedZodi` -- pre-computed photon flux supplied
  externally (e.g. extracted from an EXOSIMS/pyEDITH run).

All three return ph/s/m^2/nm per arcsec^2. Each is its own ``eqx.Module``;
there is intentionally no shared abstract base yet (see the package
docstring).
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import interpax
import jax.numpy as jnp
from hwoutils.constants import Jy, h
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy

from skyscapes.background.leinert import (
    create_zodi_spectrum_jax,
    leinert_zodi_mag,
    zodi_color_correction,
)


@final
class AYOZodi(eqx.Module):
    """Zodiacal light using AYO-compatible default settings.

    Uses a fixed surface brightness at V-band with a Leinert
    wavelength-dependent color correction. Position-independent at query
    time -- the AYO convention bakes in solar longitude 135 deg.

    Example:
        >>> import jax.numpy as jnp
        >>> wavelengths = jnp.linspace(400, 1000, 50)
        >>> zodi = AYOZodi(wavelengths, surface_brightness_mag=22.0)
    """

    _wavelengths_nm: jnp.ndarray
    _flux_density_phot: jnp.ndarray
    _flux_interp: interpax.Interpolator1D
    _reference_wavelength_nm: float
    _reference_mag_arcsec2: float

    def __init__(
        self,
        wavelengths_nm: jnp.ndarray,
        surface_brightness_mag: float = 22.0,
        reference_wavelength_nm: float = 550.0,
    ):
        """Initialise AYOZodi.

        Args:
            wavelengths_nm: Array of wavelengths in nm spanning the
                queryable range.
            surface_brightness_mag: Surface brightness at the reference
                wavelength in mag/arcsec^2. Default 22.0 (AYO standard).
            reference_wavelength_nm: Reference wavelength in nm
                (default 550 = V-band).
        """
        self._wavelengths_nm = wavelengths_nm
        self._reference_wavelength_nm = reference_wavelength_nm
        self._reference_mag_arcsec2 = surface_brightness_mag

        reference_flux_jy = mag_to_flux_jy(surface_brightness_mag)
        flux_spectrum_jy = create_zodi_spectrum_jax(
            wavelengths_nm,
            reference_flux_jy=reference_flux_jy,
            reference_wavelength_nm=reference_wavelength_nm,
        )
        self._flux_density_phot = jy_to_photons_per_nm_per_m2(
            flux_spectrum_jy, wavelengths_nm
        )
        self._flux_interp = interpax.Interpolator1D(
            wavelengths_nm, self._flux_density_phot, method="linear"
        )

    @property
    def reference_wavelength_nm(self) -> float:
        """Reference wavelength for the zodi model in nm."""
        return self._reference_wavelength_nm

    @property
    def reference_mag_arcsec2(self) -> float:
        """Surface brightness at the reference wavelength in mag/arcsec^2."""
        return self._reference_mag_arcsec2

    def spec_flux_density(
        self,
        wavelength_nm: float,
        time_jd: float,
        ecliptic_lat_deg: float = 0.0,
        solar_lon_deg: float = 135.0,
    ) -> float:
        """Return surface brightness in ph/s/m^2/nm per arcsec^2.

        ``time_jd`` and the ecliptic/solar arguments are accepted for
        interface compatibility but ignored: the AYO convention is a
        fixed-angle assumption.
        """
        return self._flux_interp(wavelength_nm)

    def __repr__(self) -> str:
        """One-line summary of the AYO zodi reference + wavelength grid."""
        wl_min = float(self._wavelengths_nm.min())
        wl_max = float(self._wavelengths_nm.max())
        return (
            f"AYOZodi(mag={self._reference_mag_arcsec2:.2f}/arcsec^2 "
            f"@ {self._reference_wavelength_nm:.0f} nm, "
            f"wl={wl_min:.0f}-{wl_max:.0f} nm, "
            f"n_wl={int(self._wavelengths_nm.size)})"
        )


@final
class LeinertZodi(eqx.Module):
    """Zodiacal light using the full Leinert (1998) position-dependent model.

    Computes the surface brightness dynamically from the Leinert et al.
    (1998) tables for both position (ecliptic latitude, solar longitude)
    and wavelength dependence.

    Example:
        >>> zodi = LeinertZodi(reference_mag_arcsec2=22.0)
        >>> flux1 = zodi.spec_flux_density(550.0, 0.0, ecliptic_lat_deg=30.0)
        >>> flux2 = zodi.spec_flux_density(550.0, 0.0, ecliptic_lat_deg=45.0)
    """

    _reference_wavelength_nm: float
    _reference_mag_arcsec2: float

    def __init__(
        self,
        reference_mag_arcsec2: float = 22.0,
        reference_wavelength_nm: float = 550.0,
    ):
        """Initialise LeinertZodi.

        Args:
            reference_mag_arcsec2: Reference surface brightness in
                mag/arcsec^2 at the reference position (ecliptic pole,
                solar longitude 90 deg).
            reference_wavelength_nm: Reference wavelength in nm
                (default 550 = V-band).
        """
        self._reference_mag_arcsec2 = reference_mag_arcsec2
        self._reference_wavelength_nm = reference_wavelength_nm

    @property
    def reference_wavelength_nm(self) -> float:
        """Reference wavelength for the zodi model in nm."""
        return self._reference_wavelength_nm

    @property
    def reference_mag_arcsec2(self) -> float:
        """Surface brightness at the reference wavelength in mag/arcsec^2."""
        return self._reference_mag_arcsec2

    def spec_flux_density(
        self,
        wavelength_nm: float,
        time_jd: float,
        ecliptic_lat_deg: float = 0.0,
        solar_lon_deg: float = 135.0,
    ) -> float:
        """Return surface brightness in ph/s/m^2/nm per arcsec^2.

        Args:
            wavelength_nm: Scalar wavelength in nm.
            time_jd: Scalar time in Julian days (ignored).
            ecliptic_lat_deg: Ecliptic latitude in degrees.
            solar_lon_deg: Solar longitude in degrees.
        """
        position_mag = leinert_zodi_mag(
            self._reference_wavelength_nm, ecliptic_lat_deg, solar_lon_deg
        )
        flux_jy_ref = mag_to_flux_jy(position_mag)
        color_correction = zodi_color_correction(
            wavelength_nm, self._reference_wavelength_nm, photon_units=True
        )
        flux_jy = (
            flux_jy_ref
            * color_correction
            * (wavelength_nm / self._reference_wavelength_nm) ** 2
        )
        flux_phot = flux_jy * Jy / (wavelength_nm * h)
        return flux_phot

    def __repr__(self) -> str:
        """One-line summary of the Leinert zodi reference."""
        return (
            f"LeinertZodi(ref_mag={self._reference_mag_arcsec2:.2f}/arcsec^2 "
            f"@ {self._reference_wavelength_nm:.0f} nm; "
            f"position-dependent)"
        )


@final
class PrecomputedZodi(eqx.Module):
    """Zodiacal light from pre-computed photon flux values.

    Wraps an externally computed array of photon fluxes (e.g. extracted
    from an EXOSIMS or pyEDITH run) so the same values flow through the
    coronagraph image simulator without recomputation. Position and time
    arguments are accepted for interface compatibility but ignored.

    Example:
        >>> exosims_flux = ...  # ph/s/m^2/nm/arcsec^2 from EXOSIMS
        >>> zodi = PrecomputedZodi(wavelengths, exosims_flux)
    """

    _wavelengths_nm: jnp.ndarray
    _flux_density_phot: jnp.ndarray
    _flux_interp: interpax.Interpolator1D
    _reference_wavelength_nm: float
    _reference_mag_arcsec2: float

    def __init__(
        self,
        wavelengths_nm: jnp.ndarray,
        flux_phot_per_arcsec2: jnp.ndarray,
        reference_mag_arcsec2: float = 22.0,
    ):
        """Initialise PrecomputedZodi.

        Args:
            wavelengths_nm: Array of wavelengths in nm.
            flux_phot_per_arcsec2: Array of photon flux values in
                ph/s/m^2/nm per arcsec^2 (same length as
                ``wavelengths_nm``).
            reference_mag_arcsec2: Reference magnitude carried as metadata
                only; the actual flux is taken from ``flux_phot_per_arcsec2``.
        """
        self._wavelengths_nm = jnp.asarray(wavelengths_nm)
        self._flux_density_phot = jnp.asarray(flux_phot_per_arcsec2)
        self._reference_wavelength_nm = 550.0
        self._reference_mag_arcsec2 = reference_mag_arcsec2
        self._flux_interp = interpax.Interpolator1D(
            self._wavelengths_nm, self._flux_density_phot, method="linear"
        )

    @property
    def reference_wavelength_nm(self) -> float:
        """Reference wavelength for the zodi model in nm."""
        return self._reference_wavelength_nm

    @property
    def reference_mag_arcsec2(self) -> float:
        """Surface brightness at the reference wavelength in mag/arcsec^2."""
        return self._reference_mag_arcsec2

    def spec_flux_density(
        self,
        wavelength_nm: float,
        time_jd: float,
        ecliptic_lat_deg: float = 0.0,
        solar_lon_deg: float = 135.0,
    ) -> float:
        """Return surface brightness in ph/s/m^2/nm per arcsec^2.

        All arguments other than ``wavelength_nm`` are accepted for
        interface compatibility but ignored.
        """
        return self._flux_interp(wavelength_nm)

    def __repr__(self) -> str:
        """One-line summary of pre-tabulated zodi photon-flux source."""
        wl_min = float(self._wavelengths_nm.min())
        wl_max = float(self._wavelengths_nm.max())
        return (
            f"PrecomputedZodi(ref_mag={self._reference_mag_arcsec2:.2f}/arcsec^2, "
            f"wl={wl_min:.0f}-{wl_max:.0f} nm, "
            f"n_wl={int(self._wavelengths_nm.size)}; pre-tabulated)"
        )
