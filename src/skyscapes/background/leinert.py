"""Zodiacal light model based on Leinert et al. (1998).

All functions are pure JAX and JIT-compatible. The tables, interpolation and
color conventions live in the ``zodi`` library; this module keeps the
skyscapes function names and argument order as thin wrappers over it.
Provides both position-dependent (Leinert Table 17) and wavelength-dependent
(Table 19) zodiacal brightness.
"""

from __future__ import annotations

import jax.numpy as jnp
import zodi
from hwoutils.conversions import mag_to_flux_jy
from zodi import tables

# =============================================================================
# Constants
# =============================================================================

# AYO default: 22 mag/arcsec^2 at V-band (ETC calibration paper, 135 deg solar lon)
AYO_DEFAULT_ZODI_MAG_V = zodi.AYO_DEFAULT_ZODI_MAG_V
V_BAND_WAVELENGTH_NM = zodi.V_BAND_WAVELENGTH_NM

# =============================================================================
# Leinert et al. (1998) tables, as JAX arrays
# =============================================================================

# Table 17 in S10 units; rows are solar longitude, columns ecliptic latitude.
# Near-Sun exclusion cells carry the sentinel -1 here (nan in ``zodi.tables``).
LEINERT_BETA_DEG = jnp.asarray(tables.BETA_GRID_DEG)
LEINERT_SOLAR_LON_DEG = jnp.asarray(tables.LON_GRID_DEG)
LEINERT_TABLE17 = jnp.where(
    jnp.isnan(jnp.asarray(tables.TABLE17_RAW)), -1.0, jnp.asarray(tables.TABLE17_RAW)
)

# Table 19: spectral radiance at 90 deg solar elongation (W m-2 sr-1 um-1).
LEINERT_WAVELENGTH_UM = jnp.asarray(tables.TABLE19_WAVELENGTH_UM)
LEINERT_B_LAMBDA = jnp.asarray(tables.TABLE19_RADIANCE_W_M2_SR_UM)


# =============================================================================
# Core functions
# =============================================================================


def zodi_color_correction(
    wavelength_nm: float,
    reference_wavelength_nm: float = V_BAND_WAVELENGTH_NM,
    photon_units: bool = True,
) -> float:
    """Wavelength-dependent color correction from Leinert Table 19.

    This is the ratio of spectral radiance per unit wavelength (or per
    photon, with ``photon_units``). It is not an AB (per unit frequency)
    ratio; see :func:`zodi.ab_color_correction` for that.

    Args:
        wavelength_nm: Target wavelength in nm.
        reference_wavelength_nm: Reference wavelength in nm.
        photon_units: If True, include lambda/lambda_ref factor for photon flux.

    Returns:
        Flux ratio (target / reference).
    """
    wavelength_nm = jnp.asarray(wavelength_nm)
    reference_wavelength_nm = jnp.asarray(reference_wavelength_nm)
    power_correction = zodi.color_correction(wavelength_nm, reference_wavelength_nm)
    photon_factor = jnp.where(
        photon_units,
        wavelength_nm / reference_wavelength_nm,
        1.0,
    )
    return power_correction * photon_factor


def leinert_zodi_factor(
    ecliptic_lat_deg: float,
    solar_lon_deg: float = 135.0,
) -> float:
    """Position-dependent zodiacal brightness factor from Leinert Table 17.

    Returns brightness relative to (solar_lon=90 deg, ecliptic_lat=0 deg).

    Queries inside the near-Sun exclusion zone return the nearest tabulated
    brightness (clamped), not NaN.

    Args:
        ecliptic_lat_deg: Ecliptic latitude in degrees.
        solar_lon_deg: Solar longitude in degrees (default 135 deg for coronagraphs).

    Returns:
        Dimensionless brightness factor.
    """
    return zodi.position_factor(
        jnp.asarray(solar_lon_deg), jnp.asarray(ecliptic_lat_deg)
    )


def leinert_zodi_spectral_radiance(
    wavelength_nm: float,
    ecliptic_lat_deg: float = 0.0,
    solar_lon_deg: float = 135.0,
) -> float:
    """Zodiacal spectral radiance in W/(m^2 sr um) from Leinert tables.

    Args:
        wavelength_nm: Observation wavelength in nm.
        ecliptic_lat_deg: Ecliptic latitude in degrees.
        solar_lon_deg: Solar longitude in degrees.

    Returns:
        Spectral radiance in W/(m^2 sr um).
    """
    return zodi.specific_intensity(
        jnp.asarray(solar_lon_deg),
        jnp.asarray(ecliptic_lat_deg),
        jnp.asarray(wavelength_nm),
        anchor="table19",
    )


def leinert_zodi_mag(
    wavelength_nm: float,
    ecliptic_lat_deg: float = 0.0,
    solar_lon_deg: float = 135.0,
) -> float:
    """Zodiacal surface brightness in mag/arcsec^2 from Leinert tables.

    Args:
        wavelength_nm: Observation wavelength in nm.
        ecliptic_lat_deg: Ecliptic latitude in degrees.
        solar_lon_deg: Solar longitude in degrees.

    Returns:
        Surface brightness in mag/arcsec^2 (AB).
    """
    return zodi.surface_brightness_ab_mag(
        jnp.asarray(solar_lon_deg),
        jnp.asarray(ecliptic_lat_deg),
        jnp.asarray(wavelength_nm),
        anchor="table19",
    )


def ayo_default_zodi_mag(wavelength_nm: float) -> float:
    """AYO-default zodiacal brightness: 22 mag/arcsec^2 at V with color correction.

    The color is the Leinert Table 19 ratio applied in AB (per unit
    frequency), the same color as :func:`leinert_zodi_mag`.

    Args:
        wavelength_nm: Observation wavelength in nm.

    Returns:
        Surface brightness in mag/arcsec^2.
    """
    return zodi.ayo_default_ab_mag(jnp.asarray(wavelength_nm))


def ayo_default_zodi_flux_jy(wavelength_nm: float) -> float:
    """AYO-default zodiacal brightness in Jy/arcsec^2.

    The AYO 22 mag V-band figure is treated as AB and converted via
    hwoutils.conversions.mag_to_flux_jy (astropy-exact AB zero point). If a
    Johnson-V calibration is ever required, the AB<->Johnson-V offset is
    ~0.044 mag.

    Args:
        wavelength_nm: Observation wavelength in nm.

    Returns:
        Surface brightness in Jy/arcsec^2.
    """
    return mag_to_flux_jy(ayo_default_zodi_mag(wavelength_nm))


def create_zodi_spectrum_jax(
    wavelengths_nm: jnp.ndarray,
    reference_flux_jy: float | None = None,
    reference_wavelength_nm: float = V_BAND_WAVELENGTH_NM,
) -> jnp.ndarray:
    """Create zodiacal light spectrum in Jy/arcsec^2 from reference flux.

    Args:
        wavelengths_nm: Array of wavelengths in nm.
        reference_flux_jy: Reference flux (default: 22 mag at V).
        reference_wavelength_nm: Reference wavelength in nm.

    Returns:
        Array of surface brightness in Jy/arcsec^2.
    """
    if reference_flux_jy is None:
        reference_flux_jy = mag_to_flux_jy(AYO_DEFAULT_ZODI_MAG_V)
    return reference_flux_jy * zodi.ab_color_correction(
        jnp.asarray(wavelengths_nm), jnp.asarray(reference_wavelength_nm)
    )
