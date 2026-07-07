"""Zodiacal light model based on Leinert et al. (1998).

All functions are pure JAX and JIT-compatible.  Provides both position-dependent
(Leinert Table 17) and wavelength-dependent (Table 19) zodiacal brightness.

Migrated from orbix.observatory.zodiacal -- see that module's history for
full references.
"""

from __future__ import annotations

import interpax
import jax.numpy as jnp
from hwoutils.constants import c
from hwoutils.conversions import flux_jy_to_mag, mag_to_flux_jy

# =============================================================================
# Constants
# =============================================================================

# AYO default: 22 mag/arcsec^2 at V-band (ETC calibration paper, 135 deg solar lon)
AYO_DEFAULT_ZODI_MAG_V = 22.0
V_BAND_WAVELENGTH_NM = 550.0

# =============================================================================
# Leinert et al. (1998) tables
# =============================================================================

# Table 17: Zodiacal brightness in S10 units
# Rows: solar longitude, Columns: ecliptic latitude
LEINERT_BETA_DEG = jnp.array([0.0, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90])
LEINERT_SOLAR_LON_DEG = jnp.array(
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
)

LEINERT_TABLE17 = jnp.array(
    [
        [-1, -1, -1, 3140, 1610, 985, 640, 275, 150, 100, 77],
        [-1, -1, -1, 2940, 1540, 945, 625, 271, 150, 100, 77],
        [-1, -1, 4740, 2470, 1370, 865, 590, 264, 148, 100, 77],
        [11500, 6780, 3440, 1860, 1110, 755, 525, 251, 146, 100, 77],
        [6400, 4480, 2410, 1410, 910, 635, 454, 237, 141, 99, 77],
        [3840, 2830, 1730, 1100, 749, 545, 410, 223, 136, 97, 77],
        [2480, 1870, 1220, 845, 615, 467, 365, 207, 131, 95, 77],
        [1650, 1270, 910, 680, 510, 397, 320, 193, 125, 93, 77],
        [1180, 940, 700, 530, 416, 338, 282, 179, 120, 92, 77],
        [910, 730, 555, 442, 356, 292, 250, 166, 116, 90, 77],
        [505, 442, 352, 292, 243, 209, 183, 134, 104, 86, 77],
        [338, 317, 269, 227, 196, 172, 151, 116, 93, 82, 77],
        [259, 251, 225, 193, 166, 147, 132, 104, 86, 79, 77],
        [212, 210, 197, 170, 150, 133, 119, 96, 82, 77, 77],
        [188, 186, 177, 154, 138, 125, 113, 90, 77, 74, 77],
        [179, 178, 166, 147, 134, 122, 110, 90, 77, 73, 77],
        [179, 178, 165, 148, 137, 127, 116, 96, 79, 72, 77],
        [196, 192, 179, 165, 151, 141, 131, 104, 82, 72, 77],
        [230, 212, 195, 178, 163, 148, 134, 105, 83, 72, 77],
    ],
)

# Table 19: Wavelength dependence (spectral radiance at 90 deg solar elongation)
LEINERT_WAVELENGTH_UM = jnp.array(
    [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 2.2, 3.5, 4.8, 12, 25, 60, 100, 140]
)

LEINERT_B_LAMBDA = jnp.array(
    [
        2.5e-8,
        5.3e-7,
        2.2e-6,
        2.6e-6,
        2.0e-6,
        1.3e-6,
        1.2e-6,
        8.1e-7,
        1.7e-7,
        5.2e-8,
        1.2e-7,
        7.5e-7,
        3.2e-7,
        1.8e-8,
        3.2e-9,
        6.9e-10,
    ]
)

_LOG_WAVELENGTH_UM = jnp.log10(LEINERT_WAVELENGTH_UM)
_LOG_B_LAMBDA = jnp.log10(LEINERT_B_LAMBDA)


def _clamped_log10_um(wavelength_nm):
    """Log10(wavelength in um), clamped to the tabulated Table 19 range.

    interpax.interp1d defaults to extrap=False (NaN outside the knot range);
    clamping the query wavelength to the table's domain before interpolating
    means a wavelength outside Table 19's 0.2-140 um span resolves to the
    nearest tabulated edge value instead of NaN.
    """
    log_um = jnp.log10(wavelength_nm / 1000.0)
    return jnp.clip(log_um, _LOG_WAVELENGTH_UM[0], _LOG_WAVELENGTH_UM[-1])


def _fill_sentinels(table):
    """Fill near-Sun exclusion cells with the nearest valid value.

    Sentinel cells (-1) are filled along the elongation axis so lookups
    clamp instead of returning NaN.
    """
    import numpy as np

    t = np.asarray(table, dtype=float)
    for row in t:
        valid = np.where(row > 0)[0]
        for j in np.where(row <= 0)[0]:
            row[j] = row[valid[np.argmin(np.abs(valid - j))]]
    return jnp.asarray(t)


_LEINERT_TABLE17_FILLED = _fill_sentinels(LEINERT_TABLE17)

_REF_IDX = jnp.argmin(jnp.abs(LEINERT_SOLAR_LON_DEG - 90.0))
_REFERENCE_VALUE = LEINERT_TABLE17[_REF_IDX, 0]
_NORMALIZED_TABLE17 = _LEINERT_TABLE17_FILLED / _REFERENCE_VALUE


# =============================================================================
# Core functions
# =============================================================================


def zodi_color_correction(
    wavelength_nm: float,
    reference_wavelength_nm: float = V_BAND_WAVELENGTH_NM,
    photon_units: bool = True,
) -> float:
    """Wavelength-dependent color correction from Leinert Table 19.

    Args:
        wavelength_nm: Target wavelength in nm.
        reference_wavelength_nm: Reference wavelength in nm.
        photon_units: If True, include lambda/lambda_ref factor for photon flux.

    Returns:
        Flux ratio (target / reference).
    """
    target_log = interpax.interp1d(
        _clamped_log10_um(wavelength_nm),
        _LOG_WAVELENGTH_UM,
        _LOG_B_LAMBDA,
        method="linear",
    )
    ref_log = interpax.interp1d(
        _clamped_log10_um(reference_wavelength_nm),
        _LOG_WAVELENGTH_UM,
        _LOG_B_LAMBDA,
        method="linear",
    )
    power_correction = 10.0 ** (target_log - ref_log)
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
    factor = interpax.interp2d(
        solar_lon_deg,
        jnp.abs(ecliptic_lat_deg),
        LEINERT_SOLAR_LON_DEG,
        LEINERT_BETA_DEG,
        _NORMALIZED_TABLE17,
        method="linear",
    )
    return factor


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
    log_radiance = interpax.interp1d(
        _clamped_log10_um(wavelength_nm),
        _LOG_WAVELENGTH_UM,
        _LOG_B_LAMBDA,
        method="linear",
    )
    base_radiance = 10.0**log_radiance
    position_factor = leinert_zodi_factor(ecliptic_lat_deg, solar_lon_deg)
    return base_radiance * position_factor


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
        Surface brightness in mag/arcsec^2.
    """
    radiance = leinert_zodi_spectral_radiance(
        wavelength_nm, ecliptic_lat_deg, solar_lon_deg
    )
    arcsec2_per_sr = (180.0 / jnp.pi * 3600.0) ** 2
    radiance_per_arcsec2 = radiance / arcsec2_per_sr
    wavelength_um = wavelength_nm / 1000.0
    c_um_per_s = c * 1e6  # m/s -> um/s
    flux_per_hz = radiance_per_arcsec2 * (wavelength_um**2) / c_um_per_s
    flux_jy = flux_per_hz * 1e26
    return flux_jy_to_mag(flux_jy)


def ayo_default_zodi_mag(wavelength_nm: float) -> float:
    """AYO-default zodiacal brightness: 22 mag/arcsec^2 at V with color correction.

    Args:
        wavelength_nm: Observation wavelength in nm.

    Returns:
        Surface brightness in mag/arcsec^2.
    """
    color_correction = zodi_color_correction(
        wavelength_nm,
        V_BAND_WAVELENGTH_NM,
        photon_units=False,
    )
    return AYO_DEFAULT_ZODI_MAG_V - 2.5 * jnp.log10(color_correction)


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

    target_log = interpax.interp1d(
        _clamped_log10_um(wavelengths_nm),
        _LOG_WAVELENGTH_UM,
        _LOG_B_LAMBDA,
        method="linear",
    )
    ref_log = interpax.interp1d(
        _clamped_log10_um(reference_wavelength_nm),
        _LOG_WAVELENGTH_UM,
        _LOG_B_LAMBDA,
        method="linear",
    )

    power_corrections = 10.0 ** (target_log - ref_log)
    jy_corrections = power_corrections * (wavelengths_nm / reference_wavelength_nm) ** 2
    return reference_flux_jy * jy_corrections
