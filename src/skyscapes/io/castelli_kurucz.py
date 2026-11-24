"""Castelli and Kurucz (2004) ATLAS9 atmospheres from STScI cdbs.

Reference: Castelli, F. and Kurucz, R. L. (2004), New Grids of ATLAS9
Model Atmospheres, arXiv:astro-ph/0405087. Solar-metallicity ckp00 files
use ANGSTROM wavelengths and FLAM surface fluxes. Gravity column g40
means log10(g / (cm/s^2)) = 4.0. All-zero columns are unavailable models,
not zero-luminosity stars, and are excluded from interpolation.
"""

from collections.abc import Callable

import numpy as np

from ..datasets import CASTELLI_KURUCZ_TEFF_K, fetch_castelli_kurucz
from ..scene.star import Star
from .phoenix import _from_surface_flux, _surface_flux


def castelli_kurucz_surface_flux(
    teff_K: float,
    logg: float,
    mh: float = 0.0,
    *,
    fetch: Callable[[int], str] = fetch_castelli_kurucz,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavelength [nm] and ATLAS9 surface flux [W/m^2/nm].

    Solar metallicity only. Interpolate log flux in temperature and gravity
    as for PHOENIX. Temperature spans 3500-50000 K; available gravity ranges
    depend on temperature. Both bracketing files must support the requested
    gravity. ``fetch`` maps a temperature node [K] to its local FITS path.

    Args:
        teff_K: Effective temperature [K], from 3500 to 50000.
        logg: Surface gravity, log10 of the cgs gravity in cm/s^2.
        mh: Metallicity [M/H] in dex. Only 0.0 is registered.
        fetch: Callable mapping a temperature node [K] to a local FITS path.

    Returns:
        ``(wavelength_nm, flux)``: wavelength [nm] and surface flux density
        [W m^-2 nm^-1], both 1-D arrays on the model grid.
    """
    if mh != 0.0:
        raise ValueError("Only solar metallicity ([M/H] = 0) is registered.")
    return _surface_flux(teff_K, logg, CASTELLI_KURUCZ_TEFF_K, fetch)


def from_castelli_kurucz(
    teff_K: float,
    logg: float,
    *,
    dist_pc: float,
    radius_m: float,
    mh: float = 0.0,
    wavelength_range_nm: tuple[float, float] | None = (100.0, 30000.0),
    fetch: Callable[[int], str] = fetch_castelli_kurucz,
) -> Star:
    """Build a static linear-spectrum Star from solar-metallicity ATLAS9.

    Parameters and units match ``from_phoenix``. ``teff_K`` spans
    3500-50000 K; ``logg`` is log10 of cgs gravity. The surface flux is
    diluted by ``(radius_m / (dist_pc * pc2m))**2``. The returned Star
    evaluates photon flux in ph/s/m^2/nm and supports exact linear-table
    bin means. Downloads are lazy, cached, and SHA-256 verified.

    Args:
        teff_K: Effective temperature [K], from 3500 to 50000.
        logg: Surface gravity, log10 of the cgs gravity in cm/s^2.
        dist_pc: Distance of the star from the observer [pc].
        radius_m: Stellar radius [m].
        mh: Metallicity [M/H] in dex. Only 0.0 is registered.
        wavelength_range_nm: ``(min, max)`` crop of the model grid [nm], or
            ``None`` to keep the full grid.
        fetch: Callable mapping a temperature node [K] to a local FITS path.

    Returns:
        Static ``Star`` whose photon flux density at the observer is in
        ph s^-1 m^-2 nm^-1.
    """
    wavelength_nm, surface_flux = castelli_kurucz_surface_flux(
        teff_K, logg, mh, fetch=fetch
    )
    return _from_surface_flux(
        wavelength_nm,
        surface_flux,
        teff_K=teff_K,
        logg=logg,
        mh=mh,
        dist_pc=dist_pc,
        radius_m=radius_m,
        wavelength_range_nm=wavelength_range_nm,
    )
