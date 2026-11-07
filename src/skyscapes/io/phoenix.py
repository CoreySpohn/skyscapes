"""PHOENIX model-atmosphere spectra from the STScI reference-atlas grid.

The grid is the solar-metallicity ``phoenixm00`` BT-Settl set distributed with
the STScI reference atlases (``cdbs``). Each file holds one effective
temperature and every tabulated surface gravity, on a common wavelength grid
of 103000 log-spaced samples from 1.3 nm to 77.6 micron (R of order 2400 near
4.5 micron). The file headers record the units: wavelength in ``ANGSTROM`` and
flux in ``FLAM`` (erg/cm^2/s/Angstrom), the emergent flux at the stellar
surface. The loader converts both through the header unit strings, so a file
in other units fails loudly instead of being misread.

Interpolation. Between Teff nodes and between ``logg`` nodes the loader
interpolates the logarithm of the flux linearly (bilinear in Teff and
``logg``). The flux falls exponentially with wavelength on the Wien side and
the band depths of molecular absorbers change smoothly in log flux with
temperature, so a linear blend in flux over-weights the hotter model by large
factors in the ultraviolet where the node spacing is 100 K. At a node the
weight is exactly zero for the neighbor, so the node spectrum is returned
unchanged.
"""

from collections.abc import Callable

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from hwoutils.constants import Jy, Rsun2m, c, nm2m, pc2m, rad2arcsec

from ..datasets import PHOENIX_TEFF_K, fetch_phoenix
from ..scene.star import Star

# Header unit strings the reader accepts, mapped to astropy units.
_WAVELENGTH_UNITS = {"ANGSTROM": u.AA}
_FLUX_UNITS = {"FLAM": u.erg / u.cm**2 / u.s / u.AA}
_SURFACE_FLUX_UNIT = u.W / u.m**2 / u.nm

# IAU 2015 Resolution B3 nominal solar effective temperature, K (hwoutils
# carries the nominal radius but not this value).
_TEFF_SUN_K = 5772.0


def _read_model(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Read one PHOENIX FITS file in SI-derived units.

    Args:
        path: Path to a ``phoenixm00_<Teff>.fits`` file.

    Returns:
        ``(wavelength_nm, logg, flux_w_m2_nm, teff_k)`` where ``flux_w_m2_nm``
        has shape ``(n_logg, n_wavelength)`` and is the surface flux density.
    """
    with fits.open(path) as hdul:
        teff_k = float(hdul[0].header["TEFF"])
        table = hdul[1].data
        header = hdul[1].header
        columns = [name for name in table.columns.names if name != "WAVELENGTH"]

        wl_unit_key = str(header["TUNIT1"]).strip().upper()
        if wl_unit_key not in _WAVELENGTH_UNITS:
            raise ValueError(f"{path}: unsupported wavelength unit {wl_unit_key!r}")
        wavelength_nm = (
            np.asarray(table["WAVELENGTH"], dtype=float)
            * _WAVELENGTH_UNITS[wl_unit_key]
        ).to_value(u.nm)

        logg = np.array([int(name[1:]) / 10.0 for name in columns])
        flux = np.empty((len(columns), wavelength_nm.size))
        for i, name in enumerate(columns):
            unit_key = str(header[f"TUNIT{table.columns.names.index(name) + 1}"])
            unit_key = unit_key.strip().upper()
            if unit_key not in _FLUX_UNITS:
                raise ValueError(f"{path}: unsupported flux unit {unit_key!r}")
            flux[i] = (
                np.asarray(table[name], dtype=float) * _FLUX_UNITS[unit_key]
            ).to_value(_SURFACE_FLUX_UNIT)
    return wavelength_nm, logg, flux, teff_k


def _bracket(value: float, nodes: np.ndarray, name: str) -> tuple[int, int, float]:
    """Indices of the bracketing nodes and the weight of the upper one."""
    if not nodes[0] <= value <= nodes[-1]:
        raise ValueError(
            f"{name} = {value} is outside the grid range {nodes[0]} to {nodes[-1]}."
        )
    hi = int(np.searchsorted(nodes, value))
    if nodes[hi] == value:
        return hi, hi, 0.0
    lo = hi - 1
    return lo, hi, float((value - nodes[lo]) / (nodes[hi] - nodes[lo]))


def phoenix_surface_flux(
    teff_k: float,
    logg: float,
    mh: float = 0.0,
    *,
    fetch: Callable[[int], str] = fetch_phoenix,
) -> tuple[np.ndarray, np.ndarray]:
    """Surface flux density of a PHOENIX model at the requested parameters.

    Interpolates log flux bilinearly in ``teff_k`` and ``logg`` (module
    docstring). The result is the emergent flux per unit stellar surface
    area; multiply by ``(radius / distance)**2`` for the flux at the observer.

    Args:
        teff_k: Effective temperature in kelvin, within the registered grid.
        logg: Surface gravity, log10 of cgs gravity in cm/s^2.
        mh: Metallicity [M/H] in dex. Only 0.0 is registered.
        fetch: Callable mapping a node Teff in kelvin to a local FITS path.
            The default downloads and caches through pooch.

    Returns:
        ``(wavelength_nm, flux_w_m2_nm)``: wavelength in nm and surface flux
        density in W/m^2/nm, both float64 numpy arrays on the model grid.

    Raises:
        ValueError: If ``mh`` is not 0.0 or a parameter is outside the grid.
    """
    if mh != 0.0:
        raise ValueError("Only solar metallicity ([M/H] = 0) is registered.")
    nodes = np.asarray(PHOENIX_TEFF_K, dtype=float)
    lo, hi, t_weight = _bracket(float(teff_k), nodes, "teff_k")

    # Each entry is (weight, log10 flux or the node flux itself for an exact
    # logg node); a lone weight-one entry returns the node bytes unchanged.
    parts = []
    for index, weight in ((lo, 1.0 - t_weight), (hi, t_weight)):
        if weight == 0.0:
            continue
        wavelength_nm, logg_nodes, flux, _ = _read_model(fetch(int(nodes[index])))
        g_lo, g_hi, g_weight = _bracket(float(logg), logg_nodes, "logg")
        if g_weight == 0.0:
            parts.append((weight, flux[g_lo]))
        else:
            log_flux = (1.0 - g_weight) * np.log10(flux[g_lo])
            log_flux = log_flux + g_weight * np.log10(flux[g_hi])
            parts.append((weight, 10.0**log_flux))
    if len(parts) == 1:
        return wavelength_nm, parts[0][1]
    log_blend = sum(weight * np.log10(flux) for weight, flux in parts)
    return wavelength_nm, 10.0**log_blend


def from_phoenix(
    teff_k: float,
    logg: float,
    *,
    Ms_kg: float,
    dist_pc: float,
    radius_m: float,
    mh: float = 0.0,
    wavelength_range_nm: tuple[float, float] | None = (100.0, 30000.0),
    ra_deg: float = 0.0,
    dec_deg: float = 0.0,
    fetch: Callable[[int], str] = fetch_phoenix,
) -> Star:
    """Build a static-spectrum ``Star`` from a PHOENIX model atmosphere.

    The surface flux is diluted by ``(radius_m / distance)**2``, the same
    geometric normalization as ``BlackbodyStar``, converted to Jy and handed
    to ``Star`` with linear wavelength interpolation (cubic splines overshoot
    across molecular band edges). The angular diameter is ``2 R / d`` and the
    luminosity is ``4 pi R^2 sigma Teff^4`` in solar units.

    Args:
        teff_k: Effective temperature in kelvin.
        logg: Surface gravity, log10 of cgs gravity in cm/s^2.
        Ms_kg: Stellar mass in kilograms.
        dist_pc: Distance in parsecs.
        radius_m: Stellar radius in meters.
        mh: Metallicity [M/H] in dex. Only 0.0 is registered.
        wavelength_range_nm: Crop of the model grid in nm, or ``None`` for the
            full 1.3 nm to 77.6 micron range.
        ra_deg: Right ascension in degrees, carried on the ``Star``.
        dec_deg: Declination in degrees, carried on the ``Star``.
        fetch: Callable mapping a node Teff in kelvin to a local FITS path.

    Returns:
        A ``Star`` whose ``spec_flux_density`` is the observer flux in
        ph/s/m^2/nm.
    """
    wavelength_nm, surface_flux = phoenix_surface_flux(teff_k, logg, mh, fetch=fetch)
    if wavelength_range_nm is not None:
        keep = (wavelength_nm >= wavelength_range_nm[0]) & (
            wavelength_nm <= wavelength_range_nm[1]
        )
        wavelength_nm, surface_flux = wavelength_nm[keep], surface_flux[keep]

    dilution = (radius_m / (dist_pc * pc2m)) ** 2
    # F_nu = F_lambda lambda^2 / c, with F_lambda converted from per nm to per m.
    flux_jy = surface_flux / nm2m * (wavelength_nm * nm2m) ** 2 / c * dilution / Jy
    return Star(
        Ms_kg=Ms_kg,
        dist_pc=dist_pc,
        wavelengths_nm=jnp.asarray(wavelength_nm),
        times_jd=jnp.zeros(1),
        flux_density_jy=jnp.asarray(flux_jy)[:, None],
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        diameter_arcsec=2.0 * radius_m / (dist_pc * pc2m) * rad2arcsec,
        luminosity_lsun=(radius_m / Rsun2m) ** 2 * (teff_k / _TEFF_SUN_K) ** 4,
        metadata={
            "source": "PHOENIX BT-Settl (STScI cdbs phoenixm00)",
            "teff_k": float(teff_k),
            "logg": float(logg),
            "mh": float(mh),
        },
        interpolation_method="linear",
    )
