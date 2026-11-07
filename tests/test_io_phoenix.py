"""PHOENIX model-atmosphere loader: units, normalization, interpolation, bands.

The offline tests read four real STScI cdbs nodes (3500, 3600, 5800 and 5900 K)
stored in ``tests/data``: block-averaged by 25 samples outside 4.0 to 5.0
micron, native sampling inside, ``logg`` 4.5 and 5.0 only. Block averaging
conserves the integrated flux. The slow test reads the full-resolution file
from the archive; a cold cache without network skips with a reason, and
``SKYSCAPES_REQUIRE_DATA=1`` turns that skip into a failure.
"""

import os
from pathlib import Path

import equinox as eqx
import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rsun2m, h, k_B, pc2m, sigma_SB
from hwoutils.constants import c as c_si

from skyscapes.datasets import fetch_phoenix
from skyscapes.io import from_phoenix, phoenix_surface_flux
from skyscapes.io.phoenix import _read_model

DATA = Path(__file__).parent / "data"
SIGMA_SB_TEST = 5.670374419e-8  # typed once here; parity with hwoutils asserted below
REQUIRE_DATA = os.environ.get("SKYSCAPES_REQUIRE_DATA") == "1"


def fixture_path(teff_k: int) -> str:
    return str(DATA / f"phoenixm00_{teff_k}_test.fits")


def photon_energy_flux_w_m2(star, wavelength_nm):
    """Integral of photon flux density times hc/lambda over wavelength."""
    photons = np.asarray(star.spec_flux_density(wavelength_nm, jnp.zeros(())))
    energy_w_m2_nm = photons * h * c_si / (wavelength_nm * 1e-9)
    return np.trapezoid(energy_w_m2_nm, wavelength_nm)


def band_mean(wavelength_nm, values, low_nm, high_nm):
    keep = (wavelength_nm >= low_nm) & (wavelength_nm <= high_nm)
    return values[keep].mean()


def planck_lambda(wavelength_nm, teff_k):
    """Shape of the Planck spectral radiance per wavelength (arbitrary scale)."""
    x = h * c_si / (wavelength_nm * 1e-9 * k_B * teff_k)
    return 1.0 / (wavelength_nm**5 * np.expm1(x))


def co_band_depth_of(wavelength_nm, flux, teff_k):
    """Fractional CO depth of 4.55-4.95 micron relative to 3.95-4.25 micron.

    Both windows are divided by a same-Teff Planck shape first, so a blackbody
    has depth exactly zero.
    """
    keep = (wavelength_nm >= 3900.0) & (wavelength_nm <= 5000.0)
    wavelength_nm = wavelength_nm[keep]
    ratio = flux[keep] / planck_lambda(wavelength_nm, teff_k)
    band = band_mean(wavelength_nm, ratio, 4550.0, 4950.0)
    continuum = band_mean(wavelength_nm, ratio, 3950.0, 4250.0)
    return 1.0 - band / continuum


def co_band_depth(teff_k, logg):
    """CO depth of a fixture model."""
    wavelength_nm, flux = phoenix_surface_flux(teff_k, logg, fetch=fixture_fetch)
    return co_band_depth_of(wavelength_nm, flux, teff_k)


def fixture_fetch(teff_k):
    return fixture_path(teff_k)


def test_sigma_parity():
    """Fails if the typed Stefan-Boltzmann constant drifts from hwoutils."""
    np.testing.assert_allclose(SIGMA_SB_TEST, sigma_SB, rtol=1e-9)


@vv.case("stellar-photometry", "code-verification")
@pytest.mark.parametrize(("teff_k", "logg"), [(3500, 5.0), (3600, 5.0), (5800, 4.5)])
def test_surface_flux_integrates_to_sigma_teff4(teff_k, logg):
    """Integrated surface flux equals sigma Teff^4 within 5 percent.

    Truth: the Stefan-Boltzmann law for a model atmosphere that conserves flux.
    Basis: spike. The full-resolution files integrate to 0.994 (3500 K), 0.985
    (3600 K) and 0.970 (5800 K) of sigma Teff^4 over 1.3 nm to 77.6 micron, a
    shortfall of the model flux convergence. The block-averaged copies agree
    to 1e-5. A wrong erg-to-joule, Angstrom-to-nm or cm^2-to-m^2 factor moves
    the integral by 10^3 or more, and a missing pi by 3.

    Fails if the header unit conversion of wavelength or flux is wrong.
    """
    wavelength_nm, flux = phoenix_surface_flux(teff_k, logg, fetch=fixture_fetch)
    bolometric = np.trapezoid(flux, wavelength_nm)
    np.testing.assert_allclose(bolometric, SIGMA_SB_TEST * teff_k**4, rtol=0.05)


@vv.seam("skyscapes", "coronagraphoto")
@vv.case("stellar-photometry", "code-verification")
def test_star_photon_flux_energy_matches_dilution():
    """Energy carried by the Star photon flux times 4 pi d^2 is 4 pi R^2 sigma T^4.

    Truth: Stefan-Boltzmann luminosity of a 0.4 Rsun, 3500 K dwarf at 5 pc.
    Tolerance 5 percent (basis: spike, the 0.6 percent flux shortfall of the
    3500 K model plus trapezoid error on the block-averaged fixture).

    Fails if the surface flux is not diluted by (R / d)^2, or if the Jy and
    photon conversions between the model and ``Star`` are off by a unit.
    """
    teff_k, radius_m, dist_pc = 3500.0, 0.4 * Rsun2m, 5.0
    star = from_phoenix(
        teff_k,
        5.0,
        Ms_kg=1.0e30,
        dist_pc=dist_pc,
        radius_m=radius_m,
        wavelength_range_nm=None,
        fetch=fixture_fetch,
    )
    wavelength_nm, _ = phoenix_surface_flux(teff_k, 5.0, fetch=fixture_fetch)
    energy = photon_energy_flux_w_m2(star, wavelength_nm)
    expected = SIGMA_SB_TEST * teff_k**4 * (radius_m / (dist_pc * pc2m)) ** 2
    np.testing.assert_allclose(energy, expected, rtol=0.05)
    np.testing.assert_allclose(
        star.diameter_arcsec,
        np.degrees(2 * radius_m / (dist_pc * pc2m)) * 3600.0,
        rtol=1e-6,
    )
    assert star.luminosity_lsun == pytest.approx(
        (radius_m / Rsun2m) ** 2 * (teff_k / 5772.0) ** 4
    )


@vv.case("stellar-photometry", "code-verification")
def test_interpolation_node_and_midpoint():
    """A node returns the node; a midpoint is the geometric mean of the nodes.

    Basis: exact-algebra (log flux is linear in the weights; tolerance is
    floating-point, rtol 1e-12). The logg midpoint is checked on one node
    because the 3500 and 3600 K files both tabulate 4.5 and 5.0.

    Fails if a node picks up weight from its neighbor, or if the blend is
    linear in flux instead of log flux.
    """
    wl, node = phoenix_surface_flux(3500, 5.0, fetch=fixture_fetch)
    _, logg_nodes, flux_nodes, _ = _read_model(fixture_path(3500))
    np.testing.assert_array_equal(node, flux_nodes[list(logg_nodes).index(5.0)])

    _, upper = phoenix_surface_flux(3600, 5.0, fetch=fixture_fetch)
    _, mid = phoenix_surface_flux(3550, 5.0, fetch=fixture_fetch)
    np.testing.assert_allclose(mid, np.sqrt(node * upper), rtol=1e-12)

    _, low_g = phoenix_surface_flux(3500, 4.5, fetch=fixture_fetch)
    _, mid_g = phoenix_surface_flux(3500, 4.75, fetch=fixture_fetch)
    np.testing.assert_allclose(mid_g, np.sqrt(low_g * node), rtol=1e-12)
    assert wl.size == node.size


def test_out_of_grid_requests_raise():
    """Teff or logg outside the grid and mh != 0 raise instead of clamping.

    Fails if a request outside the registered grid returns the edge model.
    """
    with pytest.raises(ValueError, match="teff_k"):
        phoenix_surface_flux(1500.0, 5.0, fetch=fixture_fetch)
    with pytest.raises(ValueError, match="logg"):
        phoenix_surface_flux(3500.0, 3.0, fetch=fixture_fetch)
    with pytest.raises(ValueError, match="metallicity"):
        phoenix_surface_flux(3500.0, 5.0, mh=-0.5, fetch=fixture_fetch)


@vv.case("stellar-photometry", "code-verification")
def test_co_band_depth_cool_dwarf():
    """The 3500 K dwarf has a CO fundamental depth of 12 percent, twice a Sun's.

    Depth: 1 - (mean of F / Planck over 4.55-4.95 micron) / (mean over
    3.95-4.25 micron), logg 5.0. Expected 0.122 +/- 0.02 (basis: spike on the
    cdbs 3500 K file, 0.1224 full resolution); the 5800 K, logg 4.5 model gives
    0.054, so the cool model must be deeper by a factor above 1.8. A
    blackbody has depth zero by construction, which is the failure this
    loader exists to remove.

    Fails if the model spectrum is read at the wrong wavelength scale or
    column, which moves the CO band out of the window.
    """
    cool = co_band_depth(3500, 5.0)
    sunlike = co_band_depth(5800, 4.5)
    assert cool == pytest.approx(0.122, abs=0.02)
    assert cool > 1.8 * sunlike


def test_star_jit_matches_eager():
    """Jitted and eager Star evaluation of a PHOENIX spectrum agree to 4 ULP.

    Fails if the loaded spectrum holds a non-array leaf or a host-side branch
    that behaves differently under tracing.
    """
    star = from_phoenix(
        3550.0,
        5.0,
        Ms_kg=1.0e30,
        dist_pc=5.0,
        radius_m=0.4 * Rsun2m,
        wavelength_range_nm=(3000.0, 6000.0),
        fetch=fixture_fetch,
    )
    wavelength_nm = jnp.linspace(3900.0, 5000.0, 64)
    time_jd = jnp.zeros(())
    eager = star.spec_flux_density(wavelength_nm, time_jd)
    jitted = eqx.filter_jit(lambda s, w, t: s.spec_flux_density(w, t))(
        star, wavelength_nm, time_jd
    )
    np.testing.assert_array_max_ulp(np.asarray(jitted), np.asarray(eager), maxulp=4)


@pytest.mark.slow
def test_archive_node_matches_vendored_copy():
    """The archive file at 3500 K agrees with the vendored block-averaged copy.

    Reads the full-resolution file through pooch (hash-verified), checks the
    full-resolution bolometric flux (0.994 of sigma Teff^4, tolerance 1
    percent, basis: spike) and the CO depth against the vendored copy's
    depth to 0.005 (basis: the copy is block-averaged outside the band
    windows' continuum edge only).

    Fails if the registered hash, URL or file layout no longer matches the
    archive, or the vendored fixture drifts from the real file.
    """
    try:
        path = fetch_phoenix(3500)
    except OSError as error:
        if REQUIRE_DATA:
            raise
        pytest.skip(
            f"data-absent: PHOENIX 3500 K node not cached and no network ({error})"
        )
    wavelength_nm, logg, flux, teff_k = _read_model(path)
    assert teff_k == 3500.0
    full = flux[list(logg).index(5.0)]
    np.testing.assert_allclose(
        np.trapezoid(full, wavelength_nm), SIGMA_SB_TEST * 3500.0**4, rtol=0.01
    )
    depth = co_band_depth_of(wavelength_nm, full, 3500.0)
    assert depth == pytest.approx(co_band_depth(3500, 5.0), abs=0.005)
