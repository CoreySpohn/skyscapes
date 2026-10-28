"""Stellar spectra, physical normalization, and JAX transformations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hwoutils.constants import Msun2kg, c, h, k_B, pc2m, sigma_SB
from hwoutils.conversions import jy_to_photons_per_nm_per_m2

from skyscapes.scene import AbstractStar, BlackbodyStar, FlatStar, Star


def test_simple_star_fields():
    """FlatStar stores Ms_kg / dist_pc and is an AbstractStar."""
    s = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    assert s.Ms_kg == 1.989e30
    assert s.dist_pc == 10.0
    assert isinstance(s, AbstractStar)


def test_simple_star_flux_is_flat():
    """FlatStar returns constant flux regardless of wavelength or time."""
    s = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    f1 = s.spec_flux_density(500.0, 0.0)
    f2 = s.spec_flux_density(800.0, 100.0)
    assert jnp.allclose(f1, 1e9)
    assert jnp.allclose(f2, 1e9)


@pytest.fixture
def blackbody_star():
    """Solar-temperature blackbody observed at ten parsecs."""
    return BlackbodyStar(Ms_kg=Msun2kg, dist_pc=10.0, teff_K=5772.0, radius_m=6.957e8)


def test_blackbody_star_fields(blackbody_star):
    """BlackbodyStar satisfies the stellar interface and stores physical inputs."""
    assert isinstance(blackbody_star, AbstractStar)
    assert blackbody_star.Ms_kg == Msun2kg
    assert blackbody_star.dist_pc == 10.0
    assert blackbody_star.teff_K == 5772.0
    assert blackbody_star.radius_m == 6.957e8


def test_blackbody_star_integrated_energy_flux(blackbody_star):
    """Photon energies integrated over wavelength recover Stefan-Boltzmann flux."""
    wavelength_nm = jnp.geomspace(50.0, 200_000.0, 20_000)
    photon_flux = blackbody_star.spec_flux_density(wavelength_nm, 0.0)
    energy_per_photon = h * c / (wavelength_nm * 1e-9)
    energy_flux = jnp.trapezoid(energy_per_photon * photon_flux, wavelength_nm)
    dilution = (blackbody_star.radius_m / (blackbody_star.dist_pc * pc2m)) ** 2
    expected = sigma_SB * blackbody_star.teff_K**4 * dilution
    assert jnp.isclose(energy_flux, expected, rtol=1e-3, atol=0.0)


def test_blackbody_star_photon_peak(blackbody_star):
    """The photon-count peak obeys the photon-number Wien displacement law."""
    wavelength_nm = jnp.linspace(100.0, 2000.0, 20_001)
    photon_flux = blackbody_star.spec_flux_density(wavelength_nm, 0.0)
    peak_nm = wavelength_nm[jnp.argmax(photon_flux)]
    expected_nm = h * c / (3.920690395 * k_B * blackbody_star.teff_K) * 1e9
    assert jnp.abs(peak_nm - expected_nm) <= wavelength_nm[1] - wavelength_nm[0]


@pytest.mark.parametrize(
    ("radius_factor", "distance_factor", "flux_factor"),
    [(2.0, 1.0, 4.0), (1.0, 2.0, 0.25)],
)
def test_blackbody_star_scaling(
    blackbody_star, radius_factor, distance_factor, flux_factor
):
    """Doubling radius or distance gives the exact geometric flux scaling."""
    scaled = BlackbodyStar(
        Ms_kg=blackbody_star.Ms_kg,
        dist_pc=blackbody_star.dist_pc * distance_factor,
        teff_K=blackbody_star.teff_K,
        radius_m=blackbody_star.radius_m * radius_factor,
    )
    wavelength_nm = jnp.array([200.0, 550.0, 1000.0, 10_000.0])
    baseline = blackbody_star.spec_flux_density(wavelength_nm, 0.0)
    assert jnp.array_equal(
        scaled.spec_flux_density(wavelength_nm, 0.0), baseline * flux_factor
    )


@pytest.mark.parametrize(
    "wavelength_nm", [550, [450, 550, 650], [[450, 550], [650, 750]]]
)
def test_blackbody_star_shape_and_time(blackbody_star, wavelength_nm):
    """Scalar and array wavelengths retain their shape, independent of time."""
    wavelength_nm = jnp.asarray(wavelength_nm)
    flux = blackbody_star.spec_flux_density(wavelength_nm, 0.0)
    assert flux.shape == wavelength_nm.shape
    assert jnp.all(jnp.isfinite(flux) & (flux > 0.0))
    assert jnp.array_equal(
        flux, blackbody_star.spec_flux_density(wavelength_nm, jnp.arange(7.0))
    )


def test_blackbody_star_repr(blackbody_star):
    """The one-line summary includes each physical parameter and its units."""
    assert repr(blackbody_star) == (
        "BlackbodyStar(M=1.000 Msun, dist=10 pc, Teff=5.77e+03 K, radius=6.96e+08 m)"
    )


def test_blackbody_star_matches_exojax(blackbody_star):
    """The cgs wavenumber energy spectrum converts to the same photon flux."""
    pytest.importorskip("exojax")
    from skyscapes.physical_model.exojax.emission import planck_flux_cgs

    wavelength_nm = jnp.geomspace(100.0, 100_000.0, 100)
    surface_flux_cgs = planck_flux_cgs(1e7 / wavelength_nm, blackbody_star.teff_K)
    # erg/s/cm^2 to W/m^2, then |d(wavenumber_cm)/d(wavelength_nm)|.
    surface_flux_per_nm = surface_flux_cgs * 1e-3 * 1e7 / wavelength_nm**2
    dilution = (blackbody_star.radius_m / (blackbody_star.dist_pc * pc2m)) ** 2
    photon_energy = h * c / (wavelength_nm * 1e-9)
    expected = surface_flux_per_nm * dilution / photon_energy
    actual = blackbody_star.spec_flux_density(wavelength_nm, 0.0)
    assert jnp.allclose(actual, expected, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_blackbody_star_jit_and_vmap(blackbody_star, dtype):
    """Compiled and wavelength-vectorized spectra match eager evaluation."""
    wavelength_nm = jnp.array([200.0, 550.0, 1000.0, 10_000.0], dtype=dtype)
    eager = blackbody_star.spec_flux_density(wavelength_nm, 0.0)
    compiled = jax.jit(blackbody_star.spec_flux_density)(wavelength_nm, 0.0)
    vectorized = jax.vmap(blackbody_star.spec_flux_density, in_axes=(0, None))(
        wavelength_nm, 0.0
    )
    assert jnp.all(jnp.isfinite(eager) & (eager > 0.0))
    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=0.0)
    assert jnp.allclose(vectorized, eager, rtol=1e-6, atol=0.0)


def test_spectrum_star_interpolates():
    """Star Jy-to-photon conversion reproduces analytic value."""
    wl = jnp.linspace(400.0, 1000.0, 7)
    t = jnp.linspace(0.0, 10.0, 5)
    # 1 Jy everywhere -> same phot number at each (wl, t) point
    flux_jy = jnp.ones((wl.size, t.size))
    s = Star(
        Ms_kg=1.989e30,
        dist_pc=10.0,
        wavelengths_nm=wl,
        times_jd=t,
        flux_density_jy=flux_jy,
    )
    f = s.spec_flux_density(600.0, 5.0)
    # Jy to phot: 1 Jy * 1e-26 / (600 nm * h) ~= 2.514e7
    expected = jy_to_photons_per_nm_per_m2(1.0, 600.0)
    assert jnp.allclose(f, expected, rtol=1e-4)


def test_spectrum_star_isinstance():
    """Star is an AbstractStar."""
    wl = jnp.linspace(400.0, 1000.0, 3)
    t = jnp.linspace(0.0, 1.0, 3)
    s = Star(
        Ms_kg=1.989e30,
        dist_pc=10.0,
        wavelengths_nm=wl,
        times_jd=t,
        flux_density_jy=jnp.zeros((wl.size, t.size)),
    )
    assert isinstance(s, AbstractStar)
