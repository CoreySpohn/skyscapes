"""Stellar spectra, physical normalization, and JAX transformations."""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from hwoutils.constants import c, h, k_B, pc2m, sigma_SB
from hwoutils.conversions import (
    jy_to_photons_per_nm_per_m2,
)
from hwoutils.spectral import Band, Quadrature

from skyscapes.scene import AbstractStar, BlackbodyStar, FlatStar, Star


@pytest.mark.parametrize("kind", AbstractStar.__subclasses__())
def test_angular_diameter_contract(kind):
    """Fails if a star detaches its size from JAX or misinterprets distance.

    Fixed diameters are exact inputs. The physical diameter uses the
    small-angle ratio in independently converted astropy units (1e-12).
    """
    distance = jnp.array([10.0, 20.0])
    fixed = 0.003
    radius = 6.957e8
    if kind is FlatStar:
        star = kind(1e9, fixed)
        expected = [fixed, fixed]
        assert repr(star) == "FlatStar(flux=1e+09 ph/s/m^2/nm, diam=0.003 arcsec)"
        assert FlatStar(1e9).angular_diameter_arcsec(10.0) == 0.0
    elif kind is BlackbodyStar:
        star = kind(5772.0, radius)
        expected = (2 * radius * u.m / (np.asarray(distance) * u.pc)).to_value(
            u.arcsec, equivalencies=u.dimensionless_angles()
        )
    else:
        assert kind is Star
        star = kind(
            wavelengths_nm=jnp.array([500.0, 600.0]),
            times_d=jnp.array([0.0]),
            flux_density_jy=jnp.ones((2, 1)),
            fixed_diameter_arcsec=fixed,
        )
        expected = [fixed, fixed]

    assert not hasattr(star, "Ms_kg")
    assert not hasattr(star, "dist_pc")

    @eqx.filter_jit
    def diameter(source, distances):
        return jax.vmap(source.angular_diameter_arcsec)(distances)

    actual = diameter(star, distance)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    size = (
        (lambda s: s.radius_m)
        if kind is BlackbodyStar
        else (lambda s: s.fixed_diameter_arcsec)
    )
    doubled = eqx.tree_at(size, star, jnp.asarray(2 * size(star)))
    np.testing.assert_allclose(diameter(doubled, distance), 2 * actual, rtol=1e-12)


def test_simple_star_flux_is_flat():
    """Fails if apparent flux changes with wavelength, time or system distance."""
    s = FlatStar(flux_phot_per_nm_m2=1e9)
    f1 = s.photon_flux_density(500.0, 0.0, dist_pc=10.0)
    f2 = s.photon_flux_density(800.0, 100.0, dist_pc=20.0)
    assert jnp.allclose(f1, 1e9)
    assert jnp.allclose(f2, 1e9)


@pytest.fixture
def blackbody_star():
    """Solar-temperature blackbody observed at ten parsecs."""
    return BlackbodyStar(teff_K=5772.0, radius_m=6.957e8)


def test_blackbody_star_integrated_energy_flux(blackbody_star):
    """Photon energies integrated over wavelength recover Stefan-Boltzmann flux."""
    wavelength_nm = jnp.geomspace(50.0, 200_000.0, 20_000)
    photon_flux = blackbody_star.photon_flux_density(wavelength_nm, 0.0, dist_pc=10.0)
    energy_per_photon = h * c / (wavelength_nm * 1e-9)
    energy_flux = jnp.trapezoid(energy_per_photon * photon_flux, wavelength_nm)
    dilution = (blackbody_star.radius_m / (10.0 * pc2m)) ** 2
    expected = sigma_SB * blackbody_star.teff_K**4 * dilution
    assert jnp.isclose(energy_flux, expected, rtol=1e-3, atol=0.0)


def test_blackbody_star_photon_peak(blackbody_star):
    """The photon-count peak obeys the photon-number Wien displacement law."""
    wavelength_nm = jnp.linspace(100.0, 2000.0, 20_001)
    photon_flux = blackbody_star.photon_flux_density(wavelength_nm, 0.0, dist_pc=10.0)
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
    """Fails if radius or distance updates leave flux stale."""
    scaled = BlackbodyStar(
        teff_K=blackbody_star.teff_K,
        radius_m=blackbody_star.radius_m * radius_factor,
    )
    wavelength_nm = jnp.array([200.0, 550.0, 1000.0, 10_000.0])
    baseline = blackbody_star.photon_flux_density(wavelength_nm, 0.0, dist_pc=10.0)
    assert jnp.array_equal(
        scaled.photon_flux_density(wavelength_nm, 0.0, dist_pc=10.0 * distance_factor),
        baseline * flux_factor,
    )


@pytest.mark.parametrize(
    "wavelength_nm", [550, [450, 550, 650], [[450, 550], [650, 750]]]
)
def test_blackbody_star_shape_and_time(blackbody_star, wavelength_nm):
    """Scalar and array wavelengths retain their shape, independent of time."""
    wavelength_nm = jnp.asarray(wavelength_nm)
    flux = blackbody_star.photon_flux_density(wavelength_nm, 0.0, dist_pc=10.0)
    assert flux.shape == wavelength_nm.shape
    assert jnp.all(jnp.isfinite(flux) & (flux > 0.0))
    assert jnp.array_equal(
        flux,
        blackbody_star.photon_flux_density(
            wavelength_nm, jnp.arange(7.0), dist_pc=10.0
        ),
    )


def test_blackbody_star_repr(blackbody_star):
    """The one-line summary includes each physical parameter and its units."""
    assert repr(blackbody_star) == ("BlackbodyStar(Teff=5.77e+03 K, radius=6.96e+08 m)")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_blackbody_star_jit_and_vmap(blackbody_star, dtype):
    """Compiled and wavelength-vectorized spectra match eager evaluation."""
    wavelength_nm = jnp.array([200.0, 550.0, 1000.0, 10_000.0], dtype=dtype)
    eager = blackbody_star.photon_flux_density(wavelength_nm, 0.0, dist_pc=10.0)
    compiled = jax.jit(blackbody_star.photon_flux_density)(
        wavelength_nm, 0.0, dist_pc=10.0
    )
    vectorized = jax.vmap(
        partial(blackbody_star.photon_flux_density, dist_pc=10.0), in_axes=(0, None)
    )(wavelength_nm, 0.0)
    assert jnp.all(jnp.isfinite(eager) & (eager > 0.0))
    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=0.0)
    assert jnp.allclose(vectorized, eager, rtol=1e-6, atol=0.0)


def test_spectrum_star_interpolates():
    """Star Jy-to-photon conversion reproduces analytic value."""
    wl = jnp.linspace(400.0, 1000.0, 7)
    t = jnp.linspace(0.0, 10.0, 5)
    # 1 Jy at each table node; photon density varies inversely with wavelength.
    flux_jy = jnp.ones((wl.size, t.size))
    s = Star(
        wavelengths_nm=wl,
        times_d=t,
        flux_density_jy=flux_jy,
    )
    f = s.photon_flux_density(600.0, 5.0, dist_pc=10.0)
    # Jy to phot: 1 Jy * 1e-26 / (600 nm * h) ~= 2.5153e4
    expected = jy_to_photons_per_nm_per_m2(1.0, 600.0)
    assert jnp.allclose(f, expected, rtol=1e-4)


@pytest.mark.parametrize("n_epochs", [1, 5])
def test_spectrum_star_wavelength_coverage(n_epochs):
    """Fails if either epoch path loses the grid or returns non-float bounds."""
    wl = jnp.linspace(500.0, 1000.0, 7)
    t = jnp.linspace(0.0, 10.0, n_epochs)
    s = Star(
        wavelengths_nm=wl,
        times_d=t,
        flux_density_jy=jnp.ones((wl.size, t.size)),
    )
    assert s.wavelengths_nm is wl
    assert jnp.array_equal(s.wavelengths_nm, wl)
    bounds = s.wavelength_range_nm
    assert bounds == (500.0, 1000.0)
    assert all(isinstance(bound, float) for bound in bounds)


@pytest.mark.parametrize(
    "kind,n_epochs",
    [
        (kind, n)
        for kind in AbstractStar.__subclasses__()
        for n in ((1, 2) if kind is Star else (1,))
    ],
)
def test_photon_contract(kind, n_epochs):
    """Fails on a unit error, Jy interpolation, or lost wavelength/time axes.

    Tolerance 3e-14 is the float64 floor for the shared algebra. The rounded
    9.9634e5 anchor allows 5e-6 relative for its five significant digits.
    FlatStar is flat in photons; Star supplies the flat-F_nu AB=5 anchor
    at a table node and interpolates in photons between nodes.
    """
    wave = jnp.array([[500.0, 550.0], [575.0, 600.0]])
    fnu = 3631.0 * 10**-2
    if kind is FlatStar:
        star = kind(9.9634e5)
        expected = np.full(wave.shape, 9.9634e5)
    elif kind is BlackbodyStar:
        star = kind(5772.0, 6.957e8)
        lam = np.asarray(wave) * 1e-9
        expected = (
            2
            * np.pi
            * 299792458.0
            / lam**4
            / np.expm1(6.62607015e-34 * 299792458.0 / (lam * 1.380649e-23 * 5772.0))
        )
        expected *= (6.957e8 / (10 * 3.085677581491367e16)) ** 2 * 1e-9
    else:
        assert kind is Star
        star = kind(
            wavelengths_nm=jnp.array([450.0, 550.0, 650.0]),
            times_d=jnp.arange(n_epochs, dtype=float),
            flux_density_jy=jnp.full((3, n_epochs), fnu),
            interpolation_method="linear",
        )
        knots = np.array([450.0, 550.0, 650.0])
        photons = fnu * 1e-26 / (6.62607015e-34 * knots * 1e-9) * 1e-9
        expected = np.interp(wave, knots, photons)
        np.testing.assert_allclose(expected[0, 1], 9.9634e5, rtol=5e-6)
    time = jnp.array([[0.0], [0.5]]) if n_epochs > 1 else 0.0
    actual = star.photon_flux_density(wave, time, dist_pc=10.0)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    assert not hasattr(star, "spec_flux_density")
    assert actual.shape == wave.shape
    np.testing.assert_allclose(
        eqx.filter_jit(star.photon_flux_density)(wave, time, dist_pc=10.0),
        expected,
        rtol=3e-14,
        atol=0,
    )


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_flat_fnu_quadrature_anchor(method):
    """Fails on a per-nm error, missing knots, or photon integration bias.

    A flat-F_nu AB=5 table sampled every 0.025 nm approximates 1/lambda.
    Linear interpolation's integral error is bounded by width*step**2/12
    times max(abs((1/lambda)'')); relative error is below 4.6e-10 here.
    Cubic interpolation is more accurate. Order 4 integrates each polynomial
    segment exactly; 1e-9 relative covers interpolation and roundoff.
    The rounded 550-nm anchor uses 5e-6 relative; constant parity is exact.
    """
    assert h == 6.62607015e-34
    fnu = 3631.0 * 10**-2
    star = Star(
        wavelengths_nm=jnp.linspace(500.0, 600.0, 4001),
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.full((4001, 1), fnu),
        interpolation_method=method,
    )
    anchor = fnu * 1e-26 / (6.62607015e-34 * 550e-9) * 1e-9
    np.testing.assert_allclose(
        star.photon_flux_density(550.0, 0.0, dist_pc=10.0), anchor, rtol=3e-14
    )
    np.testing.assert_allclose(anchor, 9.9634e5, rtol=5e-6)
    np.testing.assert_array_equal(star.spectral_breaks_nm(), star.wavelengths_nm)
    rule = Quadrature.build([Band(500.0, 600.0)], star, order=4)
    actual = rule.integrate(star.photon_flux_density(rule.nodes_nm, 0.0, dist_pc=10.0))
    expected = fnu * 1e-26 / 6.62607015e-34 * np.log(600 / 500)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)
    if method == "linear":
        midpoint = Quadrature.build([Band(500.0, 600.0)], star, order=1)
        np.testing.assert_allclose(
            midpoint.integrate(
                star.photon_flux_density(midpoint.nodes_nm, 0.0, dist_pc=10.0)
            ),
            actual,
            rtol=3e-14,
            atol=0,
        )


def test_photon_jax_and_stellar_gradient(blackbody_star):
    """Fails if tracing or batching changes photons or temperature derivatives detach.

    JIT/vmap use 3e-14 relative float64 algebra; central difference step 1e-4
    in T/5772 gives O(step**2) truncation, bounded at 1e-7 relative.
    """
    wave = jnp.array([500.0, 550.0, 600.0])

    def evaluate(w):
        return blackbody_star.photon_flux_density(w, 0.0, dist_pc=10.0)

    loop = jnp.stack([evaluate(w) for w in wave])
    np.testing.assert_allclose(eqx.filter_jit(evaluate)(wave), loop, rtol=3e-14, atol=0)
    np.testing.assert_allclose(jax.vmap(evaluate)(wave), loop, rtol=3e-14, atol=0)

    def normalized_flux(scale):
        varied = eqx.tree_at(lambda s: s.teff_K, blackbody_star, scale * 5772.0)
        return varied.photon_flux_density(550.0, 0.0, dist_pc=10.0) / evaluate(550.0)

    step = 1e-4
    fd = (normalized_flux(1 + step) - normalized_flux(1 - step)) / (2 * step)
    derivative = jax.grad(normalized_flux)(1.0)
    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(derivative, fd, rtol=1e-7, atol=0)
