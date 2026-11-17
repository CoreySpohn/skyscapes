"""BlackbodyPhysicalModel -- thermal contrast against the host star."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg, Msun2kg, Rearth2m, Rsun2m

from skyscapes.physical_model import BlackbodyPhysicalModel, LambertianPhysicalModel
from skyscapes.scene import BlackbodyStar, FlatStar, Planet, System

# CODATA 2018 exact values, typed here so the expectation is independent.
H, C, KB = 6.62607015e-34, 299792458.0, 1.380649e-23
STAR = BlackbodyStar(teff_K=8840.0, radius_m=1.77 * Rsun2m)
ARGS = (jnp.zeros((1, 2)), jnp.ones((1, 2)))
T_D = jnp.array([0.0, 1.0])


def _system(model, star=STAR, rp=15.7):
    planet = Planet(jnp.array([rp]), model)
    return System.from_classical(
        star=star,
        planets=(planet,),
        star_mass_Msun=1.96,
        planet_mass_Msun=jnp.array([2256.0 * Mearth2kg / Msun2kg]),
        dist_pc=107.5,
        a_AU=jnp.array([110.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([2.6]),
        i_rad=jnp.array([1.87]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.5]),
        epoch_d=0.0,
    )


@pytest.mark.parametrize(
    "temperature, wavelength", [(1560.0, 4400.0), (150.0, 10000.0), (100.0, 1650.0)]
)
@pytest.mark.parametrize("x64", [False, True])
def test_contrast_matches_planck_ratio(temperature, wavelength, x64):
    """Fails if photon units change or float32 Wien-tail flux overflows to zero.

    At 100 K, the exponent is 87: float32 exponent rounding allows 2e-5
    relative flux error. Other float32 cases allow 1e-5, float64 1e-12.
    """
    with jax.enable_x64(x64):
        star = STAR
        dist_pc = 107.5
        if temperature == 100.0:
            star = BlackbodyStar(1000.0, 0.1 * Rsun2m)
            dist_pc = 10.0
        model = BlackbodyPhysicalModel(teff_K=jnp.array([temperature]))
        wl, rp = jnp.asarray(wavelength), jnp.array([15.7])
        args = (jnp.zeros((1, 2)), jnp.ones((1, 2)))
        times = jnp.array([0.0, 1.0])
        got = model.contrast_at_time(
            *args, wl, rp, time_d=times, star=star, dist_pc=dist_pc
        )
        flat = FlatStar(
            flux_phot_per_nm_m2=star.photon_flux_density(wl, times, dist_pc=dist_pc),
        )
        generic = model.contrast_at_time(
            *args, wl, rp, time_d=times, star=flat, dist_pc=dist_pc
        )
        x = H * C / (wavelength * 1e-9 * KB)
        area = (15.7 * 6.3781e6 / star.radius_m) ** 2
        expected = area * np.expm1(x / star.teff_K) / np.expm1(x / temperature)
        assert got.shape == (1, 2)
        rtol = 1e-12 if x64 else 1e-5
        if not x64 and temperature == 100.0:
            rtol = 2e-5
        assert jnp.allclose(got, expected, rtol=rtol, atol=0.0)
        assert jnp.allclose(got, generic, rtol=rtol, atol=0.0)
        flux = model._photon_flux_density(
            wl,
            phase_angle_rad=args[0],
            dist_AU=args[1],
            Rp_Rearth=rp,
            time_d=times,
            star=star,
            dist_pc=dist_pc,
        )
        # Planck photon radiance per nm, diluted by the apparent disk solid angle.
        radiance = 2 * C / (wavelength * 1e-9) ** 4 * 1e-9 / np.expm1(x / temperature)
        expected_flux = (
            radiance * np.pi * (15.7 * 6.3781e6 / (dist_pc * 3.085677581491367e16)) ** 2
        )
        np.testing.assert_allclose(flux, expected_flux, rtol=rtol, atol=0)
        direct = model._photon_flux_density(wl, rp, dist_pc=dist_pc)
        np.testing.assert_allclose(direct, expected_flux, rtol=rtol, atol=0)
        np.testing.assert_allclose(
            flux, jnp.broadcast_to(direct[:, None], (1, 2)), rtol=rtol, atol=0
        )
    assert Rearth2m == pytest.approx(6.3781e6, rel=1e-4)
    assert Rsun2m == pytest.approx(6.957e8, rel=1e-4)


def test_star_twin_has_unit_contrast():
    """A planet with the star's radius and temperature is as bright as the star."""
    model = BlackbodyPhysicalModel(teff_K=jnp.array([STAR.teff_K]))
    rp = jnp.array([STAR.radius_m / Rearth2m])
    got = model.contrast_at_time(
        *ARGS, jnp.asarray(1000.0), rp, time_d=T_D, star=STAR, dist_pc=107.5
    )
    assert jnp.allclose(got, 1.0, rtol=1e-5, atol=0.0)


def test_contrast_follows_the_star_model():
    """Against a flat star the contrast is the planet flux over the flat flux."""
    model = BlackbodyPhysicalModel(teff_K=jnp.array([1560.0]))
    flat = FlatStar(flux_phot_per_nm_m2=3.0)
    rp = jnp.array([15.7])
    got = model.contrast_at_time(
        *ARGS, jnp.asarray(4400.0), rp, time_d=T_D, star=flat, dist_pc=107.5
    )
    flux = model._photon_flux_density(jnp.asarray(4400.0), rp, dist_pc=107.5)
    assert jnp.allclose(got, flux[:, None] / 3.0, rtol=1e-6, atol=0.0)


def test_contrast_without_star_raises():
    """The phase-only hook and a missing star both refuse."""
    model = BlackbodyPhysicalModel(teff_K=jnp.array([1560.0]))
    with pytest.raises(ValueError, match="star"):
        model.contrast(*ARGS, jnp.asarray(4400.0), jnp.array([15.7]))
    with pytest.raises(ValueError, match="host star"):
        model.contrast_at_time(
            *ARGS, jnp.asarray(4400.0), jnp.array([15.7]), time_d=T_D
        )


def test_system_contrast_cube_stacks_wavelengths():
    """Check disk dilution and Planck contrast from SI primitives.

    The 1e-14 relative tolerance permits float64 algebraic rounding.
    """
    system = _system(BlackbodyPhysicalModel(teff_K=jnp.array([1560.0])))
    wavelength_m = 4400.0e-9
    x = H * C / (wavelength_m * KB)
    area = (15.7 * Rearth2m / STAR.radius_m) ** 2
    expected_contrast = area * np.expm1(x / STAR.teff_K) / np.expm1(x / 1560.0)
    expected_flux = (
        2
        * np.pi
        * C
        * 1e-9
        / wavelength_m**4
        / np.expm1(x / 1560.0)
        * (15.7 * Rearth2m / (107.5 * 3.085677581491367e16)) ** 2
    )
    np.testing.assert_allclose(
        system.contrasts(jnp.asarray(4400.0), jnp.array([0.0, 100.0])),
        [[expected_contrast, expected_contrast]],
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        system.planet_flux_densities(jnp.asarray(4400.0), jnp.array([0.0, 100.0])),
        [[expected_flux, expected_flux]],
        rtol=1e-14,
        atol=0,
    )
    wl = jnp.array([3900.0, 4400.0, 4900.0])
    cube = system.contrast_cube(wl, T_D)
    assert cube.shape == (3, 1, 2)
    for k in range(3):
        assert jnp.allclose(cube[k], system.contrasts(wl[k], T_D), rtol=1e-6, atol=0.0)
        alpha, dmag = system.alpha_dMag(T_D, wavelength_nm=wl[k])
        np.testing.assert_allclose(dmag, -2.5 * jnp.log10(cube[k]), rtol=1e-12)
        np.testing.assert_allclose(
            alpha, jnp.linalg.norm(system.positions(T_D), axis=0), rtol=1e-12
        )
    for default, explicit in zip(
        system.alpha_dMag(T_D),
        system.alpha_dMag(T_D, wavelength_nm=600.0),
        strict=True,
    ):
        np.testing.assert_array_equal(default, explicit)
    assert cube[2, 0, 0] > cube[0, 0, 0]  # a cool planet is redder than a hot star


def test_system_contrast_cube_reflected_light():
    """A grey reflector gives the same contrast at every wavelength."""
    system = _system(LambertianPhysicalModel(Ag=jnp.array([0.3])), rp=1.0)
    cube = system.contrast_cube(jnp.array([500.0, 900.0]), T_D)
    assert cube.shape == (2, 1, 2)
    assert jnp.array_equal(cube[0], cube[1])


def test_temperature_gradient_matches_finite_difference():
    """d ln(contrast) / dT through the system against a central difference."""
    wl = jnp.asarray(4400.0)

    def log_contrast(teff):
        system = _system(BlackbodyPhysicalModel(teff_K=teff))
        return jnp.log(system.contrasts(wl, T_D))[0, 0]

    t0 = jnp.array([1560.0])
    auto = jax.grad(log_contrast)(t0)[0]
    x = H * C / (4400.0e-9 * KB)
    expected = x / 1560.0**2 / (1.0 - jnp.exp(-x / 1560.0))
    assert jnp.allclose(auto, expected, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize(
    "temperature, wavelength", [(1560.0, 4400.0), (150.0, 10000.0)]
)
@pytest.mark.parametrize(
    "quantity", ["log_contrast", "log_contrast_star", "planet_flux", "star_flux"]
)
def test_temperature_gradient_is_finite_in_single_precision(
    temperature, wavelength, quantity
):
    """Check both temperature derivatives and curvature against SI primitives.

    Float64 arithmetic supplies the reference; float32 rounding permits
    1e-5 relative value, 1e-4 gradient, and 1e-3 curvature error.
    """

    def reference(teff):
        x = H * C / (wavelength * 1e-9 * KB)
        if quantity in ("log_contrast", "log_contrast_star"):
            area = (15.7 * 6.3781e6 / STAR.radius_m) ** 2
            planet_t = temperature if quantity == "log_contrast_star" else teff[0]
            star_t = teff[0] if quantity == "log_contrast_star" else STAR.teff_K
            return jnp.log(area * jnp.expm1(x / star_t) / jnp.expm1(x / planet_t))
        radius = STAR.radius_m if quantity == "star_flux" else 15.7 * 6.3781e6
        dilution = (radius / (107.5 * 3.085677581491367e16)) ** 2
        return (
            2
            * jnp.pi
            * C
            / (wavelength * 1e-9) ** 4
            * dilution
            * 1e-9
            / jnp.expm1(x / teff[0])
        )

    variable_temperature = (
        STAR.teff_K if quantity == "log_contrast_star" else temperature
    )
    with jax.enable_x64(True):
        t64 = jnp.array([variable_temperature], jnp.float64)
        expected_value = reference(t64)
        expected_grad, expected_hvp = jax.jvp(
            jax.grad(reference), (t64,), (jnp.ones_like(t64),)
        )

    with jax.enable_x64(False):
        wl = jnp.asarray(wavelength, jnp.float32)
        args = (jnp.zeros((1, 2), jnp.float32), jnp.ones((1, 2), jnp.float32))
        time_d = jnp.array([0.0, 1.0], jnp.float32)
        rp = jnp.array([15.7], jnp.float32)

        def observable(teff):
            model = BlackbodyPhysicalModel(
                teff_K=jnp.array([temperature])
                if quantity == "log_contrast_star"
                else teff
            )
            if quantity == "planet_flux":
                return model._photon_flux_density(wl, rp, dist_pc=107.5)[0]
            if quantity == "star_flux":
                star = BlackbodyStar(teff[0], STAR.radius_m)
                return star.photon_flux_density(wl, time_d, dist_pc=107.5)
            star = (
                BlackbodyStar(teff[0], STAR.radius_m)
                if quantity == "log_contrast_star"
                else STAR
            )
            out = model.contrast_at_time(
                *args, wl, rp, time_d=time_d, star=star, dist_pc=107.5
            )
            return jnp.log(out)[0, 0]

        def derivatives(teff):
            grad, hvp = jax.jvp(jax.grad(observable), (teff,), (jnp.ones_like(teff),))
            return observable(teff), grad, hvp

        t0 = jnp.array([variable_temperature], jnp.float32)
        for evaluate in (derivatives, jax.jit(derivatives)):
            value, auto, curvature = evaluate(t0)
            assert auto.dtype == curvature.dtype == jnp.float32
            np.testing.assert_allclose(value, expected_value, rtol=1e-5, atol=0.0)
            np.testing.assert_allclose(auto, expected_grad, rtol=1e-4, atol=0.0)
            np.testing.assert_allclose(curvature, expected_hvp, rtol=1e-3, atol=0.0)
