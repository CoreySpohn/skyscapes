"""BlackbodyPhysicalModel -- thermal contrast against the host star."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg, Rearth2m, Rsun2m
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit

from skyscapes.physical_model import BlackbodyPhysicalModel, LambertianPhysicalModel
from skyscapes.scene import BlackbodyStar, FlatStar, Planet, System

SOLVER = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
# CODATA 2018 exact values, typed here so the expectation is independent.
H, C, KB = 6.62607015e-34, 299792458.0, 1.380649e-23
STAR = BlackbodyStar(
    Ms_kg=1.96 * Msun2kg, dist_pc=107.5, teff_K=8840.0, radius_m=1.77 * Rsun2m
)
ARGS = (jnp.zeros((1, 2)), jnp.ones((1, 2)))
T_JD = jnp.array([0.0, 1.0])


def _system(model, star=STAR, rp=15.7):
    orbit = KeplerianOrbit(
        a_AU=jnp.array([110.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([2.6]),
        i_rad=jnp.array([1.87]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.5]),
        t0_d=jnp.array([0.0]),
    )
    planet = Planet(jnp.array([rp]), jnp.array([2256.0]), orbit, model)
    return System(star=star, planets=(planet,), trig_solver=SOLVER)


@pytest.mark.parametrize(
    "temperature, wavelength", [(1560.0, 4400.0), (150.0, 10000.0)]
)
@pytest.mark.parametrize("x64", [False, True])
def test_contrast_matches_planck_ratio(temperature, wavelength, x64):
    """Fails if the direct ratio or generic stellar-flux division changes units."""
    with jax.enable_x64(x64):
        model = BlackbodyPhysicalModel(teff_K=jnp.array([temperature]))
        wl, rp = jnp.asarray(wavelength), jnp.array([15.7])
        args = (jnp.zeros((1, 2)), jnp.ones((1, 2)))
        times = jnp.array([0.0, 1.0])
        got = model.contrast_at_time(*args, wl, rp, time_jd=times, star=STAR)
        flat = FlatStar(
            Ms_kg=STAR.Ms_kg,
            dist_pc=STAR.dist_pc,
            flux_phot_per_nm_m2=STAR.spec_flux_density(wl, times),
        )
        generic = model.contrast_at_time(*args, wl, rp, time_jd=times, star=flat)
        x = H * C / (wavelength * 1e-9 * KB)
        area = (15.7 * 6.3781e6 / (1.77 * 6.957e8)) ** 2
        expected = area * jnp.expm1(x / 8840.0) / jnp.expm1(x / temperature)
        assert got.shape == (1, 2)
        rtol = 1e-12 if x64 else 1e-5
        assert jnp.allclose(got, expected, rtol=rtol, atol=0.0)
        assert jnp.allclose(got, generic, rtol=rtol, atol=0.0)
    assert Rearth2m == pytest.approx(6.3781e6, rel=1e-4)
    assert Rsun2m == pytest.approx(6.957e8, rel=1e-4)


def test_star_twin_has_unit_contrast():
    """A planet with the star's radius and temperature is as bright as the star."""
    model = BlackbodyPhysicalModel(teff_K=jnp.array([STAR.teff_K]))
    rp = jnp.array([STAR.radius_m / Rearth2m])
    got = model.contrast_at_time(
        *ARGS, jnp.asarray(1000.0), rp, time_jd=T_JD, star=STAR
    )
    assert jnp.allclose(got, 1.0, rtol=1e-5, atol=0.0)


def test_contrast_follows_the_star_model():
    """Against a flat star the contrast is the planet flux over the flat flux."""
    model = BlackbodyPhysicalModel(teff_K=jnp.array([1560.0]))
    flat = FlatStar(Ms_kg=STAR.Ms_kg, dist_pc=STAR.dist_pc, flux_phot_per_nm_m2=3.0)
    rp = jnp.array([15.7])
    got = model.contrast_at_time(
        *ARGS, jnp.asarray(4400.0), rp, time_jd=T_JD, star=flat
    )
    flux = model.spec_flux_density(jnp.asarray(4400.0), rp, flat.dist_pc)
    assert jnp.allclose(got, flux[:, None] / 3.0, rtol=1e-6, atol=0.0)


def test_contrast_without_star_raises():
    """The phase-only hook and a missing star both refuse."""
    model = BlackbodyPhysicalModel(teff_K=jnp.array([1560.0]))
    with pytest.raises(ValueError, match="star"):
        model.contrast(*ARGS, jnp.asarray(4400.0), jnp.array([15.7]))
    with pytest.raises(ValueError, match="host star"):
        model.contrast_at_time(
            *ARGS, jnp.asarray(4400.0), jnp.array([15.7]), time_jd=T_JD
        )


def test_system_contrast_cube_stacks_wavelengths():
    """contrast_cube equals contrasts wavelength by wavelength, shape (W, K, T)."""
    system = _system(BlackbodyPhysicalModel(teff_K=jnp.array([1560.0])))
    wl = jnp.array([3900.0, 4400.0, 4900.0])
    cube = system.contrast_cube(wl, T_JD)
    assert cube.shape == (3, 1, 2)
    for k in range(3):
        assert jnp.allclose(cube[k], system.contrasts(wl[k], T_JD), rtol=1e-6, atol=0.0)
        alpha, dmag = system.alpha_dMag(T_JD, wavelength_nm=wl[k])
        np.testing.assert_allclose(dmag, -2.5 * jnp.log10(cube[k]), rtol=1e-12)
        np.testing.assert_allclose(
            alpha, jnp.linalg.norm(system.positions(T_JD), axis=0), rtol=1e-12
        )
    for default, explicit in zip(
        system.alpha_dMag(T_JD),
        system.alpha_dMag(T_JD, wavelength_nm=600.0),
        strict=True,
    ):
        np.testing.assert_array_equal(default, explicit)
    assert cube[2, 0, 0] > cube[0, 0, 0]  # a cool planet is redder than a hot star


def test_system_contrast_cube_reflected_light():
    """A grey reflector gives the same contrast at every wavelength."""
    system = _system(LambertianPhysicalModel(Ag=jnp.array([0.3])), rp=1.0)
    cube = system.contrast_cube(jnp.array([500.0, 900.0]), T_JD)
    assert cube.shape == (2, 1, 2)
    assert jnp.array_equal(cube[0], cube[1])


def test_temperature_gradient_matches_finite_difference():
    """d ln(contrast) / dT through the system against a central difference."""
    wl = jnp.asarray(4400.0)

    def log_contrast(teff):
        system = _system(BlackbodyPhysicalModel(teff_K=teff))
        return jnp.log(system.contrasts(wl, T_JD))[0, 0]

    t0 = jnp.array([1560.0])
    auto = jax.grad(log_contrast)(t0)[0]
    x = H * C / (4400.0e-9 * KB)
    expected = x / 1560.0**2 / (1.0 - jnp.exp(-x / 1560.0))
    assert jnp.allclose(auto, expected, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize(
    "temperature, wavelength", [(1560.0, 4400.0), (150.0, 10000.0)]
)
@pytest.mark.parametrize("quantity", ["log_contrast", "planet_flux", "star_flux"])
def test_temperature_gradient_is_finite_in_single_precision(
    temperature, wavelength, quantity
):
    """Fails if unbalanced Planck factors corrupt float32 gradients or HVPs."""

    def reference(teff):
        x = H * C / (wavelength * 1e-9 * KB)
        if quantity == "log_contrast":
            area = (15.7 * 6.3781e6 / STAR.radius_m) ** 2
            return jnp.log(area * jnp.expm1(x / STAR.teff_K) / jnp.expm1(x / teff[0]))
        radius = STAR.radius_m if quantity == "star_flux" else 15.7 * 6.3781e6
        dilution = (radius / (STAR.dist_pc * 3.085677581491367e16)) ** 2
        return (
            2
            * jnp.pi
            * C
            / (wavelength * 1e-9) ** 4
            * dilution
            * 1e-9
            / jnp.expm1(x / teff[0])
        )

    with jax.enable_x64(True):
        t64 = jnp.array([temperature], jnp.float64)
        expected_value = reference(t64)
        expected_grad, expected_hvp = jax.jvp(
            jax.grad(reference), (t64,), (jnp.ones_like(t64),)
        )

    with jax.enable_x64(False):
        wl = jnp.asarray(wavelength, jnp.float32)
        args = (jnp.zeros((1, 2), jnp.float32), jnp.ones((1, 2), jnp.float32))
        time_jd = jnp.array([0.0, 1.0], jnp.float32)
        rp = jnp.array([15.7], jnp.float32)

        def observable(teff):
            model = BlackbodyPhysicalModel(teff_K=teff)
            if quantity == "planet_flux":
                return model.spec_flux_density(wl, rp, STAR.dist_pc)[0]
            if quantity == "star_flux":
                star = BlackbodyStar(STAR.Ms_kg, STAR.dist_pc, teff[0], STAR.radius_m)
                return star.spec_flux_density(wl, time_jd)
            out = model.contrast_at_time(*args, wl, rp, time_jd=time_jd, star=STAR)
            return jnp.log(out)[0, 0]

        def derivatives(teff):
            grad, hvp = jax.jvp(jax.grad(observable), (teff,), (jnp.ones_like(teff),))
            return observable(teff), grad, hvp

        t0 = jnp.array([temperature], jnp.float32)
        for evaluate in (derivatives, jax.jit(derivatives)):
            value, auto, curvature = evaluate(t0)
            assert auto.dtype == curvature.dtype == jnp.float32
            np.testing.assert_allclose(value, expected_value, rtol=1e-5, atol=0.0)
            np.testing.assert_allclose(auto, expected_grad, rtol=1e-4, atol=0.0)
            np.testing.assert_allclose(curvature, expected_hvp, rtol=1e-3, atol=0.0)
