"""BlackbodyPhysicalModel -- thermal contrast against the host star."""

from __future__ import annotations

import jax
import jax.numpy as jnp
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


def test_contrast_matches_planck_ratio():
    """Contrast = (Rp / Rs)^2 expm1(x / Ts) / expm1(x / Tp), x = h c / (lambda k)."""
    model = BlackbodyPhysicalModel(teff_K=jnp.array([1560.0]))
    wl = 4400.0
    got = model.contrast_at_time(
        *ARGS, jnp.asarray(wl), jnp.array([15.7]), time_jd=T_JD, star=STAR
    )
    x = H * C / (wl * 1e-9 * KB)
    area = (15.7 * 6.3781e6 / (1.77 * 6.957e8)) ** 2
    expected = area * jnp.expm1(x / 8840.0) / jnp.expm1(x / 1560.0)
    assert got.shape == (1, 2)
    # float32: relative round-off of the two Planck factors.
    assert jnp.allclose(got, expected, rtol=2e-5, atol=0.0)
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


def test_temperature_gradient_is_finite_in_single_precision():
    """The same derivative in float32: ``lambda k T`` squared would underflow."""
    with jax.enable_x64(False):
        wl = jnp.asarray(4400.0, jnp.float32)
        args = (jnp.zeros((1, 2), jnp.float32), jnp.ones((1, 2), jnp.float32))
        t_jd = jnp.array([0.0, 1.0], jnp.float32)
        rp = jnp.array([15.7], jnp.float32)

        def log_contrast(teff):
            model = BlackbodyPhysicalModel(teff_K=teff)
            out = model.contrast_at_time(*args, wl, rp, time_jd=t_jd, star=STAR)
            return jnp.log(out)[0, 0]

        t0 = jnp.array([1560.0], jnp.float32)
        auto = jax.grad(log_contrast)(t0)[0]
        curvature = jax.hessian(log_contrast)(t0)[0, 0]
        assert auto.dtype == jnp.float32
    x = H * C / (4400.0e-9 * KB)
    expected = x / 1560.0**2 / (1.0 - jnp.exp(-x / 1560.0))
    assert jnp.allclose(auto, expected, rtol=1e-4, atol=0.0)
    assert jnp.isfinite(curvature) and curvature < 0.0
