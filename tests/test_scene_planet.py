"""scene.Planet -- composes AbstractOrbit + AbstractPhysicalModel."""

from __future__ import annotations

import jax.numpy as jnp
from hwoutils.constants import Rearth2AU
from orbix.equations.phase import lambert_phase_exact
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit

from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet

SOLVER = get_grid_solver(level="scalar", E=False, trig=True, jit=True)


def _single_planet():
    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([jnp.pi / 3]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([jnp.pi / 4]),
        t0_d=jnp.array([0.0]),
    )
    physical_model = LambertianPhysicalModel(Ag=jnp.array([0.3]))
    return Planet(
        Rp_Rearth=jnp.array([1.0]),
        Mp_Mearth=jnp.array([1.0]),
        orbit=orbit,
        physical_model=physical_model,
    )


def test_planet_n_planets_property():
    """Planet.n_planets reflects the K dimension of intrinsic params."""
    p = _single_planet()
    assert p.n_planets == 1


def test_planet_position_shape():
    """Planet.position_arcsec returns shape (2, K, T)."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p = _single_planet()
    ra_dec = p.position_arcsec(SOLVER, jnp.array([0.0]), star=star)
    # (2, K, T) per contract
    assert ra_dec.shape == (2, 1, 1)


def test_planet_alpha_dMag_matches_orbix():
    """Planet.alpha_dMag reproduces the exact Lambert dMag formula."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p = _single_planet()
    t_jd = jnp.array([0.0, 100.0, 200.0])

    alpha_new, dMag_new = p.alpha_dMag(SOLVER, t_jd, star=star)

    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([jnp.pi / 3]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([jnp.pi / 4]),
        t0_d=jnp.array([0.0]),
    )
    Ms_kg = jnp.array([1.989e30])
    dist_pc = jnp.array([10.0])
    Ag = jnp.array([0.3])
    Rp_AU = jnp.array([1.0]) * Rearth2AU

    _, beta, dist_AU = orbit.propagate(SOLVER, t_jd, Ms_kg=Ms_kg)
    phase = lambert_phase_exact(jnp.cos(beta), jnp.sin(beta))
    log_arg = Ag[:, None] * (Rp_AU[:, None] / dist_AU) ** 2 * phase
    dMag_ref = -2.5 * jnp.log10(log_arg + jnp.finfo(log_arg.dtype).tiny)
    alpha_ref = orbit.separation_arcsec(SOLVER, t_jd, Ms_kg=Ms_kg, dist_pc=dist_pc)

    assert jnp.allclose(alpha_new, alpha_ref, rtol=1e-6)
    assert jnp.allclose(dMag_new, dMag_ref, rtol=1e-6)


def test_planet_contrast_wavelength_agnostic_for_lambertian():
    """Lambertian -> contrast independent of wavelength."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p = _single_planet()
    t_jd = jnp.array([0.0])
    c_blue = p.contrast(SOLVER, jnp.array([400.0]), t_jd, star=star)
    c_red = p.contrast(SOLVER, jnp.array([800.0]), t_jd, star=star)
    assert jnp.allclose(c_blue, c_red, rtol=1e-6)


def test_planet_spec_flux_density_equals_contrast_times_star():
    """spec_flux_density equals contrast times star.spec_flux_density."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p = _single_planet()
    t_jd = jnp.array([0.0])
    wl = jnp.array([600.0])
    c = p.contrast(SOLVER, wl, t_jd, star=star)
    f = p.spec_flux_density(SOLVER, wl, t_jd, star=star)
    assert jnp.allclose(f, c * 1e9, rtol=1e-6)


def test_planet_mean_anomaly_range():
    """mean_anomaly returns (K, T) in [0, 360)."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p = _single_planet()
    M = p.mean_anomaly(jnp.array([0.0, 100.0, 365.0]), star=star)
    # Shape (K=1, T=3), and the values should be in [0, 360)
    assert M.shape == (1, 3)
    assert jnp.all(M >= 0.0)
    assert jnp.all(M < 360.0)
