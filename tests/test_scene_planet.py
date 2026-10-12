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

    r_AU, _, dist_AU = orbit.propagate(SOLVER, t_jd, Ms_kg=Ms_kg)
    # Independent phase angle: cos(alpha) = -z / r with +z toward the observer.
    cos_a = -r_AU[:, 2, :] / dist_AU
    phase = lambert_phase_exact(cos_a, jnp.sqrt(1.0 - cos_a**2))
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


def _circular_planet(i_rad, M0_rad=0.0):
    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([i_rad]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([M0_rad]),
        t0_d=jnp.array([0.0]),
    )
    return Planet(
        Rp_Rearth=jnp.array([1.0]),
        Mp_Mearth=jnp.array([1.0]),
        orbit=orbit,
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3])),
    )


_STAR = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)


def _sample_orbit(planet, n=361):
    """Positions, phase angle and dMag over one period (K=1)."""
    t_jd = jnp.linspace(0.0, 365.25, n)
    r_AU, alpha, _ = planet.propagate(SOLVER, t_jd, star=_STAR)
    _, dMag = planet.alpha_dMag(SOLVER, t_jd, star=_STAR)
    return r_AU[0], alpha[0], dMag[0]


def test_phase_angle_is_arccos_minus_z_over_r():
    """Phase angle is star-planet-observer: cos(alpha) = -z / r, +z to observer."""
    p = _circular_planet(jnp.deg2rad(60.0), 0.7)
    r, alpha, _ = _sample_orbit(p)
    cos_ref = -r[2] / jnp.linalg.norm(r, axis=0)
    assert jnp.allclose(jnp.cos(alpha), cos_ref, atol=1e-5)


def test_edge_on_superior_conjunction_is_brightest():
    """Full phase (planet behind the star, z < 0) is brightest, new phase faintest."""
    p = _circular_planet(jnp.deg2rad(89.0))
    r, alpha, dMag = _sample_orbit(p)
    z = r[2]
    sup, inf = jnp.argmin(z), jnp.argmax(z)
    assert alpha[sup] < 0.05 and alpha[inf] > jnp.pi - 0.05
    assert dMag[sup] == jnp.nanmin(dMag)
    assert dMag[sup] < 23.5  # near-full Lambert phase at 1 AU, 10 pc, Ag 0.3
    assert not (dMag[inf] <= dMag[sup])  # faintest (or NaN) at inferior conjunction
    assert jnp.nanmax(dMag) > dMag[sup] + 1.0


def test_quadrature_phase_angle_is_half_pi():
    """With the planet on the sky plane (z = 0) the phase angle is pi / 2."""
    p = _circular_planet(jnp.deg2rad(89.0))
    r, alpha, dMag = _sample_orbit(p)
    k = int(jnp.argmin(jnp.abs(r[2])))
    assert abs(float(alpha[k]) - jnp.pi / 2) < 0.03
    assert abs(float(dMag[k]) - 24.40) < 0.05


def test_inclined_phase_angle_range():
    """For i = 60 deg the phase angle spans [90 - i, 90 + i] = [30, 150] deg."""
    p = _circular_planet(jnp.deg2rad(60.0))
    _, alpha, _ = _sample_orbit(p, n=1461)
    assert abs(float(jnp.rad2deg(alpha.min())) - 30.0) < 0.2
    assert abs(float(jnp.rad2deg(alpha.max())) - 150.0) < 0.2
