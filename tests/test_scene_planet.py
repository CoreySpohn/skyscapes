"""scene.Planet -- composes AbstractOrbit + AbstractPhysicalModel."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Rearth2AU
from orbix.equations.orbit import mean_motion, period_n
from orbix.equations.phase import lambert_phase_exact
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit

from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet, System

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
    """Pin Lambert dMag to the raw away-axis phase, across orbital epochs.

    Green (1985), Spherical Astronomy, ch. 17; Brandt et al. (2021),
    AJ 162, 186: raw z is away. Brown (2005), ApJ 624, 1010, Eq. 4:
    the Lambert law uses the standard star-planet-observer angle.
    """
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
    # Raw orbix z points away: cos(alpha) = z / r.
    cos_a = r_AU[:, 2, :] / dist_AU
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


def _circular_planet(i_rad, M0_rad=0.0, W_rad=0.0):
    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([W_rad]),
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


def _quarter_period_times(planet):
    """Epochs with M = 90 and 270 deg for a circular orbit starting at M = 0."""
    period = period_n(mean_motion(planet.orbit.a_AU, G * _STAR.Ms_kg))[0]
    return period * jnp.array([0.25, 0.75])


def test_phase_angle_and_sky_position_at_known_anomalies():
    """Pin the away-axis sign and standard phase at M = 90 and 270 deg.

    Green (1985), Spherical Astronomy, ch. 17, and Brandt et al. (2021),
    AJ 162, 186 give (north, east, away); the right-handed sky chart is (east, north,
    away). Brown (2005), ApJ 624, 1010, Eq. 4 uses zero at full phase.
    """
    p = _circular_planet(jnp.deg2rad(60.0))
    r, alpha, dist = p.propagate(SOLVER, _quarter_period_times(p), star=_STAR)
    np.testing.assert_allclose(jnp.rad2deg(alpha[0]), [30.0, 150.0], atol=1e-6)
    np.testing.assert_allclose(
        r[0],
        [[0.5, -0.5], [0.0, 0.0], [np.sqrt(3) / 2, -np.sqrt(3) / 2]],
        atol=1e-6,
    )
    np.testing.assert_allclose(dist, 1.0, atol=1e-8)
    np.testing.assert_allclose(jnp.cos(alpha), r[:, 2, :] / dist, atol=1e-8)


@pytest.mark.parametrize("observable", ["contrast", "dMag", "flux"])
def test_lambert_brightness_at_mean_anomaly_90(observable):
    """Pin gibbous brightness, including every public photometric consumer.

    Sobolev (1975), Light Scattering in Planetary Atmospheres; Brown (2005),
    ApJ 624, 1010, Eq. 4: Phi(30 deg) = 0.88084278 for a Lambert sphere.
    """
    p = _circular_planet(jnp.deg2rad(60.0))
    times = _quarter_period_times(p)[:1]
    alpha = np.pi / 6
    phase_factor = (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi
    assert phase_factor == pytest.approx(0.88084278, abs=1e-8)
    expected_contrast = 0.3 * Rearth2AU**2 * phase_factor
    if observable == "contrast":
        actual = p.contrast(SOLVER, jnp.array(600.0), times, star=_STAR)
        expected = expected_contrast
    elif observable == "dMag":
        _, actual = p.alpha_dMag(SOLVER, times, star=_STAR)
        expected = -2.5 * np.log10(expected_contrast)
    else:
        actual = p.spec_flux_density(SOLVER, jnp.array(600.0), times, star=_STAR)
        expected = expected_contrast * 1e9
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0)


@pytest.mark.parametrize(
    ("node_deg", "mean_anomaly_deg", "east_north_AU"),
    [(0.0, 90.0, [0.5, 0.0]), (0.0, 0.0, [0.0, 1.0]), (90.0, 0.0, [1.0, 0.0])],
)
def test_position_arcsec_is_east_then_north(node_deg, mean_anomaly_deg, east_north_AU):
    """Pin Delta alpha* = Delta RA cos Dec, then Delta Dec, including nodes.

    Green (1985), Spherical Astronomy, ch. 17; Brandt et al. (2021),
    AJ 162, 186: Omega is measured north through east.
    """
    p = _circular_planet(jnp.deg2rad(60.0), W_rad=jnp.deg2rad(node_deg))
    times = _quarter_period_times(p)[:1] * (mean_anomaly_deg / 90.0)
    expected = np.asarray(east_north_AU) / float(_STAR.dist_pc)
    np.testing.assert_allclose(
        p.position_arcsec(SOLVER, times, star=_STAR)[:, 0, 0],
        expected,
        atol=1e-8,
    )
    system = System(star=_STAR, planets=(p,), trig_solver=SOLVER)
    np.testing.assert_allclose(system.positions(times)[:, 0, 0], expected, atol=1e-8)


def test_edge_on_superior_conjunction_is_brightest():
    """Pin superior conjunction to M = 90 deg, behind the star in the sky chart.

    Green (1985), Spherical Astronomy, ch. 17; Brandt et al. (2021),
    AJ 162, 186: positive raw away-z at M = 90 deg for Omega = omega = 0.
    Brown (2005), ApJ 624, 1010, Eq. 4: full phase is brightest.
    """
    p = _circular_planet(jnp.deg2rad(89.0))
    times = _quarter_period_times(p)
    r, alpha, _ = p.propagate(SOLVER, times, star=_STAR)
    _, dMag = p.alpha_dMag(SOLVER, times, star=_STAR)
    np.testing.assert_allclose(jnp.rad2deg(alpha[0]), [1.0, 179.0], atol=1e-6)
    # Positive z is behind the star in the away chart.
    assert r[0, 2, 0] > 0 > r[0, 2, 1]
    assert dMag[0, 0] < 23.5
    assert dMag[0, 1] > dMag[0, 0] + 1.0


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


def test_planet_propagate_passes_orbix_through_exactly():
    """Positions, phase and distance share orbix's public frame unchanged."""
    p = _single_planet()
    times = jnp.array([0.0, 42.0, 200.0])
    expected = p.orbit.propagate(SOLVER, times, Ms_kg=_STAR.Ms_kg)
    actual = p.propagate(SOLVER, times, star=_STAR)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)
