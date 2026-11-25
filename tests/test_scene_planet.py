"""Photometric planets and scene-level geometry/photometry contracts."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Mearth2kg, Msun2kg, Rearth2AU, pc2AU, rad2arcsec
from orbix.equations.orbit import mean_motion, period_n

from skyscapes.physical_model import LambertianPhysicalModel, lambert_phase
from skyscapes.scene import FlatStar, Planet, Scene

from .conftest import two_body_reference


def _single_scene(*, mass=1.0, i_rad=jnp.pi / 3, M0_rad=jnp.pi / 4, W_rad=0.0):
    return Scene.from_classical(
        FlatStar(1e9),
        (Planet(jnp.ones(1), LambertianPhysicalModel(jnp.array([0.3]))),),
        star_mass_Msun=1.989e30 / Msun2kg,
        planet_mass_Msun=jnp.array([mass]) * Mearth2kg / Msun2kg,
        dist_pc=10.0,
        a_AU=jnp.ones(1),
        e=0.0,
        W_rad=W_rad,
        i_rad=i_rad,
        w_rad=0.0,
        M0_rad=M0_rad,
    )


def _reference_elements(scene):
    return dict(
        **scene.dynamics.orbits.to_classical(), epoch_d=scene.dynamics.orbits.epoch_d
    )


def _geometry(scene, times):
    eph = scene.ephemeris(times)
    return (
        (eph.position_AU[1:] - eph.position_AU[:1]).transpose(0, 2, 1),
        eph.phase_angle_rad(jnp.array([1]), 0),
        eph.distance_AU(jnp.array([1]), 0),
    )


def test_planet_n_planets_property():
    """Planet.n_planets reflects the K dimension of intrinsic params."""
    p = _single_scene()
    assert p.planets[0].n_planets == 1


def test_planet_position_shape():
    """Scene.positions returns shape (2, K, T)."""
    p = _single_scene()
    ra_dec = p.positions(jnp.array([0.0]))
    # (2, K, T) per contract
    assert ra_dec.shape == (2, 1, 1)


def test_planet_alpha_dMag_matches_orbix():
    """Pin Lambert dMag to the raw away-axis phase, across orbital epochs.

    Green (1985), Spherical Astronomy, ch. 17; Brandt et al. (2021),
    AJ 162, 186: raw z is away. Brown (2005), ApJ 624, 1010, Eq. 4:
    the Lambert law uses the standard star-planet-observer angle.
    """
    p = _single_scene()
    time_d = jnp.array([0.0, 100.0, 200.0])

    alpha_new, dMag_new = p.alpha_dMag(time_d)

    elements = dict(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([jnp.pi / 3]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([jnp.pi / 4]),
        epoch_d=jnp.array([0.0]),
    )
    # The planet adds its own mass to the central mass (two-body mu).
    total_mass_kg = jnp.array([1.989e30]) + p.planet_mass_Msun * Msun2kg
    dist_pc = jnp.array([10.0])
    Ag = jnp.array([0.3])
    Rp_AU = jnp.array([1.0]) * Rearth2AU

    reference = two_body_reference(
        time_d, **elements, star_mass_kg=total_mass_kg, dist_pc=dist_pc
    )
    r_AU, star_planet_distance_AU = reference.position_AU, reference.distance_AU
    # Raw orbix z points away: cos(beta) = z / r.
    cos_a = r_AU[:, 2, :] / star_planet_distance_AU
    phase = lambert_phase(jnp.arccos(cos_a))
    log_arg = Ag[:, None] * (Rp_AU[:, None] / star_planet_distance_AU) ** 2 * phase
    dMag_ref = -2.5 * jnp.log10(log_arg + jnp.finfo(log_arg.dtype).tiny)
    alpha_ref = reference.separation_arcsec

    assert jnp.allclose(alpha_new, alpha_ref, rtol=1e-6)
    assert jnp.allclose(dMag_new, dMag_ref, rtol=1e-6)


def test_planet_contrast_wavelength_agnostic_for_lambertian():
    """Lambertian -> contrast independent of wavelength."""
    p = _single_scene()
    time_d = jnp.array([0.0])
    c_blue = p.contrasts(jnp.array(400.0), time_d)
    c_red = p.contrasts(jnp.array(800.0), time_d)
    assert jnp.allclose(c_blue, c_red, rtol=1e-6)


def test_planet_photon_flux_density_formula_phase_and_jax():
    """Fails on inverted phase, photon units, or dropped quadrature-node axes.

    Brown (2005) Eq. 4 gives Ag*(Rp/r)^2*(sin b+(pi-b)*cos b)/pi.
    At M=90 deg, i=60 deg the planet is on the far (superior) side with
    phase 30 deg; M=0 is quadrature. 3e-13 relative bounds float64 orbit
    rotation and photon algebra, with atol=0 for tiny planetary fluxes.
    """
    p = _circular_scene(jnp.pi / 3)
    times = jnp.array([0.0, _quarter_period_times(p)[0]])
    wave = jnp.array([[500.0, 550.0], [600.0, 700.0]])
    eph = p.ephemeris(times)
    model = p.planets[0].physical_model
    geometry = dict(
        phase_angle_rad=eph.phase_angle_rad(jnp.array([1]), 0),
        star_planet_distance_AU=eph.distance_AU(jnp.array([1]), 0),
        Rp_Rearth=p.planets[0].Rp_Rearth,
        star=p.star,
        dist_pc=p.dist_pc,
    )

    def evaluate(w):
        return model.photon_flux_density(w, times, **geometry)

    beta = np.array([np.pi / 2, np.pi / 6])
    phase = (np.sin(beta) + (np.pi - beta) * np.cos(beta)) / np.pi
    # Earth equatorial radius and AU in metres, independently typed.
    contrast = 0.3 * (6.3781e6 / 1.495978707e11) ** 2 * phase
    expected = np.broadcast_to(contrast[None, :, None, None] * 1e9, (1, 2, 2, 2))
    actual = evaluate(wave)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)
    assert np.all(np.asarray(actual[:, 1]) > np.asarray(actual[:, 0]))
    np.testing.assert_allclose(
        eqx.filter_jit(evaluate)(wave), actual, rtol=3e-13, atol=0
    )
    loop = jnp.stack([evaluate(w) for w in wave.ravel()])
    np.testing.assert_allclose(
        jax.vmap(evaluate)(wave.ravel()), loop, rtol=3e-13, atol=0
    )
    np.testing.assert_allclose(
        p.planet_flux_densities(550.0, times), expected[:, :, 0, 0], rtol=3e-13, atol=0
    )


def test_scene_reference_mean_anomaly_range():
    """mean_anomaly returns (K, T) in [0, 360)."""
    p = _single_scene()
    elements = _reference_elements(p)
    times = jnp.array([0.0, 100.0, 365.0])
    M = jnp.rad2deg(
        (elements["M0_rad"][:, None] + p.dynamics.orbits.n_rad_per_d[:, None] * times)
        % (2 * jnp.pi)
    )
    # Shape (K=1, T=3), and the values should be in [0, 360)
    assert M.shape == (1, 3)
    assert jnp.all(M >= 0.0)
    assert jnp.all(M < 360.0)


def _circular_scene(i_rad, M0_rad=0.0, W_rad=0.0):
    # Zero mass fixes the hand-derived periods to the stellar mass alone.
    return _single_scene(mass=0.0, i_rad=i_rad, M0_rad=M0_rad, W_rad=W_rad)


def _sample_orbit(planet, n=361):
    """Positions, phase angle and dMag over one period (K=1)."""
    time_d = jnp.linspace(0.0, 365.25, n)
    r_AU, beta, _ = _geometry(planet, time_d)
    _, dMag = planet.alpha_dMag(time_d)
    return r_AU[0], beta[0], dMag[0]


def _quarter_period_times(planet):
    """Epochs with M = 90 and 270 deg for a circular orbit starting at M = 0."""
    period = period_n(
        mean_motion(planet.dynamics.orbits.a_AU, G * planet.star_mass_Msun * Msun2kg)
    )[0]
    return period * jnp.array([0.25, 0.75])


def test_phase_angle_and_sky_position_at_known_anomalies():
    """Pin the away-axis sign and standard phase at M = 90 and 270 deg.

    Green (1985), Spherical Astronomy, ch. 17, and Brandt et al. (2021),
    AJ 162, 186 give (north, east, away); the right-handed sky chart is (east, north,
    away). Brown (2005), ApJ 624, 1010, Eq. 4 uses zero at full phase.
    """
    p = _circular_scene(jnp.deg2rad(60.0))
    r, beta, dist = _geometry(p, _quarter_period_times(p))
    np.testing.assert_allclose(jnp.rad2deg(beta[0]), [30.0, 150.0], atol=1e-6)
    np.testing.assert_allclose(
        r[0],
        [[0.5, -0.5], [0.0, 0.0], [np.sqrt(3) / 2, -np.sqrt(3) / 2]],
        atol=1e-6,
    )
    np.testing.assert_allclose(dist, 1.0, atol=1e-8)
    np.testing.assert_allclose(jnp.cos(beta), r[:, 2, :] / dist, atol=1e-8)


@pytest.mark.parametrize("observable", ["contrast", "dMag", "flux"])
def test_lambert_brightness_at_mean_anomaly_90(observable):
    """Pin gibbous brightness, including every public photometric consumer.

    Sobolev (1975), Light Scattering in Planetary Atmospheres; Brown (2005),
    ApJ 624, 1010, Eq. 4: Phi(30 deg) = 0.88084278 for a Lambert sphere.
    """
    p = _circular_scene(jnp.deg2rad(60.0))
    times = _quarter_period_times(p)[:1]
    beta = np.pi / 6
    phase_factor = (np.sin(beta) + (np.pi - beta) * np.cos(beta)) / np.pi
    assert phase_factor == pytest.approx(0.88084278, abs=1e-8)
    expected_contrast = 0.3 * Rearth2AU**2 * phase_factor
    if observable == "contrast":
        actual = p.contrasts(jnp.array(600.0), times)
        expected = expected_contrast
    elif observable == "dMag":
        _, actual = p.alpha_dMag(times)
        expected = -2.5 * np.log10(expected_contrast)
    else:
        actual = p.planet_flux_densities(jnp.array(600.0), times)
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
    p = _circular_scene(jnp.deg2rad(60.0), W_rad=jnp.deg2rad(node_deg))
    times = _quarter_period_times(p)[:1] * (mean_anomaly_deg / 90.0)
    expected = np.asarray(east_north_AU) / float(p.dist_pc)
    np.testing.assert_allclose(
        p.positions(times)[:, 0, 0],
        expected,
        atol=1e-8,
    )
    system = Scene(p.star, p.planets, p.dynamics)
    np.testing.assert_allclose(system.positions(times)[:, 0, 0], expected, atol=1e-8)


def test_edge_on_superior_conjunction_is_brightest():
    """Pin superior conjunction to M = 90 deg, behind the star in the sky chart.

    Green (1985), Spherical Astronomy, ch. 17; Brandt et al. (2021),
    AJ 162, 186: positive raw away-z at M = 90 deg for Omega = omega = 0.
    Brown (2005), ApJ 624, 1010, Eq. 4: full phase is brightest.
    """
    p = _circular_scene(jnp.deg2rad(89.0))
    times = _quarter_period_times(p)
    r, beta, _ = _geometry(p, times)
    _, dMag = p.alpha_dMag(times)
    np.testing.assert_allclose(jnp.rad2deg(beta[0]), [1.0, 179.0], atol=1e-6)
    # Positive z is behind the star in the away chart.
    assert r[0, 2, 0] > 0 > r[0, 2, 1]
    assert dMag[0, 0] < 23.5
    assert dMag[0, 1] > dMag[0, 0] + 1.0


def test_quadrature_phase_angle_is_half_pi():
    """With the planet on the sky plane (z = 0) the phase angle is pi / 2."""
    p = _circular_scene(jnp.deg2rad(89.0))
    r, beta, dMag = _sample_orbit(p)
    k = int(jnp.argmin(jnp.abs(r[2])))
    assert abs(float(beta[k]) - jnp.pi / 2) < 0.03
    assert abs(float(dMag[k]) - 24.40) < 0.05


def test_inclined_phase_angle_range():
    """For i = 60 deg the phase angle spans [90 - i, 90 + i] = [30, 150] deg."""
    p = _circular_scene(jnp.deg2rad(60.0))
    _, beta, _ = _sample_orbit(p, n=1461)
    assert abs(float(jnp.rad2deg(beta.min())) - 30.0) < 0.2
    assert abs(float(jnp.rad2deg(beta.max())) - 150.0) < 0.2


def test_scene_ephemeris_passes_orbix_through_exactly():
    """States and distances preserve orbix's public frame unchanged.

    Basis: 16 float64 eps of each field's largest magnitude. The reference is
    the eager op-by-op solve and the scene solves through one compiled program,
    so the two differ by fusion and fused multiply-add rounding of the Kepler
    solve and the frame rotation (a chain of about a dozen operations per
    element), not bitwise. This machine sees about 3 eps; a changed frame,
    sign or unit moves a field by orders of magnitude more.
    """
    p = _single_scene()
    times = jnp.array([0.0, 42.0, 200.0])
    expected = p.dynamics.solve(times)
    actual = p.ephemeris(times)
    eps = np.finfo(np.float64).eps
    for name in ("position_AU", "velocity_AU_per_d", "dist_pc"):
        want = np.asarray(getattr(expected, name))
        np.testing.assert_allclose(
            getattr(actual, name),
            want,
            rtol=16 * eps,
            atol=16 * eps * np.max(np.abs(want)),
        )


@pytest.mark.parametrize("mass", [1.0, float("nan"), -1.0, float("inf"), 0.0])
def test_scene_mass_and_observables(mass):
    """Fails if mass sanitization changes existing positions or contrasts.

    Truth is the independent NumPy two-body reference with Newton iteration.
    Mass sanitization remains exact; positions and contrast allow 3e-13
    relative float64 basis/epoch roundoff (3e-13 arcsec absolute for position).
    The system no longer includes grid interpolation error.
    """
    p = _single_scene(mass=mass)
    times = jnp.array([0.0, 42.0, 200.0])
    mp = p.planet_mass_Msun * Msun2kg
    old_mass = (p.star_mass_Msun * Msun2kg) + jnp.where(
        jnp.isfinite(mp) & (mp > 0), mp, 0.0
    )
    np.testing.assert_array_equal(
        p.star_mass_Msun + p.planet_mass_Msun,
        [1.989e30 / Msun2kg + (Mearth2kg / Msun2kg if mass == 1 else 0.0)],
    )
    np.testing.assert_array_equal(
        p.planet_mass_Msun, [Mearth2kg / Msun2kg if mass == 1 else 0]
    )
    reference = two_body_reference(
        times, **_reference_elements(p), star_mass_kg=old_mass
    )
    r, phase, dist = (
        reference.position_AU,
        reference.phase_angle_rad,
        reference.distance_AU,
    )
    scale = rad2arcsec / (p.dist_pc * pc2AU)
    old_position = jnp.stack([r[:, 0, :] * scale, r[:, 1, :] * scale])
    old_contrast = p.planets[0].physical_model.contrast_at_time(
        phase_angle_rad=phase,
        star_planet_distance_AU=dist,
        wavelength_nm=jnp.asarray(600.0),
        Rp_Rearth=p.planets[0].Rp_Rearth,
        time_d=times,
        star=p.star,
        dist_pc=p.dist_pc,
    )
    system = Scene(p.star, p.planets, p.dynamics)
    np.testing.assert_allclose(
        system.positions(times), old_position, rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        system.contrasts(jnp.asarray(600.0), times), old_contrast, rtol=3e-13, atol=0
    )


@pytest.mark.parametrize("node_deg, expected", [(90.0, 90.0), (0.0, 0.0)])
def test_position_angle_cardinal_anchors(node_deg, expected):
    """Fails on an east/north angle swap; geometric axes give 90/0 degrees.

    At the ascending node PA equals the node angle; 2e-14 deg absolute
    tolerance allows float64 trigonometric roundoff, not a sign convention.
    """
    p = _circular_scene(jnp.pi / 3, W_rad=jnp.deg2rad(node_deg))
    times = jnp.zeros(1)
    np.testing.assert_allclose(p.position_angles(times), expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        Scene(p.star, p.planets, p.dynamics).position_angles(times),
        expected,
        rtol=0,
        atol=2e-14,
    )


@pytest.mark.parametrize("observable", ["star_reflex", "star_radial_velocity"])
def test_unknown_mass_zero_observables_and_orbit_gradients(observable):
    """Fails if unknown mass leaks NaNs into a stellar signal or its gradient.

    A zero-mass body exerts zero force: both values and all seven orbital
    derivatives must be exactly zero (no numerical tolerance needed).
    """
    p = _single_scene(mass=jnp.nan)
    times = jnp.array([0.0, 42.0])

    def signal(orbit):
        varied = eqx.tree_at(lambda p: p.dynamics.orbits, p, orbit)
        return getattr(varied, observable)(times)

    np.testing.assert_array_equal(signal(p.dynamics.orbits), 0.0)
    gradient = jax.grad(lambda orbit: jnp.sum(signal(orbit)))(p.dynamics.orbits)
    for leaf in jax.tree.leaves(gradient):
        assert jnp.all(jnp.isfinite(leaf))
        np.testing.assert_array_equal(leaf, 0.0)


def _two_planet_scene():
    return Scene.from_classical(
        FlatStar(1e9),
        (Planet(jnp.ones(2), LambertianPhysicalModel(jnp.array([0.3, 0.3]))),),
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.array([1e-6, 1e-6]),
        dist_pc=10.0,
        a_AU=jnp.array([1.0, 2.0]),
        e=0.0,
        W_rad=0.0,
        i_rad=jnp.pi / 3,
        w_rad=0.0,
        M0_rad=jnp.array([0.5, 1.0]),
    )


def _with_scales(scene, star=1.0, planets=(1.0, 1.0)):
    scene = eqx.tree_at(lambda s: s.star.flux_scale, scene, jnp.asarray(star))
    return eqx.tree_at(lambda s: s.planets[0].flux_scale, scene, jnp.asarray(planets))


def test_planet_flux_scale_is_applied_once_beside_the_radius():
    """Fails if a planet's scale is skipped, doubled, shared or not a live leaf.

    Basis: exact equality at a scale of one against the physical model called
    with the same geometry; a power-of-two scale is exact, so the factors are
    compared to rounding of one multiplication (rtol 1e-15). The planet scale
    multiplies contrast and flux of its own planet alone.
    """
    scene = _two_planet_scene()
    assert scene.planets[0].flux_scale.shape == (2,)
    time_d, wave = jnp.array([0.0, 30.0]), jnp.asarray(550.0)
    eph = scene.ephemeris(time_d)
    model = scene.planets[0].physical_model
    direct = model.contrast_at_time(
        phase_angle_rad=eph.phase_angle_rad(eph.scene_planet_indices, 0),
        star_planet_distance_AU=eph.distance_AU(eph.scene_planet_indices, 0),
        wavelength_nm=wave,
        Rp_Rearth=scene.planets[0].Rp_Rearth,
        time_d=time_d,
        star=scene.star,
        dist_pc=10.0,
    )
    base = scene.contrasts(wave, time_d)
    np.testing.assert_array_equal(base, direct)

    flux = scene.planet_flux_densities(wave, time_d)
    scaled = _with_scales(scene, planets=(1.0, 3.0))
    np.testing.assert_allclose(
        scaled.contrasts(wave, time_d), base * jnp.array([[1.0], [3.0]]), rtol=1e-15
    )
    np.testing.assert_allclose(
        scaled.planet_flux_densities(wave, time_d),
        flux * jnp.array([[1.0], [3.0]]),
        rtol=1e-15,
    )

    grad = eqx.filter_grad(lambda s: jnp.sum(s.planet_flux_densities(wave, time_d)))(
        scene
    )
    np.testing.assert_allclose(
        grad.planets[0].flux_scale, jnp.sum(flux, axis=1), rtol=1e-12
    )


def test_star_and_planet_scales_are_bilinear_not_jointly_linear():
    """Fails if the star scale reaches a contrast or the pair is jointly linear.

    The planet flux is the star flux times the contrast, so the flux is
    bilinear in the two scales and the contrast does not see the star scale.
    Rounding of three products bounds the checks at rtol 1e-14; the joint
    first-order prediction 2 + 3 - 1 = 4 misses the product 6 by a third.
    """
    scene = _two_planet_scene()
    time_d, wave = jnp.array([0.0]), jnp.asarray(550.0)

    def flux(star, planet):
        return _with_scales(scene, star, (planet, planet)).planet_flux_densities(
            wave, time_d
        )

    one = flux(1.0, 1.0)
    np.testing.assert_allclose(flux(2.0, 1.0), 2.0 * one, rtol=1e-14)
    np.testing.assert_allclose(flux(1.0, 3.0), 3.0 * one, rtol=1e-14)
    np.testing.assert_allclose(flux(2.0, 3.0), 6.0 * one, rtol=1e-14)
    first_order = flux(2.0, 1.0) + flux(1.0, 3.0) - one
    assert float(jnp.max(jnp.abs(first_order / flux(2.0, 3.0) - 1.0))) > 0.3
    np.testing.assert_array_equal(
        _with_scales(scene, 2.0).contrasts(wave, time_d), scene.contrasts(wave, time_d)
    )


def test_planet_flux_scale_defaults_broadcast_and_is_checked():
    """Fails if a scalar scale does not follow the radii, or a nonfinite one passes."""
    model = LambertianPhysicalModel(jnp.array([0.3, 0.3]))
    assert Planet(jnp.ones(2), model).flux_scale.shape == (2,)
    assert float(jnp.sum(Planet(jnp.ones(2), model, 2.0).flux_scale)) == 4.0
    assert float(Planet(jnp.ones(2), model, -1.0).flux_scale[0]) == -1.0
    with pytest.raises(ValueError, match="finite"):
        Planet(jnp.ones(2), model, jnp.nan)
