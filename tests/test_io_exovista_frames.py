"""ExoVista states retain their sky positions and known orbital elements."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.constants import J2000_JD, G, Mearth2kg, Msun2kg, d2s, yr2s
from orbix.equations.orbit import AB_matrices, mean_motion
from orbix.equations.propagation import single_r_v
from orbix.kepler import markley_solve_trig, solve_trig

from skyscapes.io import from_exovista
from skyscapes.io.exovista import _exovista_to_orbix


@pytest.fixture(params=[(38.0, 63.0, 41.0, 27.0), (127.0, 213.0, 302.0, 231.0)])
def known_orbit_file(tmp_path, request):
    """Write float32 body rows in ExoVista's (west, north, toward) axes.

    The star is stationary at the origin to isolate the axis convention.
    A massless planet keeps the state's and propagator's gravitational
    parameters identical without requiring barycentric corrections.
    """
    a, e = 1.3, 0.23
    inc, node, peri, anomaly = np.deg2rad(request.param)
    elapsed_years = np.array([0.0, 0.125, 0.25, 0.5, 0.75], dtype=np.float32)
    elapsed_days = jnp.asarray(elapsed_years.astype(np.float64)) * (yr2s / d2s)
    n = mean_motion(a, G * Msun2kg)
    sin_e, cos_e = solve_trig(anomaly + n * elapsed_days, e)
    A, B = AB_matrices(a, e, inc, node, peri)
    r, v = single_r_v(A, B, e, sin_e, cos_e, n)

    wavelengths = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    primary = fits.PrimaryHDU(wavelengths)
    primary.header["VERSION"] = 2.6
    hdus = [
        primary,
        fits.ImageHDU(wavelengths),
        fits.ImageHDU(),
        fits.ImageHDU(),
        fits.ImageHDU(np.zeros((len(elapsed_years), 0, 4), dtype=np.float32)),
    ]
    star = np.zeros((len(elapsed_years), 19), dtype=np.float32)
    star[:, 0] = elapsed_years
    star[:, 16:] = 1.0
    hdus.append(
        fits.ImageHDU(
            star,
            header=fits.Header(
                {
                    "MASS": 1.0,
                    "DIST": 10.0,
                    "RSTAR": 1.0,
                    "ANGDIAM": 0.93,
                    "SPECUNIT": "Jy",
                    "I": 54.0,
                    "PA": 32.0,
                }
            ),
        )
    )
    planet = np.zeros_like(star)
    planet[:, 0] = elapsed_years
    # Orbix rows are east/north/away; the FITS chart is west/north/toward.
    planet[:, 9:12] = np.stack([-r[0], r[1], -r[2]], axis=-1)
    planet[:, 12:15] = np.stack([-v[0], v[1], -v[2]], axis=-1) * (yr2s / d2s)
    planet[:, 16:] = 1e-10
    hdus.append(
        fits.ImageHDU(planet, header=fits.Header({"A": a, "E": e, "M": 0.0, "R": 1.0}))
    )
    path = tmp_path / "known-orbit.fits"
    fits.HDUList(hdus).writeto(path)
    return path, (a, e, inc, node, peri, anomaly)


def test_exovista_sky_positions_round_trip(known_orbit_file):
    """FITS west/north positions load as east/north at every recorded epoch."""
    path, _ = known_orbit_file
    system = from_exovista(path, load_disk=False)
    with fits.open(path) as hdul:
        rows = hdul[6].data.astype(np.float64)
        star_rows = hdul[5].data.astype(np.float64)
    times = J2000_JD + rows[:, 0] * (yr2s / d2s)
    relative = rows[:, 9:12] - star_rows[:, 9:12]
    expected = np.stack([-relative[:, 0], relative[:, 1]]) / system.star.dist_pc
    positions = system.planets[0].position_arcsec(
        markley_solve_trig, times, star=system.star
    )
    # Use the exact Kepler solver to isolate float32 FITS storage error from
    # grid interpolation error. Allow five float32 epsilons at the orbit scale.
    orbit_scale = float(system.planets[0].orbit.a_AU[0]) / system.star.dist_pc
    tolerance = 5 * np.finfo(np.float32).eps * orbit_scale
    np.testing.assert_allclose(positions[:, 0], expected, rtol=0, atol=tolerance)


def test_exovista_recovers_known_elements(known_orbit_file):
    """Preserve header a/e and recover i, Omega, omega, M from both states."""
    path, expected = known_orbit_file
    orbit = from_exovista(path, load_disk=False).planets[0].orbit
    actual = np.array(
        [
            orbit.a_AU[0],
            orbit.e[0],
            orbit.i_rad[0],
            orbit.W_rad[0],
            orbit.w_rad[0],
            orbit.M0_rad[0],
        ]
    )
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=1e-7)
    # Compare periodic angles modulo a full turn, including negative atan2 output.
    difference = np.angle(np.exp(1j * (actual[2:] - expected[2:])))
    np.testing.assert_allclose(difference, 0.0, atol=2e-6)


@pytest.mark.parametrize("version", [2.33, 2.5, 2.6])
def test_exovista_recovers_known_elements_with_stellar_reflex(
    known_orbit_file, version
):
    """Preserve header a/e; recover star-relative i, Omega, omega, M from states.

    Both layouts store float32 barycentric positions and velocities.
    """
    path, expected = known_orbit_file
    a, e, inc, node, peri, anomaly = expected
    mass_earth = 317.8
    mass_ratio = mass_earth * Mearth2kg / Msun2kg
    with fits.open(path, mode="update") as hdul:
        star, planet = hdul[5].data, hdul[6].data
        elapsed_days = jnp.asarray(planet[:, 0].astype(np.float64)) * (yr2s / d2s)
        n = mean_motion(a, G * (Msun2kg + mass_earth * Mearth2kg))
        sin_e, cos_e = solve_trig(anomaly + n * elapsed_days, e)
        A, B = AB_matrices(a, e, inc, node, peri)
        r, v = single_r_v(A, B, e, sin_e, cos_e, n)
        # ExoVista's generate_scene.py writes barycentric west/north/toward
        # states for both bodies, including the star's reflex motion.
        relative_r = np.stack([-r[0], r[1], -r[2]], axis=-1)
        relative_v = np.stack([-v[0], v[1], -v[2]], axis=-1) * (yr2s / d2s)
        # Orbix rows are east/north/away; the FITS chart is west/north/toward.
        planet[:, 9:12] = relative_r / (1 + mass_ratio)
        planet[:, 12:15] = relative_v / (1 + mass_ratio)
        star[:, 9:12] = -mass_ratio * relative_r / (1 + mass_ratio)
        star[:, 12:15] = -mass_ratio * relative_v / (1 + mass_ratio)
        hdul[6].header["M"] = mass_earth
        hdul[6].header["R"] = 11.2
        hdul[0].header["VERSION"] = version
        if version < 2.5:
            del hdul[4]  # Legacy files have no starspot extension.

    orbit = from_exovista(path, load_disk=False, planet_spectra="time").planets[0].orbit
    actual = np.array(
        [
            orbit.a_AU[0],
            orbit.e[0],
            orbit.i_rad[0],
            orbit.W_rad[0],
            orbit.w_rad[0],
            orbit.M0_rad[0],
        ]
    )
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=1e-7)
    # Float32 state storage limits angle recovery; wrap atan2's signed angles.
    difference = np.angle(np.exp(1j * (actual[2:] - expected[2:])))
    np.testing.assert_allclose(difference, 0.0, rtol=0, atol=2e-6)


def test_exovista_phase_matches_fits_observer_direction(known_orbit_file):
    """A planet toward the observer has a crescent, not a gibbous phase."""
    path, _ = known_orbit_file
    system = from_exovista(path, load_disk=False, planet_spectra="phase")
    with fits.open(path) as hdul:
        rows = hdul[6].data.astype(np.float64)
    times = J2000_JD + rows[:, 0] * (yr2s / d2s)
    expected_cosine = -rows[:, 11] / np.linalg.norm(rows[:, 9:12], axis=1)
    _, phase, _ = system.planets[0].propagate(
        system.trig_solver, times, star=system.star
    )
    np.testing.assert_allclose(np.cos(phase[0]), expected_cosine, atol=2e-6)


def test_exovista_phase_model_preserves_analytic_light_curve(known_orbit_file):
    """Phase regridding preserves a contrast linear in the observer phase angle."""
    path, _ = known_orbit_file
    with fits.open(path, mode="update") as hdul:
        rows = hdul[6].data
        relative = rows[:, 9:12].astype(np.float64) - hdul[5].data[:, 9:12]
        beta = np.arccos(-relative[:, 2] / np.linalg.norm(relative, axis=1))
        # A linear curve is exact under phase interpolation, independent of
        # the fitted orbit or the placement of the loader's phase grid.
        contrast = 1e-10 * (1 + beta / np.pi)
        rows[:, 16:] = contrast[:, None] * np.array([1.0, 2.0, 3.0])
        times = J2000_JD + rows[:, 0].astype(np.float64) * (yr2s / d2s)
    system = from_exovista(path, load_disk=False, planet_spectra="phase")
    np.testing.assert_allclose(
        system.contrasts(600.0, times)[0], 2 * contrast, rtol=2e-6, atol=0
    )
    expected_flux = 2 * contrast * system.star.spec_flux_density(600.0, times)
    np.testing.assert_allclose(
        system.planet_flux_densities(600.0, times)[0], expected_flux, rtol=2e-6, atol=0
    )


def test_exovista_frame_conversion_is_proper():
    """West/north/toward to east/north/away is a proper half-turn about north."""
    rotation = np.asarray(_exovista_to_orbix(jnp.eye(3)))
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_array_equal(rotation, np.diag([-1.0, 1.0, -1.0]))
    np.testing.assert_array_equal(rotation @ rotation.T, np.eye(3))
