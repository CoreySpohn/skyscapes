"""ExoVista states retain their sky positions and known orbital elements."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.constants import J2000_JD, G, Msun2kg, d2s, yr2s
from orbix.equations.orbit import AB_matrices, mean_motion
from orbix.equations.propagation import single_r_v
from orbix.kepler import markley_solve_trig, solve_trig

from skyscapes.io import from_exovista


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
    planet[:, 9:12] = np.stack([-r[1], r[0], -r[2]], axis=-1)
    planet[:, 12:15] = np.stack([-v[1], v[0], -v[2]], axis=-1) * (yr2s / d2s)
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
    """Both state vectors must use the orbital-element axes on import."""
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
