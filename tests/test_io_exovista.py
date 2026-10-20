"""skyscapes.io.exovista -- ExoVista FITS to scene.System."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G
from orbix.equations.orbit import mean_motion, period_n

from skyscapes.disk import ExovistaDisk
from skyscapes.io import from_exovista, get_earth_like_planet_indices
from skyscapes.scene import Planet, Star, System


def test_from_exovista_returns_scene_system(fits_fixture):
    """from_exovista returns a scene.System instance."""
    sys_obj = from_exovista(fits_fixture)
    assert isinstance(sys_obj, System)


def test_from_exovista_star_is_spectrum_star(fits_fixture):
    """Loaded star is a Star with Ms_kg populated from FITS."""
    sys_obj = from_exovista(fits_fixture)
    assert isinstance(sys_obj.star, Star)
    assert sys_obj.star.Ms_kg > 0.0


def test_from_exovista_planets_are_tuple(fits_fixture):
    """Planets is a non-empty tuple of scene.Planet."""
    sys_obj = from_exovista(fits_fixture)
    assert isinstance(sys_obj.planets, tuple)
    assert len(sys_obj.planets) > 0
    for p in sys_obj.planets:
        assert isinstance(p, Planet)


def test_from_exovista_planets_coplanar_with_disk(fits_fixture):
    """Loaded planets are coplanar with the disk midplane.

    Each planet's orbital inclination matches the system midplane inclination
    (within the small mutual spread). Regression guard for the double
    sky-rotation bug that tilted every planetary system off its disk.
    """
    sys_obj = from_exovista(fits_fixture)
    disk_inc = sys_obj.midplane_inc_deg
    for p in sys_obj.planets:
        i_deg = float(jnp.rad2deg(p.orbit.i_rad[0]))
        # The disk plane is i == disk_inc (or its 180 - i mirror).
        delta = min(abs(i_deg - disk_inc), abs(i_deg - (180.0 - disk_inc)))
        assert delta < 8.0, (
            f"planet inclination {i_deg:.1f} deg not coplanar with disk "
            f"midplane {disk_inc:.1f} deg"
        )


def test_from_exovista_disk_is_exovista_disk(fits_fixture):
    """Loaded disk is an ExovistaDisk."""
    sys_obj = from_exovista(fits_fixture)
    assert isinstance(sys_obj.disk, ExovistaDisk)


def test_exovista_disk_and_planet_share_east_north_chart(fits_fixture):
    """Image and orbit second moments have the same projected major axis.

    Angles run from +column (east) toward +row (north), modulo 180 deg.
    The demo disk and outer orbit need not align exactly: the planets have
    small mutual inclinations and the brightness weights disk structure.
    The measured axes are 147.34 deg (disk) and 148.46 deg (outer orbit).
    A 3 deg tolerance allows their 1.12 deg spread but rejects the original
    mirrored image at 32.66 deg (64.20 deg from the orbit, modulo 180).
    """
    system = from_exovista(fits_fixture)
    disk = system.disk
    brightness = np.asarray(
        disk.surface_brightness(
            disk.wavelengths_nm[0],
            jnp.asarray(2451545.0),
            jnp.asarray(system.midplane_inc_deg),
            jnp.asarray(system.midplane_pa_deg),
        ),
        dtype=float,
    )
    row, col = np.indices(brightness.shape, dtype=float)

    def major_axis_angle(x, y, weights):
        """Brightness-weighted central second-moment axis, modulo 180 deg."""
        x = x - np.average(x, weights=weights)
        y = y - np.average(y, weights=weights)
        xx = np.average(x * x, weights=weights)
        yy = np.average(y * y, weights=weights)
        xy = np.average(x * y, weights=weights)
        return np.degrees(0.5 * np.arctan2(2 * xy, xx - yy)) % 180

    disk_angle = major_axis_angle(col, row, brightness)
    planet = max(system.planets, key=lambda p: float(p.orbit.a_AU[0]))
    period = period_n(mean_motion(planet.orbit.a_AU, G * system.star.Ms_kg))[0]
    times = planet.orbit.t0_d[0] + period * jnp.linspace(0.0, 1.0, 1024, endpoint=False)
    east, north = np.asarray(
        planet.position_arcsec(system.trig_solver, times, star=system.star)
    )[:, 0, :]
    orbit_angle = major_axis_angle(east, north, np.ones_like(east))
    difference = abs((disk_angle - orbit_angle + 90) % 180 - 90)
    assert difference < 3.0, (
        f"disk major axis {disk_angle:.2f} deg vs planet orbit "
        f"{orbit_angle:.2f} deg: {difference:.2f} deg apart in the east/north chart"
    )

    # The axis angle is blind to a row-for-column mistake (both mirrors map
    # theta to 180 - theta). The near, forward-scattering half of the disk
    # lies on the side of the major axis where the orbit is toward the
    # observer, which a row flip would move across the axis. Measured: the
    # brightness centroid sits 10.6 px toward that side; a row flip gives
    # -10.6 px.
    r_sky, _, _ = planet.propagate(system.trig_solver, times, star=system.star)
    r_sky = np.asarray(r_sky)[0]
    # Toward the observer is negative z in the away chart.
    near = r_sky[:2, r_sky[2] < 0].mean(axis=1)
    near = near / np.hypot(*near)
    centre = (np.array(brightness.shape, dtype=float) - 1) / 2
    offset = np.average(
        (col - centre[1]) * near[0] + (row - centre[0]) * near[1], weights=brightness
    )
    assert offset > 2.0, (
        f"disk near side lies {offset:.2f} px from the orbit's near side"
    )


def test_from_exovista_load_disk_false_skips_disk(fits_fixture):
    """load_disk=False leaves System.disk as None."""
    sys_obj = from_exovista(fits_fixture, load_disk=False)
    assert sys_obj.disk is None


def test_from_exovista_only_earths_filters(fits_fixture):
    """only_earths=True keeps only Earth-like planets."""
    sys_obj_all = from_exovista(fits_fixture)
    sys_obj_earths = from_exovista(fits_fixture, only_earths=True)
    assert len(sys_obj_earths.planets) <= len(sys_obj_all.planets)


def test_from_exovista_respects_planet_indices(fits_fixture):
    """Explicit planet_indices controls the set of planets loaded."""
    sys_obj = from_exovista(fits_fixture, planet_indices=[0, 2])
    assert len(sys_obj.planets) == 2


def test_get_earth_like_planet_indices_returns_list(fits_fixture):
    """Earth-like filter produces a list of integer planet indices."""
    idx = get_earth_like_planet_indices(fits_fixture)
    assert isinstance(idx, list)
    assert all(isinstance(i, int) for i in idx)


def test_from_exovista_positions_runnable(fits_fixture):
    """Sanity: a System loaded from FITS can be propagated."""
    sys_obj = from_exovista(fits_fixture, planet_indices=[0])
    t0 = sys_obj.planets[0].orbit.t0_d
    pos = sys_obj.positions(jnp.atleast_1d(t0))
    assert pos.shape[0] == 2
    assert pos.shape[2] == 1


@pytest.mark.parametrize(
    ("mode", "expected_contrast", "expected_flux"),
    [
        (
            "time",
            [
                [2.356564187987321e-10, 2.56341309788488e-10, 1.8167898333894579e-10],
                [5.670880637443406e-12, 1.5466097976366512e-10, 1.3689403177376184e-10],
                [5.45207491898194e-10, 3.7782533688573034e-10, 2.0653046850982173e-10],
            ],
            [
                [0.00018996762923968123, 0.00020664215787096466, 0.0001464552755384602],
                [4.571416963254523e-06, 0.00012467549074775237, 0.0001103531777563576],
                [
                    0.00043950330403718193,
                    0.00030457300454933436,
                    0.00016648858396715383,
                ],
            ],
        ),
    ],
)
def test_exovista_flux_preserved_with_standard_phase(
    fits_fixture, mode, expected_contrast, expected_flux
):
    """Preserve imported spectra while exposing the standard phase angle.

    The 600 nm contrasts and photon fluxes are recorded outputs for demo
    planets 0, 2, 4 at FITS epochs 0, 10, 25. Time-indexed spectra must be
    unchanged by corrections to the imported orbit. The tolerance allows
    float32 solver tables on any backend.

    Green (1985), Spherical Astronomy, ch. 17; Brandt et al. (2021),
    AJ 162, 186: raw z points away. Brown (2005), ApJ 624, 1010, Eq. 4
    uses the standard star-planet-observer phase angle.
    """
    system = from_exovista(
        fits_fixture, planet_indices=[0, 2, 4], load_disk=False, planet_spectra=mode
    )
    times = jnp.array([2451545.0, 2451645.0, 2451795.0])
    np.testing.assert_allclose(
        system.contrasts(600.0, times), expected_contrast, rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        system.planet_flux_densities(600.0, times), expected_flux, rtol=1e-6, atol=0
    )
    for planet in system.planets:
        raw, _, dist = planet.orbit.propagate(
            system.trig_solver, times, Ms_kg=system.star.Ms_kg
        )
        _, phase, _ = planet.propagate(system.trig_solver, times, star=system.star)
        np.testing.assert_allclose(jnp.cos(phase), raw[:, 2, :] / dist, atol=1e-6)
