"""skyscapes.io.exovista -- ExoVista FITS to scene.System."""

from __future__ import annotations

import jax.numpy as jnp

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
