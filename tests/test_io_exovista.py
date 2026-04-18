"""skyscapes.io.exovista -- ExoVista FITS to scene.System."""

from __future__ import annotations

import jax.numpy as jnp

from skyscapes.disk import ExovistaDisk
from skyscapes.io import from_exovista, get_earth_like_planet_indices
from skyscapes.scene import Planet, SpectrumStar, System


def test_from_exovista_returns_scene_system(fits_fixture):
    """from_exovista returns a scene.System instance."""
    sys_obj = from_exovista(fits_fixture)
    assert isinstance(sys_obj, System)


def test_from_exovista_star_is_spectrum_star(fits_fixture):
    """Loaded star is a SpectrumStar with Ms_kg populated from FITS."""
    sys_obj = from_exovista(fits_fixture)
    assert isinstance(sys_obj.star, SpectrumStar)
    assert sys_obj.star.Ms_kg > 0.0


def test_from_exovista_planets_are_tuple(fits_fixture):
    """Planets is a non-empty tuple of scene.Planet."""
    sys_obj = from_exovista(fits_fixture)
    assert isinstance(sys_obj.planets, tuple)
    assert len(sys_obj.planets) > 0
    for p in sys_obj.planets:
        assert isinstance(p, Planet)


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
