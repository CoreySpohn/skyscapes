"""Smoke test for legacy shim re-exports."""

from __future__ import annotations


def test_top_level_legacy_reexports():
    import skyscapes

    assert skyscapes.Planet is skyscapes._legacy.planet.Planet
    assert skyscapes.Star is skyscapes._legacy.star.Star
    assert skyscapes.System is skyscapes._legacy.system.System
    assert skyscapes.Disk is skyscapes._legacy.disk.Disk
    assert skyscapes.from_exovista is skyscapes._legacy.loaders.from_exovista
    assert (
        skyscapes.get_earth_like_planet_indices
        is skyscapes._legacy.loaders.get_earth_like_planet_indices
    )
