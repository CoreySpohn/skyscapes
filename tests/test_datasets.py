"""skyscapes.datasets -- pooch-backed fixture fetcher."""

from __future__ import annotations

import os

from skyscapes.datasets import fetch_scene


def test_fetch_scene_returns_existing_fits():
    """fetch_scene returns an existing solar_system_mod.fits path."""
    path = fetch_scene()
    assert os.path.isfile(path)
    assert path.endswith("solar_system_mod.fits")
