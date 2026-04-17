"""Parity-suite fixtures: build the same FITS as both legacy and scene systems."""

from __future__ import annotations

import pytest

from skyscapes._legacy.loaders import from_exovista as from_exovista_legacy
from skyscapes.io import from_exovista as from_exovista_scene


@pytest.fixture(scope="session")
def legacy_system(fits_fixture):
    """Legacy skyscapes.System built from the canonical FITS."""
    return from_exovista_legacy(fits_fixture)


@pytest.fixture(scope="session")
def scene_system(fits_fixture):
    """New skyscapes.scene.System built from the canonical FITS."""
    return from_exovista_scene(fits_fixture)
