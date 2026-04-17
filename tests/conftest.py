"""Shared pytest fixtures for skyscapes tests."""

from __future__ import annotations

import pytest

from skyscapes.datasets import fetch_scene


@pytest.fixture(scope="session")
def fits_fixture() -> str:
    """Path to the canonical ExoVista demo FITS (cached via pooch)."""
    return fetch_scene()
