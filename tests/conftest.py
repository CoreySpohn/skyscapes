"""Shared pytest fixtures for skyscapes tests."""

from __future__ import annotations

# ExoJAX refuses to initialize in JAX's default 32-bit mode (raises on
# import of MdbHitran / Art* engines). Flip x64 before any test imports
# touch JAX state so the slow integration test can run on CI.
from jax import config

config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from skyscapes.datasets import fetch_scene  # noqa: E402


@pytest.fixture(scope="session")
def fits_fixture() -> str:
    """Path to the canonical ExoVista demo FITS (cached via pooch)."""
    return fetch_scene()
