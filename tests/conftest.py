"""Shared pytest fixtures for skyscapes tests."""

from __future__ import annotations

import importlib.util
import os

from hypothesis import settings
from jax import config

# ExoJAX refuses to initialize in JAX's default 32-bit mode (raises on
# import of MdbHitran / Art* engines). Flip x64 before any test imports
# touch JAX state so the slow integration test can run on CI.
config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from skyscapes.datasets import fetch_scene  # noqa: E402

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


@pytest.fixture(scope="session")
def fits_fixture() -> str:
    """Path to the canonical ExoVista demo FITS (cached via pooch)."""
    return fetch_scene()
