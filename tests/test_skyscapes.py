"""Smoke test for the top-level skyscapes surface."""

from __future__ import annotations

import skyscapes


def test_toplevel_identity():
    """Top-level convenience names point at the canonical scene/io locations."""
    assert skyscapes.System is skyscapes.scene.System
    assert skyscapes.from_exovista is skyscapes.io.from_exovista


def test_submodules_importable():
    """All four public submodules round-trip through the top-level package."""
    for name in ("scene", "disk", "atmosphere", "io"):
        assert hasattr(skyscapes, name), f"skyscapes.{name} missing"
