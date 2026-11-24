"""Smoke test for the top-level skyscapes surface."""

from __future__ import annotations

import skyscapes


def test_toplevel_identity():
    """Top-level convenience names point at the canonical scene/io locations."""
    assert skyscapes.Scene is skyscapes.scene.Scene
    assert skyscapes.from_exovista is skyscapes.io.from_exovista


def test_system_is_gone():
    """The planetary System container was merged into Scene."""
    assert not hasattr(skyscapes, "System")
    assert not hasattr(skyscapes.scene, "System")
    assert not hasattr(skyscapes.Scene, "system")


def test_submodules_importable():
    """All public submodules round-trip through the top-level package."""
    for name in (
        "scene",
        "disk",
        "physical_model",
        "io",
        "habitable_zone",
    ):
        assert hasattr(skyscapes, name), f"skyscapes.{name} missing"


def test_no_module_holds_a_static_dict():
    """Fails if a public Module stores a dict as a static field.

    A static dict is part of the tree structure, so two scenes whose dicts
    differ cannot batch, and a dict of host data does not belong in a Module.
    Truth: the field annotations of every Module exported by the scene, disk,
    physical_model, population and io packages; an annotation naming ``dict``
    or a mapping type fails. The check is exact.
    """
    import dataclasses
    import importlib
    import inspect
    import re

    import equinox as eqx

    seen = 0
    offenders = []
    for package in ("scene", "disk", "physical_model", "population", "io"):
        module = importlib.import_module(f"skyscapes.{package}")
        for name in module.__all__:
            cls = getattr(module, name)
            if not (inspect.isclass(cls) and issubclass(cls, eqx.Module)):
                continue
            seen += 1
            offenders += [
                f"{name}.{f.name}"
                for f in dataclasses.fields(cls)
                if f.metadata.get("static")
                and re.search(r"\b(dict|Dict|Mapping)\b", str(f.type))
            ]
    assert seen > 20
    assert offenders == []
