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
    """Fails if a public Module stores a dict, or untyped host data, as a static field.

    A static dict is part of the tree structure, so two scenes whose dicts
    differ cannot batch, and a dict of host data does not belong in a Module.
    Truth: the field annotations of every public ``eqx.Module`` class defined
    in any skyscapes submodule (walked, so nested classes such as the ExoJAX
    components count). An annotation naming ``dict`` or a mapping type fails,
    and so does a static ``Any`` field other than the three named ExoJAX
    engine handles. The check is exact.
    """
    import dataclasses
    import importlib
    import inspect
    import pkgutil
    import re

    import equinox as eqx

    engine_handles = {
        ("MolecularSpecies", "opa"),
        ("ExoJaxEmissionModel", "rt_engine"),
        ("ExoJaxPhysicalModel", "rt_engine"),
    }
    classes = {}
    for info in pkgutil.walk_packages(skyscapes.__path__, "skyscapes."):
        module = importlib.import_module(info.name)
        for cls in vars(module).values():
            if (
                inspect.isclass(cls)
                and issubclass(cls, eqx.Module)
                and cls.__module__.startswith("skyscapes")
                and not cls.__name__.startswith("_")
            ):
                classes[cls] = None
    offenders = []
    for cls in classes:
        for f in dataclasses.fields(cls):
            if not f.metadata.get("static"):
                continue
            kind = str(f.type)
            if re.search(r"\b(dict|Dict|Mapping)\b", kind) or (
                re.search(r"\b(Any|object)\b", kind)
                and (cls.__name__, f.name) not in engine_handles
            ):
                offenders.append(f"{cls.__name__}.{f.name}: {kind}")
    assert len(classes) > 50
    assert any(cls.__name__ == "GrayCloud" for cls in classes)
    assert offenders == []
