"""skyscapes.datasets -- pooch-backed fixture fetcher."""

from __future__ import annotations

import os

import pytest

from skyscapes.datasets import PIKACHU, fetch_phoenix, fetch_scene, list_phoenix


def test_fetch_scene_returns_existing_fits():
    """fetch_scene returns an existing solar_system_mod.fits path."""
    path = fetch_scene()
    assert os.path.isfile(path)
    assert path.endswith("solar_system_mod.fits")


def test_phoenix_registry_is_complete_and_hashed():
    """All 76 PHOENIX nodes carry a SHA-256 hash and an archive URL.

    Fails if a node is registered without a hash (pooch would then accept any
    bytes) or points at a host other than the STScI archive.
    """
    nodes = list_phoenix()
    assert len(nodes) == 76
    assert nodes[0] == 2000 and nodes[-1] == 12000
    for teff_k in nodes:
        name = f"phoenixm00-{teff_k}-v1.fits"
        assert PIKACHU.registry[name].startswith("sha256:")
        assert PIKACHU.get_url(name).startswith("https://archive.stsci.edu/")
    with pytest.raises(ValueError, match="registered PHOENIX node"):
        fetch_phoenix(3550)
