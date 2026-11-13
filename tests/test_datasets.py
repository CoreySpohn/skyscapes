"""skyscapes.datasets -- pooch-backed fixture fetcher."""

from __future__ import annotations

import os

import pytest

from skyscapes.datasets import (
    PIKACHU,
    fetch_ck04,
    fetch_phoenix,
    fetch_scene,
    list_ck04,
    list_phoenix,
)


def test_fetch_scene_returns_existing_fits():
    """fetch_scene returns an existing solar_system_mod.fits path."""
    path = fetch_scene()
    assert os.path.isfile(path)
    assert path.endswith("solar_system_mod.fits")


@pytest.mark.parametrize(
    "list_nodes, fetch_node, prefix, first, last",
    [
        (list_phoenix, fetch_phoenix, "phoenixm00", 2000, 12000),
        (list_ck04, fetch_ck04, "ckp00", 3500, 50000),
    ],
)
def test_phoenix_registry_is_complete_and_hashed(
    list_nodes, fetch_node, prefix, first, last
):
    """All 76 nodes of each atmosphere grid carry a SHA-256 hash and an archive URL.

    Fails if a node is registered without a hash (pooch would then accept any
    bytes) or points at a host other than the STScI archive.
    """
    nodes = list_nodes()
    assert len(nodes) == 76
    assert nodes[0] == first and nodes[-1] == last
    for teff_k in nodes:
        name = f"{prefix}-{teff_k}-v1.fits"
        assert PIKACHU.registry[name].startswith("sha256:")
        assert PIKACHU.get_url(name).startswith("https://archive.stsci.edu/")
    with pytest.raises(ValueError, match="registered"):
        fetch_node(3550)
