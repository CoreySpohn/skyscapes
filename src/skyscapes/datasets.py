"""Data management utilities for skyscapes.

Pooch-managed fetchers for test/example FITS fixtures. Mirrors the
``scenes.zip`` entry from ``coronagraphoto.datasets`` (same file, hash,
and URL) but caches under its own ``skyscapes`` directory.
"""

from __future__ import annotations

import pooch
from pooch import Unzip

REGISTRY = {
    "scenes.zip": "md5:c777aefb65887249892093b1aba6d86a",
}

PIKACHU = pooch.create(
    path=pooch.os_cache("skyscapes"),
    base_url="https://github.com/CoreySpohn/coronalyze/raw/main/data/",
    registry=REGISTRY,
)


def fetch_scene() -> str:
    """Fetch and unpack the canonical ExoVista demo scene.

    Returns:
        Absolute path to ``solar_system_mod.fits``.
    """
    PIKACHU.fetch("scenes.zip", processor=Unzip())
    return str(
        PIKACHU.abspath / "scenes.zip.unzip" / "scenes" / "solar_system_mod.fits"
    )
