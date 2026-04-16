"""Astrophysical scene modeling for HWO direct imaging.

Legacy types (``Planet``, ``Star``, ``System``, ``Disk``, ``from_exovista``) are
preserved under ``skyscapes._legacy`` and re-exported at the top level so
existing consumers keep working while the new scene hierarchy lands.

New code should import from ``skyscapes.scene``, ``skyscapes.atmosphere``,
``skyscapes.disk``, and ``skyscapes.io`` (populated by later tasks in this
plan).
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import _legacy
from ._legacy import (
    Disk,
    Planet,
    Star,
    System,
    from_exovista,
    get_earth_like_planet_indices,
)

__all__ = [
    "Disk",
    "Planet",
    "Star",
    "System",
    "_legacy",
    "from_exovista",
    "get_earth_like_planet_indices",
]
