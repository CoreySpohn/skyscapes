"""Frozen legacy implementations — preserved as a safety net during the scene migration.

Everything here is a direct move of the pre-migration top-level modules.
Do not add features. Do not refactor. This subpackage is deleted in Plan 4.
"""

from __future__ import annotations

from . import disk, loaders, planet, star, system
from .disk import Disk
from .loaders import from_exovista, get_earth_like_planet_indices
from .planet import Planet
from .star import Star
from .system import System

__all__ = [
    "Disk",
    "Planet",
    "Star",
    "System",
    "disk",
    "from_exovista",
    "get_earth_like_planet_indices",
    "loaders",
    "planet",
    "star",
    "system",
]
