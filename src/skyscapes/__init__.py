"""Astrophysical scene modeling for HWO direct imaging.

Public API:

- :mod:`skyscapes.scene` -- star + planet + system hierarchy.
- :mod:`skyscapes.disk` -- extended-source surface brightness maps.
- :mod:`skyscapes.atmosphere` -- phase/wavelength-dependent planet-to-star contrast.
- :mod:`skyscapes.io` -- data loaders (e.g. ExoVista FITS).

For convenience, ``System`` and ``from_exovista`` are hoisted to the top level,
so common flows can write ``from skyscapes import System, from_exovista``.
Mix-and-match construction goes through the submodules
(``skyscapes.disk.ExovistaDisk``, ``skyscapes.atmosphere.GridAtmosphere``, ...).
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import atmosphere, disk, io, scene
from .io import from_exovista
from .scene import System

__all__ = [
    "System",
    "__version__",
    "atmosphere",
    "disk",
    "from_exovista",
    "io",
    "scene",
]
