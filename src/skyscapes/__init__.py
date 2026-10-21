"""Astrophysical scene modeling for HWO direct imaging.

Public API:

- :mod:`skyscapes.scene` -- star + planet + system hierarchy, plus the
  top-level ``Scene`` container.
- :mod:`skyscapes.disk` -- extended-source surface brightness maps.
- :mod:`skyscapes.physical_model` -- phase/wavelength-dependent
  planet-to-star contrast (reflective, emissive, future joint models).
- :mod:`skyscapes.background` -- field-of-view background sources (zodi, ...).
- :mod:`skyscapes.io` -- data loaders (e.g. ExoVista FITS).
- :mod:`skyscapes.habitable_zone` -- habitable-zone edges for a host star.

For convenience, ``Scene``, ``System``, and ``from_exovista`` are hoisted to
the top level, so common flows can write
``from skyscapes import Scene, System, from_exovista``.
Mix-and-match construction goes through the submodules
(``skyscapes.disk.ExovistaDisk``, ``skyscapes.physical_model.GridPhysicalModel``,
``skyscapes.background.AYOZodi``, ...).
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import background, disk, habitable_zone, io, physical_model, scene
from .io import from_exovista
from .scene import Scene, System

__all__ = [
    "Scene",
    "System",
    "__version__",
    "background",
    "disk",
    "from_exovista",
    "habitable_zone",
    "io",
    "physical_model",
    "scene",
]
