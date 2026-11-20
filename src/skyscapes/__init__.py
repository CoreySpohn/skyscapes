"""Astrophysical scene modeling for HWO direct imaging.

Public API:

- :mod:`skyscapes.scene` -- star + planet hierarchy and the ``Scene`` that
  holds star, planets, disk and local zodi.
- :mod:`skyscapes.disk` -- extended-source surface brightness maps.
- :mod:`skyscapes.physical_model` -- planet photon flux and planet-to-star
  contrast (reflective, emissive, and additive models).
- :mod:`skyscapes.io` -- data loaders (e.g. ExoVista FITS).
- :mod:`skyscapes.habitable_zone` -- habitable-zone edges for a host star.
- :mod:`skyscapes.population` -- planet occurrence tables and population draws.
- :mod:`skyscapes.datasets` -- cached reference and example data.
- :mod:`skyscapes.viz` -- optional plotting with lazily loaded dependencies.

For convenience, ``Scene``, the local-zodi models and ``from_exovista`` are
hoisted to the top level, so common flows can write
``from skyscapes import Scene, from_exovista``.
Mix-and-match construction goes through the submodules
(``skyscapes.disk.ExovistaDisk``, ``skyscapes.physical_model.GridPhysicalModel``,
``skyscapes.AYOZodi``, ...).
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import (
    datasets,
    disk,
    habitable_zone,
    io,
    physical_model,
    population,
    scene,
    viz,
)
from .io import from_exovista
from .scene import AbstractLocalZodi, AYOZodi, LeinertZodi, PrecomputedZodi, Scene

__all__ = [
    "AYOZodi",
    "AbstractLocalZodi",
    "LeinertZodi",
    "PrecomputedZodi",
    "Scene",
    "__version__",
    "datasets",
    "disk",
    "from_exovista",
    "habitable_zone",
    "io",
    "physical_model",
    "population",
    "scene",
    "viz",
]
