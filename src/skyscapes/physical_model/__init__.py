"""Physical-model hierarchy: planet photon flux and planet-to-star contrast."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .base import AbstractPhysicalModel
from .blackbody import BlackbodyPhysicalModel
from .cached import PrecomputedPhysicalModel
from .grid import GridPhysicalModel
from .lambertian import LambertianPhysicalModel, lambert_phase, reflected_contrast
from .sum import SumPhysicalModel
from .time_series import TimeSeriesPhysicalModel

if TYPE_CHECKING:
    from .exojax import ExoJaxEmissionModel, ExoJaxPhysicalModel

_LAZY = {"ExoJaxEmissionModel", "ExoJaxPhysicalModel"}


def __getattr__(name):
    """Load the ExoJAX models on first use, so importing skyscapes stays light."""
    if name in _LAZY:
        value = getattr(import_module(".exojax", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractPhysicalModel",
    "BlackbodyPhysicalModel",
    "ExoJaxEmissionModel",
    "ExoJaxPhysicalModel",
    "GridPhysicalModel",
    "LambertianPhysicalModel",
    "PrecomputedPhysicalModel",
    "SumPhysicalModel",
    "TimeSeriesPhysicalModel",
    "lambert_phase",
    "reflected_contrast",
]
