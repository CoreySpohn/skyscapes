"""Atmosphere hierarchy: phase/wavelength-dependent planet-to-star contrast."""

from __future__ import annotations

from .base import AbstractAtmosphere
from .grid import GridAtmosphere
from .lambertian import LambertianAtmosphere
from .parametric import ParametricAtmosphere

__all__ = [
    "AbstractAtmosphere",
    "GridAtmosphere",
    "LambertianAtmosphere",
    "ParametricAtmosphere",
]
