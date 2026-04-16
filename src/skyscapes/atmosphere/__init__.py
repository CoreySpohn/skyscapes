"""Atmosphere hierarchy: phase/wavelength-dependent planet-to-star contrast."""

from __future__ import annotations

from .base import AbstractAtmosphere
from .lambertian import LambertianAtmosphere

__all__ = [
    "AbstractAtmosphere",
    "LambertianAtmosphere",
]
