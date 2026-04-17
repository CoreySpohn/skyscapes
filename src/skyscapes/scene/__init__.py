"""Scene hierarchy: AbstractStar + Planet + System wiring."""

from __future__ import annotations

from .planet import Planet
from .star import AbstractStar, SimpleStar, SpectrumStar
from .system import System

__all__ = [
    "AbstractStar",
    "Planet",
    "SimpleStar",
    "SpectrumStar",
    "System",
]
