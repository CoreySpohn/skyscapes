"""Scene hierarchy: AbstractStar + Planet + System + Scene wiring."""

from __future__ import annotations

from .container import Scene
from .planet import Planet
from .star import AbstractStar, SimpleStar, SpectrumStar
from .system import System

__all__ = [
    "AbstractStar",
    "Planet",
    "Scene",
    "SimpleStar",
    "SpectrumStar",
    "System",
]
