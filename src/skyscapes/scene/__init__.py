"""Scene hierarchy: AbstractStar + Planet + System wiring.

New-style scene classes. Top-level ``skyscapes.{Star,Planet,System}``
still point at the legacy shim until Plan 4 flips them.
"""

from __future__ import annotations

from .star import AbstractStar, SimpleStar, SpectrumStar

__all__ = [
    "AbstractStar",
    "SimpleStar",
    "SpectrumStar",
]
