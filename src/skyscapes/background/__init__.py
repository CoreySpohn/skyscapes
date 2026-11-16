"""Local backgrounds with a common per-steradian photon radiance contract."""

from .zodi import AbstractLocalZodi, AYOZodi, LeinertZodi, PrecomputedZodi

Zodi = AYOZodi | LeinertZodi | PrecomputedZodi

__all__ = ["AYOZodi", "AbstractLocalZodi", "LeinertZodi", "PrecomputedZodi", "Zodi"]
