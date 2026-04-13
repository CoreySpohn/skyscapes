"""JAX-friendly planetary system model for skyscapes.

Simple container grouping a star, its planets, and an optional debris disk.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx

from .disk import Disk
from .planet import Planet
from .star import Star


class System(eqx.Module):
    """Complete planetary system: star + planets + disk.

    This is the pure astrophysical system — no backgrounds, no observatory.
    Background sources (zodiacal light, etc.) are handled by consumers
    like coronagraphoto's ``SkyScene``.
    """

    star: Star
    planet: Planet
    disk: Optional[Disk] = None
