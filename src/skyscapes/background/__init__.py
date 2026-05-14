"""Background sources that fill the field of view.

A background is anything in the telescope's field of view that is not part
of the planetary system itself: zodiacal light, extragalactic background,
line-of-sight stars, etc. Each background type is its own concrete class;
there is deliberately no shared abstract base yet -- the eventual taxonomy
will be designed once a second background type beyond zodi exists.

Currently this module exposes three zodi flavours, all returning
``ph/s/m^2/nm`` per ``arcsec^2``.
"""

from __future__ import annotations

from .zodi import ZodiSourceAYO, ZodiSourceLeinert, ZodiSourcePhotonFlux

__all__ = [
    "ZodiSourceAYO",
    "ZodiSourceLeinert",
    "ZodiSourcePhotonFlux",
]
