"""Top-level Scene container.

A ``Scene`` is everything in the telescope's field of view. It wraps a
planetary :class:`System` (star + planets + disk) together with a small,
explicit set of named background sources -- one field per type.

Backgrounds are deliberately *not* grouped into a single
``tuple[AbstractBackground, ...]`` field. With JAX, an explicit static
PyTree shape is preferable to a heterogeneous container: JIT can fully
trace each named field without having to retrace over a Python loop, and
the eventual concrete background types (zodi, galaxy fields, exozodiacal
dust clumps, line-of-sight transients) are likely to need very different
integration kernels in the downstream simulator. Each new source type is
added as its own explicit ``Scene`` field with a concrete type.

For now, the only background field is ``zodi``. Additional fields
(``galaxy_field``, ``exozodi_clumps``, ...) will be added one at a time as
the corresponding source types are implemented.
"""

from __future__ import annotations

import equinox as eqx

from ..background import ZodiSource
from .system import System


class Scene(eqx.Module):
    """Field-of-view container: planetary system + named background sources.

    Attributes:
        system: The planetary system (star + planets + disk).
        zodi: Optional zodiacal-light background (any
            :mod:`skyscapes.background` zodi variant).
    """

    system: System
    zodi: ZodiSource | None = None

    @property
    def star(self):
        """Convenience accessor for the host star."""
        return self.system.star

    @property
    def planets(self):
        """Convenience accessor for the planet tuple."""
        return self.system.planets

    @property
    def disk(self):
        """Convenience accessor for the system's disk (may be ``None``)."""
        return self.system.disk
