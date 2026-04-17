"""Abstract disk interface for extended-source surface brightness."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array


class AbstractDisk(eqx.Module):
    """Extended-source surface brightness map.

    Subclasses return contrast (flux ratio relative to the host star) per
    pixel. The concrete ``System`` multiplies by ``star.spec_flux_density``
    to turn that into ph/s/m²/nm per pixel.
    """

    @abstractmethod
    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Return contrast per pixel, shape ``(ny, nx)``."""

    @abstractmethod
    def spatial_extent(self) -> tuple[float, float]:
        """Return ``(width_arcsec, height_arcsec)``."""
