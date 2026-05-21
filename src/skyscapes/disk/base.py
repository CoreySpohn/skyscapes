"""Abstract disk interface for extended-source surface brightness."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array


class AbstractDisk(eqx.Module):
    """Extended-source surface brightness map.

    Subclasses return contrast (flux ratio relative to the host star) per
    pixel. The concrete ``System`` multiplies by ``star.spec_flux_density``
    to turn that into ph/s/m^2/nm per pixel.

    ``incl_deg`` / ``pa_deg`` are supplied at render time rather than stored
    on the disk so the System's midplane orientation drives every disk
    component consistently. Disks whose geometry is pre-baked into their
    representation (e.g. ``ExovistaDisk``'s contrast cube) may ignore the
    arguments; parametric disks (``GraterDisk``, ``ExovistaParametricDisk``)
    consume them.
    """

    @abstractmethod
    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_jd: Array,
        incl_deg: Array,
        pa_deg: Array,
    ) -> Array:
        """Return contrast per pixel, shape ``(ny, nx)``."""

    @abstractmethod
    def spatial_extent(self) -> tuple[float, float]:
        """Return ``(width_arcsec, height_arcsec)``."""
