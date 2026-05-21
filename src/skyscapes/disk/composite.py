"""CompositeDisk: sum of multiple ``AbstractDisk`` components.

Mirrors how ``skyscapes.scene.System`` composes multiple ``Planet`` objects
via a tuple: a ``CompositeDisk`` holds a tuple of disk components and
renders their summed ``surface_brightness`` on a shared grid.

All components must share the same ``spatial_extent()`` so the per-component
rendered arrays can be added directly without resampling. Mismatched
geometries are rejected at construction.
"""

from __future__ import annotations

from jaxtyping import Array

from .base import AbstractDisk


class CompositeDisk(AbstractDisk):
    """Sum of multiple disk components rendered on a shared grid.

    Attributes:
        components: Tuple of ``AbstractDisk`` instances. At least one
            component is required. All components must report the same
            ``spatial_extent()`` so their rendered ``(ny, nx)`` arrays
            sum elementwise.
    """

    components: tuple[AbstractDisk, ...]

    def __check_init__(self):
        """Reject empty composites and mismatched component extents."""
        if len(self.components) == 0:
            raise ValueError("CompositeDisk requires at least one component.")
        ref_extent = self.components[0].spatial_extent()
        for i, c in enumerate(self.components[1:], start=1):
            ext = c.spatial_extent()
            if ext != ref_extent:
                raise ValueError(
                    f"CompositeDisk component {i} has spatial_extent={ext}, "
                    f"which does not match component 0's {ref_extent}. All "
                    f"components must share the same grid for direct summation."
                )

    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_jd: Array,
        incl_deg: Array,
        pa_deg: Array,
    ) -> Array:
        """Return the per-pixel sum of all component contrast maps."""
        contrib = self.components[0].surface_brightness(
            wavelength_nm, time_jd, incl_deg, pa_deg
        )
        for c in self.components[1:]:
            contrib = contrib + c.surface_brightness(
                wavelength_nm, time_jd, incl_deg, pa_deg
            )
        return contrib

    def spatial_extent(self) -> tuple[float, float]:
        """Return the shared ``(width_arcsec, height_arcsec)``."""
        return self.components[0].spatial_extent()
