"""Grid-interpolated contrast (per-planet 2D interpax)."""

from __future__ import annotations

import interpax
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractAtmosphere


class GridAtmosphere(AbstractAtmosphere):
    """Per-planet 2D interpolated contrast over (wavelength, phase-angle).

    Distance is ignored — the grid already encodes a flux ratio.

    Attributes:
        Rp_Rearth: Planetary radii, shape ``(K,)``.
        wavelengths_nm: 1-D wavelength grid [nm], shape ``(n_wl,)``.
        phase_angle_deg: 1-D phase-angle grid [deg], shape ``(n_phase,)``.
        contrast_grid: Contrast cube, shape ``(K, n_wl, n_phase)``.
    """

    Rp_Rearth: Array
    wavelengths_nm: Array
    phase_angle_deg: Array
    contrast_grid: Array

    def reflected_spectrum(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
    ) -> Array:
        """Per-planet contrast at (wavelength, phase).

        Args:
            phase_angle_rad: Phase angle per planet [rad]. Shape ``(K,)``
                or ``(K, T)``.
            dist_AU: Ignored (grid already encodes flux ratio).
            wavelength_nm: Scalar wavelength.

        Returns:
            Contrast, shape matching ``phase_angle_rad``.
        """
        del dist_AU
        wl_scalar = jnp.asarray(wavelength_nm)
        phase_deg = jnp.rad2deg(phase_angle_rad) % 360.0
        # Promote shape (K,) → (K, 1) for uniform vmap handling.
        phase_2d = phase_deg if phase_deg.ndim == 2 else phase_deg[:, None]

        def per_planet(grid_k, phase_row):
            # grid_k: (n_wl, n_phase); phase_row: (T,)
            wl_arr = jnp.broadcast_to(wl_scalar, phase_row.shape)
            return interpax.interp2d(
                wl_arr,
                phase_row,
                self.wavelengths_nm,
                self.phase_angle_deg,
                grid_k,
                method="linear",
                extrap=True,
            )

        result = jax.vmap(per_planet)(self.contrast_grid, phase_2d)
        # Squeeze the synthetic T axis if the caller passed (K,).
        if phase_deg.ndim == 1:
            result = result.squeeze(axis=-1)
        return result
