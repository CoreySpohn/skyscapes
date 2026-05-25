"""Grid-interpolated contrast (per-planet 2D interpax)."""

from __future__ import annotations

import interpax
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractPhysicalModel


class GridPhysicalModel(AbstractPhysicalModel):
    """Per-planet 2D interpolated contrast over (wavelength, phase-angle).

    Distance and planet radius are ignored -- the grid already encodes
    a flux ratio.

    Attributes:
        wavelengths_nm: 1-D wavelength grid [nm], shape ``(n_wl,)``.
        phase_angle_deg: 1-D phase-angle grid [deg], shape ``(n_phase,)``.
        contrast_grid: Contrast cube, shape ``(K, n_wl, n_phase)``.
    """

    wavelengths_nm: Array
    phase_angle_deg: Array
    contrast_grid: Array

    def contrast(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Per-planet contrast at (wavelength, phase).

        Args:
            phase_angle_rad: Phase angle per planet [rad], shape ``(K, T)``.
            dist_AU: Shape ``(K, T)``; ignored (grid encodes flux ratio).
            wavelength_nm: Scalar wavelength.
            Rp_Rearth: Shape ``(K,)``; ignored (grid encodes flux ratio).

        Returns:
            Contrast, shape ``(K, T)``.
        """
        wl_scalar = jnp.asarray(wavelength_nm)
        phase_deg = jnp.rad2deg(phase_angle_rad) % 360.0

        # Promote all interp inputs to a common dtype. Loaded ExoVista
        # contrast cubes are float32; under ``with jax.enable_x64():``
        # the orbit propagator returns float64 phase angles, and
        # interpax raises ``TypeError: switch branches must have equal
        # output types`` on the mismatch. Casting here keeps the load
        # representation cheap (float32) and pays the promotion only at
        # query time when needed.
        dtype = jnp.result_type(
            wl_scalar,
            phase_deg,
            self.wavelengths_nm,
            self.phase_angle_deg,
            self.contrast_grid,
        )
        wl_arr_scalar = wl_scalar.astype(dtype)
        phase_deg_x = phase_deg.astype(dtype)
        wavelengths_nm = self.wavelengths_nm.astype(dtype)
        phase_angle_deg = self.phase_angle_deg.astype(dtype)
        contrast_grid = self.contrast_grid.astype(dtype)

        def per_planet(grid_k, phase_row):
            # grid_k: (n_wl, n_phase); phase_row: (T,)
            wl_arr = jnp.broadcast_to(wl_arr_scalar, phase_row.shape)
            return interpax.interp2d(
                wl_arr,
                phase_row,
                wavelengths_nm,
                phase_angle_deg,
                grid_k,
                method="linear",
                extrap=True,
            )

        return jax.vmap(per_planet)(contrast_grid, phase_deg_x)

    def __repr__(self) -> str:
        """One-line summary of grid shape (K, n_wl, n_phase) and wl range."""
        K = int(self.contrast_grid.shape[0])
        n_wl = int(self.wavelengths_nm.shape[0])
        n_phase = int(self.phase_angle_deg.shape[0])
        wl_min = float(self.wavelengths_nm.min())
        wl_max = float(self.wavelengths_nm.max())
        return (
            f"GridPhysicalModel(K={K}, "
            f"wl={wl_min:.0f}-{wl_max:.0f} nm ({n_wl} pts), "
            f"phase={n_phase} pts)"
        )
