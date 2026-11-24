"""Grid-interpolated contrast (per-planet 2D interpax)."""

from __future__ import annotations

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_bulk_leaf, as_float_leaf
from jaxtyping import Array
from optixstuff.spectral import Knots, as_knots

from .base import AbstractPhysicalModel


class GridPhysicalModel(AbstractPhysicalModel):
    """Per-planet 2D interpolated contrast over (wavelength, phase-angle).

    Distance and planet radius are ignored -- the grid already encodes
    a flux ratio. The wavelength grid is a static ``Knots``, so the table's
    operator is built for it on the host and no fit can move it.

    The contrast cube is a declared bulk field: ExoVista planet contrast
    cubes are float32 on disk, and a float32 cube stays float32 under x64;
    the interpolation promotes it at use.

    Attributes:
        knots: Wavelength grid [nm], shape ``(n_wl,)``, static.
        phase_angle_deg: 1-D phase-angle grid [deg], shape ``(n_phase,)``.
        contrast_grid: Contrast cube, shape ``(K, n_wl, n_phase)``.
    """

    knots: Knots = eqx.field(static=True, converter=as_knots)
    phase_angle_deg: Array = eqx.field(converter=as_float_leaf)
    contrast_grid: Array = eqx.field(converter=as_bulk_leaf)

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """The wavelength grid [nm], float64 on the host."""
        return self.knots.nm

    def contrast(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Per-planet contrast at (wavelength, phase).

        Args:
            phase_angle_rad: Phase angle per planet [rad], shape ``(K, T)``.
            star_planet_distance_AU: Shape ``(K, T)``; ignored (grid has the ratio).
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
        knots = self.knots.array()
        dtype = jnp.result_type(
            wl_scalar,
            phase_deg,
            knots,
            self.phase_angle_deg,
            self.contrast_grid,
        )
        wl_arr_scalar = wl_scalar.astype(dtype)
        phase_deg_x = phase_deg.astype(dtype)
        wavelengths_nm = knots.astype(dtype)
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
        """One-line summary of grid shape (K, n_wl, n_phase) and wl range.

        Returns:
            One-line summary string.
        """
        K = int(self.contrast_grid.shape[0])
        n_wl = len(self.knots)
        n_phase = int(self.phase_angle_deg.shape[0])
        wl_min = float(self.knots.nm[0])
        wl_max = float(self.knots.nm[-1])
        return (
            f"GridPhysicalModel(K={K}, "
            f"wl={wl_min:.0f}-{wl_max:.0f} nm ({n_wl} pts), "
            f"phase={n_phase} pts)"
        )
