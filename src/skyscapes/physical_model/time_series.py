"""Bounded interpolation of planet contrast at physical observation epochs."""

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .base import AbstractPhysicalModel


class TimeSeriesPhysicalModel(AbstractPhysicalModel):
    """Tabulated planet-to-star contrast over wavelength and Julian Date.

    Linear interpolation preserves nonnegative contrast inside the table.
    Multiple epochs define bounded temporal support: queries before the
    first epoch or after the last return NaN. A single epoch explicitly
    represents a constant snapshot at any time. Wavelength support is
    always bounded; a single wavelength is valid only at that wavelength.

    Phase, distance, and radius are ignored because their effects are
    already included in the tabulated ratios. Equal orbital phases do not
    imply equal contrasts at different epochs.

    Attributes:
        wavelengths_nm: Finite, strictly increasing grid [nm], shape ``(W,)``.
        times_jd: Finite, strictly increasing Julian Dates, shape ``(T,)``.
            Use float64 for subday resolution at modern Julian Dates.
        contrast_grid: Finite, nonnegative flux ratios, shape ``(K, W, T)``.
            Storage dtype is retained, including float32 contrast cubes.
    """

    wavelengths_nm: Array = eqx.field(converter=jnp.asarray)
    times_jd: Array = eqx.field(converter=jnp.asarray)
    contrast_grid: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Reject invalid table coordinates and contrasts before evaluation."""
        for name in ("wavelengths_nm", "times_jd"):
            grid = np.asarray(getattr(self, name))
            if grid.ndim != 1 or grid.size == 0:
                raise ValueError(f"{name} must be a nonempty one-dimensional grid")
            if (
                np.iscomplexobj(grid)
                or not np.all(np.isfinite(grid))
                or np.any(grid[1:] <= grid[:-1])
            ):
                raise ValueError(f"{name} must be finite and strictly increasing")

        contrast = np.asarray(self.contrast_grid)
        if contrast.ndim != 3 or contrast.shape[1:] != (
            self.wavelengths_nm.size,
            self.times_jd.size,
        ):
            raise ValueError(
                "contrast_grid must have shape (K, W, T) matching the grids"
            )
        if (
            np.iscomplexobj(contrast)
            or not np.all(np.isfinite(contrast))
            or np.any(contrast < 0)
        ):
            raise ValueError("contrast_grid must contain finite nonnegative values")

    def contrast(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Require physical epochs rather than substituting orbital phase."""
        raise ValueError(
            "Use contrast_at_time(..., time_jd=...) for a time-series model"
        )

    def contrast_at_time(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
        *,
        time_jd: Array,
    ) -> Array:
        """Interpolate contrast at the supplied physical epochs.

        Args:
            phase_angle_rad: Phase angle [rad]; ignored.
            dist_AU: Star-planet distance [AU]; ignored.
            wavelength_nm: Scalar wavelength, or one wavelength per query epoch.
            Rp_Rearth: Planet radius [Earth radii]; ignored.
            time_jd: Observation Julian Dates, shape ``(T_query,)``.

        Returns:
            Contrast, shape ``(K, T_query)``, with NaN outside table support.
        """
        dtype = jnp.result_type(
            wavelength_nm,
            time_jd,
            self.wavelengths_nm,
            self.times_jd,
            self.contrast_grid,
            jnp.float32,
        )
        times = jnp.atleast_1d(jnp.asarray(time_jd, dtype=dtype))
        wavelengths = jnp.broadcast_to(
            jnp.asarray(wavelength_nm, dtype=dtype), times.shape
        )
        time_grid = self.times_jd.astype(dtype)
        wavelength_grid = self.wavelengths_nm.astype(dtype)
        contrast_grid = self.contrast_grid.astype(dtype)

        def per_planet(grid):
            if time_grid.size == 1:
                if wavelength_grid.size == 1:
                    values = jnp.broadcast_to(grid[0, 0], times.shape)
                else:
                    values = interpax.interp1d(
                        wavelengths, wavelength_grid, grid[:, 0], method="linear"
                    )
            elif wavelength_grid.size == 1:
                values = interpax.interp1d(times, time_grid, grid[0], method="linear")
            else:
                values = interpax.interp2d(
                    wavelengths,
                    times,
                    wavelength_grid,
                    time_grid,
                    grid,
                    method="linear",
                )
            in_spectrum = (wavelengths >= wavelength_grid[0]) & (
                wavelengths <= wavelength_grid[-1]
            )
            return jnp.where(in_spectrum & jnp.isfinite(times), values, jnp.nan)

        return jax.vmap(per_planet)(contrast_grid)
