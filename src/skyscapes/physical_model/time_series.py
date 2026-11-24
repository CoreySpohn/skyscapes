"""Bounded interpolation of planet contrast at physical observation epochs."""

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import HostGrid, as_bulk_leaf, as_float_leaf
from jaxtyping import Array
from optixstuff.spectral import Knots

from .._tables import as_host_grid, knots_of
from .base import AbstractPhysicalModel


def _wavelength_grid(values) -> HostGrid:
    """The static wavelength grid, refusals naming the field."""
    try:
        return as_host_grid(values)
    except ValueError as error:
        raise ValueError(f"wavelengths_nm: {error}") from None


class TimeSeriesPhysicalModel(AbstractPhysicalModel):
    """Tabulated planet-to-star contrast over wavelength and days from the scene epoch.

    Linear interpolation preserves nonnegative contrast inside the table.
    Multiple epochs define bounded temporal support: queries before the
    first epoch or after the last return NaN. A single epoch explicitly
    represents a constant snapshot at any time. Wavelength support is
    always bounded; a single wavelength is valid only at that wavelength.

    Phase, distance, and radius are ignored because their effects are
    already included in the tabulated ratios. Equal orbital phases do not
    imply equal contrasts at different epochs.

    The wavelength grid is static host data (a ``HostGrid``, float64), and
    the table's ``knots`` are that grid; a table of one wavelength has no
    knots and cannot be integrated over a band.

    Attributes:
        wavelengths_nm: Finite, strictly increasing grid [nm], shape ``(W,)``,
            static.
        times_d: Finite, strictly increasing days from the scene epoch,
            shape ``(T,)``. The owning scene supplies the epoch metadata.
        contrast_grid: Finite, nonnegative flux ratios, shape ``(K, W, T)``.
            A declared bulk field: ExoVista planet contrast cubes are float32
            on disk, and a float32 cube is retained under x64.
    """

    wavelengths_nm: HostGrid = eqx.field(static=True, converter=_wavelength_grid)
    times_d: Array = eqx.field(converter=as_float_leaf)
    contrast_grid: Array = eqx.field(converter=as_bulk_leaf)

    @property
    def knots(self) -> Knots:
        """The wavelength grid as ``Knots``; refused for a single wavelength."""
        return knots_of(self.wavelengths_nm)

    @property
    def _tabulated(self) -> bool:
        """A time series is always a table, even one it cannot integrate."""
        return True

    def __check_init__(self):
        """Reject invalid table coordinates and contrasts before evaluation."""
        for name in ("wavelengths_nm", "times_d"):
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
            len(self.wavelengths_nm),
            self.times_d.size,
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

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the time-dependent contrast table [nm].

        Returns:
            1-D array of the table wavelengths [nm], float64 host data.
        """
        return self.wavelengths_nm.values

    def contrast(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Require physical epochs rather than substituting orbital phase.

        Args:
            phase_angle_rad: Phase angle [rad]; unused.
            star_planet_distance_AU: Star-planet distance [AU]; unused.
            wavelength_nm: Wavelength [nm]; unused.
            Rp_Rearth: Planet radius [Earth radii]; unused.

        Returns:
            Nothing; this method always raises ``ValueError``.
        """
        raise ValueError(
            "Use contrast_at_time(..., time_d=...) for a time-series model"
        )

    def contrast_at_time(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
        *,
        time_d: Array,
        star=None,
        dist_pc: Array | None = None,
    ) -> Array:
        """Interpolate contrast at the supplied physical epochs.

        Args:
            phase_angle_rad: Phase angle [rad]; ignored.
            star_planet_distance_AU: Star-planet distance [AU]; ignored.
            wavelength_nm: Scalar wavelength, or one wavelength per query epoch.
            Rp_Rearth: Planet radius [Earth radii]; ignored.
            time_d: Observation times in days from the scene epoch,
                shape ``(T_query,)``.
            star: The host star; ignored.
            dist_pc: Observer distance [pc]; ignored.

        Returns:
            Contrast, shape ``(K, T_query)``, with NaN outside table support.
        """
        grid = self.wavelengths_nm.array()
        dtype = jnp.result_type(
            wavelength_nm,
            time_d,
            grid,
            self.times_d,
            self.contrast_grid,
            jnp.float32,
        )
        times = jnp.atleast_1d(jnp.asarray(time_d, dtype=dtype))
        wavelengths = jnp.broadcast_to(
            jnp.asarray(wavelength_nm, dtype=dtype), times.shape
        )
        time_grid = self.times_d.astype(dtype)
        wavelength_grid = grid.astype(dtype)
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
