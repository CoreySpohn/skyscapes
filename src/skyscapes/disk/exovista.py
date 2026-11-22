"""Stored host-relative pixel contrasts with explicit native sky geometry."""

import functools

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.map_coordinates import map_coordinates
from jaxtyping import Array
from optixstuff.grids import PixelGrid

from .base import AbstractDisk, check_sky_grid


class _Knots:
    """Hashable, immutable wavelength knots, so they can be a static field."""

    __slots__ = ("_hash", "values")

    def __init__(self, values):
        values = np.array(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
            values = np.array(values, dtype=float).reshape(-1)
        values.flags.writeable = False
        self.values = values
        self._hash = hash((values.dtype.str, values.tobytes()))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, _Knots) and (
            self.values.dtype == other.values.dtype
            and np.array_equal(self.values, other.values)
        )


@functools.lru_cache(maxsize=16)
def _spline_operator(knots: _Knots) -> np.ndarray:
    """Cubic-spline coefficients as a linear operator of the knot values.

    Returns shape (4, n - 1, n): the polynomial coefficients of each segment
    (highest power first) are ``operator @ values``. The spline is linear in
    the values for fixed knots, so applying the operator reproduces
    ``interpax.CubicSpline(knots, values).c`` and stays differentiable in the
    values, which interpax cannot offer under a trace.
    """
    n = knots.values.size
    # The build is concrete even when the first render is being traced.
    with jax.ensure_compile_time_eval(), jax.enable_x64(True):
        spline = interpax.CubicSpline(
            np.asarray(knots.values, dtype=np.float64), np.eye(n), axis=0
        )
        return np.asarray(spline.c)


class ExovistaDisk(AbstractDisk):
    """Wavelength-interpolated contrast cube on a native PixelGrid.

    contrast_cube holds F_pixel/Phi_star, shape (n_wave, ny, nx). Since
    F_pixel = L * Omega_native and Omega_native = native_grid.cell_area("sr");
    L = contrast_cube * Phi_star / Omega_native.
    Spatial sampling is bilinear, with zero extension of the native samples.
    There is no target pixel area factor. Geometry is baked into the cube;
    system angles are ignored. ``flux_scale`` is a dimensionless factor on the
    radiance, default one; see :class:`~skyscapes.disk.AbstractDisk`.

    The cube is the only stored representation and a live leaf. The cubic
    spline between wavelengths is a fixed linear operator of the cube (the
    knots are static), applied to the cube on each call, so ``eqx.tree_at`` on
    ``contrast_cube`` reaches the image. The wavelength knots are a static
    grid, not a parameter, and must be concrete when the disk is built.
    """

    native_grid: PixelGrid = eqx.field(static=True)
    _knots_nm: _Knots = eqx.field(static=True)
    contrast_cube: Array
    flux_scale: Array

    def __init__(self, native_grid, wavelengths_nm, contrast_cube, flux_scale=1.0):
        """Store the native geometry, the static knots and the cube."""
        self.flux_scale = jnp.asarray(flux_scale)
        self.native_grid = native_grid
        self._knots_nm = _Knots(wavelengths_nm)
        self.contrast_cube = contrast_cube

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Static wavelength knots of the cube [nm]."""
        return self._knots_nm.values

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the stored contrast cube [nm]."""
        return self.wavelengths_nm

    def __check_init__(self):
        """Validate the declared disk inputs."""
        check_sky_grid(self.native_grid)
        if self.contrast_cube.shape != (
            len(self.wavelengths_nm),
            *self.native_grid.shape,
        ):
            raise ValueError("contrast_cube must match wavelengths and native_grid")

    def _photon_radiance_at(
        self, wavelength_nm, time_d, east_arcsec, north_arcsec, *, star, dist_pc
    ):
        """Photon radiance in photons/s/m^2/nm/sr from the stored contrast cube."""
        east, north = jnp.broadcast_arrays(east_arcsec, north_arcsec)
        grid = self.native_grid
        pixels = grid.unit_to_pixel(*grid.orientation.sky_to_detector(east, north))
        cube = self.contrast_cube
        knots = jnp.asarray(self.wavelengths_nm)
        operator = jnp.asarray(_spline_operator(self._knots_nm))

        def at_wavelength(wave):
            # A wavelength needs only the four planes from its spline segment.
            index = jnp.clip(
                jnp.searchsorted(knots, wave, side="right") - 1,
                0,
                knots.size - 2,
            )
            segment = jax.lax.dynamic_slice_in_dim(knots, index, 2)
            dtype = jnp.result_type(cube, wave)
            rows = operator[:, index].astype(dtype)
            planes = jnp.einsum("km,myx->kyx", rows, cube.astype(dtype))

            def sample(plane):
                return map_coordinates(plane, (pixels.y, pixels.x), order=1)

            # Spatial interpolation commutes with the spectral polynomial.
            coefficients = jax.vmap(sample)(planes)
            sampled = interpax.PPoly.construct_fast(
                coefficients[:, None], segment, True
            )
            return sampled(wave)

        wave = jnp.asarray(wavelength_nm, dtype=float)
        contrast = jax.vmap(at_wavelength)(wave.reshape(-1))
        contrast = contrast.reshape(wave.shape + east.shape)
        flux = star.photon_flux_density(wave, time_d, dist_pc=dist_pc)
        flux = flux.reshape(wave.shape + (1,) * east.ndim)
        return contrast * flux / grid.cell_area("sr")
