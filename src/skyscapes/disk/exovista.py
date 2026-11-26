"""Stored host-relative pixel contrasts with explicit native sky geometry."""

import functools

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_bulk_leaf, as_float_leaf
from hwoutils.map_coordinates import map_coordinates
from jaxtyping import Array
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Knots, as_knots
from scipy.interpolate import CubicSpline

from .base import AbstractDisk, check_sky_grid


@functools.lru_cache(maxsize=16)
def _spline_operator(knots: Knots) -> np.ndarray:
    """Cubic-spline coefficients as a linear operator of the knot values.

    Returns shape (4, n - 1, n): the polynomial coefficients of each segment
    (highest power first) are ``operator @ values``. The spline is linear in
    the values for fixed knots, so applying the operator reproduces
    ``interpax.CubicSpline(knots, values).c`` and stays differentiable in the
    values, which interpax cannot offer under a trace.
    """
    n = knots.values.size
    # Both splines use not-a-knot boundary conditions. Static host knots
    # allow this build even when the first render is traced in float32.
    return CubicSpline(knots.values, np.eye(n), axis=0).c


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

    Attributes:
        native_grid: Sky grid of the stored cube, in arcsec with a
            ``SkyOrientation``. Static.
        knots: Wavelength grid of the cube [nm], a static ``Knots``.
        contrast_cube: Host-relative contrast of each pixel (pixel flux over host
            flux, dimensionless), shape ``(n_wave, ny, nx)``. A declared bulk
            field: a float32 array given to the constructor stays float32
            under x64 and is promoted where it is read, and any other input
            becomes the active dtype. The ExoVista loader reads the float32
            cube from disk and casts it to the active dtype.
        flux_scale: Dimensionless scalar factor on the radiance, default one.
    """

    native_grid: PixelGrid = eqx.field(static=True)
    knots: Knots = eqx.field(static=True, converter=as_knots)
    contrast_cube: Array = eqx.field(converter=as_bulk_leaf)
    flux_scale: Array = eqx.field(converter=as_float_leaf)

    def __init__(self, native_grid, wavelengths_nm, contrast_cube, flux_scale=1.0):
        """Store the native geometry, the static knots and the cube.

        Args:
            native_grid: Sky grid of the stored cube, in arcsec with a
                ``SkyOrientation``.
            wavelengths_nm: 1-D wavelength knots of the cube [nm]; must be concrete.
            contrast_cube: Host-relative contrast of each pixel (dimensionless), shape
                ``(len(wavelengths_nm), ny, nx)``.
            flux_scale: Dimensionless scalar factor on the radiance; default one.
        """
        self.flux_scale = flux_scale
        self.native_grid = native_grid
        self.knots = wavelengths_nm
        self.contrast_cube = contrast_cube

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Static wavelength knots of the cube [nm]."""
        return self.knots.nm

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the stored contrast cube [nm].

        Returns:
            1-D array of the wavelength knots of the stored cube [nm].
        """
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
        self,
        wavelength_nm,
        time_d,
        east_arcsec,
        north_arcsec,
        *,
        star,
        dist_pc,
        inclination_deg,
        position_angle_deg,
    ):
        """Photon radiance in photons/s/m^2/nm/sr from the stored contrast cube.

        The cube already carries its geometry: ``inclination_deg`` and
        ``position_angle_deg`` are accepted and ignored.
        """
        east, north = jnp.broadcast_arrays(east_arcsec, north_arcsec)
        grid = self.native_grid
        pixels = grid.unit_to_pixel(*grid.orientation.sky_to_detector(east, north))
        cube = self.contrast_cube
        knots = self.knots.array()
        operator = jnp.asarray(_spline_operator(self.knots))

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
