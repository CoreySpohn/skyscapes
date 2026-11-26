"""Gray Lambertian physical model -- analytic phase, no wavelength dependence."""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import (
    nm_to_wavenumber_cm,
    spherical_to_geometric_albedo,
    wavenumber_cm_to_nm,
)
from jaxtyping import Array

from .._dtype import _working_dtype
from .._repr import fmt_scalar_or_array
from .base import AbstractPhysicalModel


def lambert_phase(phase_angle_rad: Array) -> Array:
    """Lambert phase on [0, pi], clipped at zero to suppress endpoint roundoff.

    Phi(b) = (sin(b) + (pi - b) cos(b)) / pi. Near new phase, a series in
    the distance from pi preserves the cubic crescent as the terms cancel.
    Clipping keeps reflected light nonnegative at new phase.

    Args:
        phase_angle_rad: Star-planet-observer phase angle [rad], in [0, pi].

    Returns:
        Dimensionless Lambert phase function, same shape as the input; one at
        zero phase angle.
    """
    sin_b = jnp.sin(phase_angle_rad)
    cos_b = jnp.cos(phase_angle_rad)
    direct = sin_b + (jnp.pi - phase_angle_rad) * cos_b
    # Trigonometric argument reduction preserves the distance from pi even
    # when pi itself is not representable at the input dtype.
    delta = jnp.arctan2(sin_b, -cos_b)
    d2 = delta * delta
    # sin(delta) - delta*cos(delta), through degree 13. At delta <= 1/4
    # the first omitted term is smaller than one float64 epsilon of the sum.
    series = (
        delta
        * d2
        * (
            1 / 3
            + d2
            * (
                -1 / 30
                + d2
                * (1 / 840 + d2 * (-1 / 45360 + d2 * (1 / 3991680 - d2 / 518918400)))
            )
        )
    )
    near_new = (phase_angle_rad >= jnp.pi - 0.25) & (phase_angle_rad <= jnp.pi)
    numerator = jnp.where(near_new, series, direct)
    return jnp.where(
        phase_angle_rad == jnp.pi,
        jnp.zeros_like(numerator),
        jnp.maximum(numerator / jnp.pi, 0.0),
    )


def reflected_contrast(phase_angle_rad, star_planet_distance_AU, Ag, Rp_Rearth):
    """Return Ag Phi(beta) (Rp/r)^2 for broadcast-compatible input arrays.

    Geometry comes from orbix. Radius is in Earth radii, distance in AU,
    and Ag is geometric albedo (possibly wavelength dependent).

    Args:
        phase_angle_rad: Star-planet-observer phase angle [rad].
        star_planet_distance_AU: Star-planet distance [AU].
        Ag: Geometric albedo, dimensionless; may depend on wavelength.
        Rp_Rearth: Planet radius [Earth radii].

    Returns:
        Contrast (planet-to-star flux ratio), broadcast shape of the inputs.
    """
    return (
        Ag
        * lambert_phase(phase_angle_rad)
        * (Rp_Rearth * Rearth2AU / star_planet_distance_AU) ** 2
    )


def _wavenumber_cell(nu_grid, wavelength_nm, dtype):
    """Return right cell indices, widths and offsets at the working precision."""
    n = len(nu_grid)
    host_knots = not isinstance(nu_grid, jax.Array)
    host_query = not isinstance(wavelength_nm, jax.Array)
    if host_knots and host_query:
        knots = np.asarray(nu_grid, dtype=np.float64)
        wave = np.asarray(wavelength_nm, dtype=np.float64).reshape(-1)
        with np.errstate(divide="ignore"):
            target = nm_to_wavenumber_cm(wave)
        right = np.clip(np.searchsorted(knots, target, side="right"), 1, n - 1)
        return (
            jnp.asarray(right, dtype=jnp.int32),
            jnp.asarray(knots[right] - knots[right - 1], dtype=dtype),
            jnp.asarray(target - knots[right - 1], dtype=dtype),
        )

    wave = jnp.asarray(wavelength_nm, dtype=dtype).reshape(-1)
    target = nm_to_wavenumber_cm(wave)
    if host_knots and np.dtype(dtype).itemsize < np.dtype(np.float64).itemsize:
        knots = np.asarray(nu_grid, dtype=np.float64)
        wave_knots = wavenumber_cm_to_nm(knots)
        high = wave_knots.astype(dtype)
        low = wave_knots - high
        # Floor each threshold so a working-width wavelength compares to the
        # exact host knot even when multiple knots round to the same value.
        floor = np.where(
            low < 0, np.nextafter(high, np.asarray(-np.inf, dtype=dtype)), high
        )
        right = n - jnp.searchsorted(jnp.asarray(floor[::-1]), wave, side="left")
        right = jnp.clip(jnp.where(wave < 0, 1, right), 1, n - 1)
        left = right - 1
        # nu(w) - nu(w0) = nu(w0) * (w0 - w) / w. Split the host
        # wavelength knot so the small difference survives the transfer.
        delta_nm = (jnp.asarray(high)[left] - wave) + jnp.asarray(low, dtype=dtype)[
            left
        ]
        offset = jnp.asarray(knots, dtype=dtype)[left] * (delta_nm / wave)
        return right, jnp.asarray(np.diff(knots), dtype=dtype)[left], offset

    knots = jnp.asarray(nu_grid, dtype=dtype)
    right = jnp.clip(jnp.searchsorted(knots, target, side="right"), 1, n - 1)
    return right, knots[right] - knots[right - 1], target - knots[right - 1]


def project_reflected(
    reflectivity: Array,
    nu_grid: Array,
    *,
    phase_angle_rad: Array,
    star_planet_distance_AU: Array,
    wavelength_nm: Array,
    radius_Rearth: Array,
) -> Array:
    """Reflected-light contrast cube from a plane-parallel reflectivity.

    Interpolates ``reflectivity`` onto the wavelengths (linear, with linear
    extrapolation, so values outside the grid can be negative). A zero-width
    cell returns the right sample. The projection converts the
    spherical albedo to geometric albedo with the Lambertian-sphere factor
    2/3 (Seager 2010, eq 3.36), and applies Lambert phase and ``(Rp/d)^2``.

    Args:
        reflectivity: Plane-parallel reflectivity ``R(nu)``, shape ``(K, n_nu)``.
        nu_grid: Wavenumber grid [cm^-1], shape ``(n_nu,)``.
        phase_angle_rad: Star-planet-observer phase angle, shape ``(K, T)``.
        star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
        wavelength_nm: Wavelengths [nm], shape ``(W,)``.
        radius_Rearth: Planet radius [Earth radii], shape ``(K,)``.

    Returns:
        Contrast, shape ``(W, K, T)``.
    """
    dtype = _working_dtype(
        reflectivity,
        wavelength_nm,
        phase_angle_rad,
        star_planet_distance_AU,
        radius_Rearth,
        nu_grid if isinstance(nu_grid, jax.Array) else None,
    )
    spectrum = jnp.asarray(reflectivity, dtype=dtype)
    if len(nu_grid) == 1:
        nw = np.size(wavelength_nm)
        sampled = jnp.broadcast_to(spectrum, (spectrum.shape[0], nw))
    else:
        right, width, offset = _wavenumber_cell(nu_grid, wavelength_nm, dtype)
        y0, y1 = spectrum[:, right - 1], spectrum[:, right]
        degenerate = width == 0
        fraction = offset / jnp.where(degenerate, jnp.ones_like(width), width)
        sampled = jnp.where(degenerate, y1, y0 + fraction * (y1 - y0))
    albedo = spherical_to_geometric_albedo(sampled)
    return reflected_contrast(
        phase_angle_rad[None, :, :],
        star_planet_distance_AU[None, :, :],
        albedo.T[:, :, None],
        radius_Rearth[None, :, None],
    )


class LambertianPhysicalModel(AbstractPhysicalModel):
    """Lambertian reflector.

    Attributes:
        Ag: Geometric albedo, shape ``(K,)``.
    """

    Ag: Array

    def contrast(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Contrast = Ag * Phi(beta) * (Rp/r)**2. Wavelength-independent.

        ``phase_angle_rad`` and ``star_planet_distance_AU`` are shape ``(K, T)``.
        ``wavelength_nm`` is part of the interface but ignored in the
        gray case. ``Rp_Rearth`` is shape ``(K,)``, supplied by the
        host ``Planet``.

        Args:
            phase_angle_rad: Star-planet-observer phase angle [rad], shape ``(K, T)``.
            star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
            wavelength_nm: Wavelength [nm]; ignored.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.

        Returns:
            Contrast (planet-to-star flux ratio), shape ``(K, T)``.
        """
        return reflected_contrast(
            phase_angle_rad,
            star_planet_distance_AU,
            self.Ag[:, None],
            Rp_Rearth[:, None],
        )

    def __repr__(self) -> str:
        """One-line summary of geometric albedo.

        Returns:
            One-line summary string.
        """
        ag = fmt_scalar_or_array(self.Ag)
        return f"LambertianPhysicalModel(Ag={ag})"
