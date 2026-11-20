"""Shared LOS-around-midplane integration kernel for parametric disks.

Coordinates are explicit east/north offsets in AU; sampling belongs to the caller.

Gauss-Legendre nodes cover the part of each line of sight where the density
is not negligible, including edge-on views.

References:
    - Augereau, J.-C., Lagrange, A.-M., Mouillet, D., Papaloizou, J. C. B.,
      & Grorod, P. A. 1999, A&A, 348, 557 (the radiative transfer).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import deg2rad
from jaxtyping import Array

DensityFn = Callable[[Array, Array, Array], Array]
PhaseFn = Callable[[Array], Array]
ScaleHeightFn = Callable[[Array], Array]

# The vertical profile exp(-(|z| / h)^gamma) is below exp(-16) = 1.1e-7 of
# its midplane value outside 16^(1 / gamma) scale heights (4 for a Gaussian).
_LOG_TRUNCATION = 16.0

_SMALL = 1e-9


def _midplane_crossing(
    x_rot: Array,
    y_rot: Array,
    cos_i: Array,
    sin_i: Array,
    rmax_AU: Array,
) -> tuple[Array, Array, Array]:
    """Depth of the midplane crossing, its radius capped at rmax, safe cos."""
    safe_cos = jnp.where(jnp.abs(cos_i) < _SMALL, _SMALL, cos_i)
    l_cross = y_rot * sin_i / safe_cos
    r_cross = jnp.sqrt(x_rot**2 + (y_rot / safe_cos) ** 2 + 1e-12)
    return l_cross, jnp.minimum(r_cross, rmax_AU), safe_cos


def _legendre_nodes(
    x_rot: Array,
    y_rot: Array,
    cos_i: Array,
    sin_i: Array,
    rmax_AU: Array,
    scale_height_fn: ScaleHeightFn,
    vertical_exponent: Array | float,
    n: int,
) -> tuple[Array, Array]:
    """Depths and path weights of Gauss-Legendre on the exact LOS window."""
    l_cross, r_cross, _ = _midplane_crossing(x_rot, y_rot, cos_i, sin_i, rmax_AU)
    h_cross, slope = jax.jvp(scale_height_fn, (r_cross,), (jnp.ones_like(r_cross),))
    n_h = _LOG_TRUNCATION ** (1.0 / vertical_exponent)
    # |z| <= n_h h(r), with h bounded by its tangent at the crossing radius.
    # When the denominator closes the ray never leaves the flared layer and
    # the window becomes the whole chord of the outer cylinder.
    closing = jnp.abs(cos_i) - n_h * slope * jnp.abs(sin_i)
    half = n_h * h_cross / jnp.maximum(closing, _SMALL)

    chord_sq = rmax_AU**2 - x_rot**2
    inside = chord_sq > 0.0
    half_chord = jnp.where(inside, jnp.sqrt(jnp.where(inside, chord_sq, 1.0)), 0.0)
    safe_sin = jnp.where(jnp.abs(sin_i) < _SMALL, _SMALL, sin_i)
    chord_a = (-half_chord - y_rot * cos_i) / safe_sin
    chord_b = (half_chord - y_rot * cos_i) / safe_sin

    low = jnp.maximum(l_cross - half, jnp.minimum(chord_a, chord_b))
    high = jnp.minimum(l_cross + half, jnp.maximum(chord_a, chord_b))
    width = jnp.maximum(high - low, 0.0)
    u, w = np.polynomial.legendre.leggauss(n)
    u = jnp.asarray(0.5 * (u + 1.0), dtype=width.dtype)
    w = jnp.asarray(0.5 * w, dtype=width.dtype)
    node_shape = (n,) + (1,) * width.ndim
    return low + u.reshape(node_shape) * width, w.reshape(node_shape) * width


def los_integrate_scattered(
    density_fn: DensityFn,
    phase_fn: PhaseFn,
    *,
    incl_deg: Array,
    position_angle_deg: Array,
    rmin_AU: Array,
    rmax_AU: Array,
    east_AU: Array,
    north_AU: Array,
    n_slices_los: int,
    scale_height_fn: ScaleHeightFn,
    vertical_exponent: Array | float = 2.0,
) -> Array:
    """Integrate scattered light along LOS around the disk midplane.

    Private integration chart: (east, north, toward), with depth ``l_grid``
    positive toward the observer (the negative of public sky z). Sky ``x``
    is east (from ``Scene.positions``), and sky ``y`` is north.
    The line of nodes is (sin(PA), cos(PA)) in (east, north), with
    astronomical position angle PA north through east. The near half for
    inclinations below 90 degrees is (-cos(PA), sin(PA)); above 90 it
    reverses. Inclination zero is face-on. Path lengths are always positive.

    Write ``y_d = y cos(i) + l sin(i)`` and
    ``z_d = -y sin(i) + l cos(i)``. The midplane crossing is at depth
    ``l0 = y tan(i)`` and radius ``r0``, and one scale height ``h`` of the
    layer spans ``h / |cos(i)|`` in depth. Radiative transfer follows
    Augereau et al. (1999), A&A 348, 557.

    Gauss-Legendre uses ``n_slices_los`` nodes on ``|z_d| <= H h(r)``,
    where ``H = 16^(1 / gamma)`` makes the vertical profile smaller than
    ``exp(-16)`` outside the window. Bounding ``h`` by its tangent at ``r0``
    gives ``l0 +- H h(r0) / (|cos(i)| - H h'(r0) |sin(i)|)``, intersected
    with the chord of the ``rmax_AU`` cylinder. The bound is exact for a
    concave ``h`` (flaring index at most 1); a convex ``h`` can lose part
    of the outer layer. When the denominator closes the whole chord is
    used. The window is finite and smooth through edge-on.

    Hard radial masks remain piecewise differentiable at sample crossings
    and leave errors of 1e-4 to 7e-4 of the peak under every rule; a density
    that tapers to zero at both radii removes them.

    Args:
        density_fn: ``(r_AU, z_AU, valid) -> rho``. The kernel ensures
            ``r_AU`` is finite everywhere (sqrt is masked) but the
            caller must substitute density-safe values at invalid
            points (e.g. avoid ``r=0`` for ``r^alpha`` profiles).
        phase_fn: ``cos_phi -> phase`` evaluated at every LOS sample.
        incl_deg: Disk inclination [deg].
        position_angle_deg: Major-axis angle north through east [deg].
        rmin_AU: Inner truncation radius [AU].
        rmax_AU: Outer truncation radius [AU].
        east_AU: East offsets in AU, broadcastable with north_AU.
        north_AU: North offsets in AU, broadcastable with east_AU.
        n_slices_los: Number of Gauss-Legendre LOS nodes (static).
        scale_height_fn: ``r_AU -> h_AU``, the scale height in
            ``exp(-(|z| / h)^gamma)``, differentiable in ``r``.
        vertical_exponent: ``gamma`` of the vertical profile. A Python
            number is a compile-time constant; an array stays
            differentiable.

    Returns:
        Scattered-light integral with the broadcast shape of the sky offsets.
        Integral of rho * phase / distance_AU**2 over depth_AU.
    """
    incl = incl_deg * deg2rad
    cos_i, sin_i = jnp.cos(incl), jnp.sin(incl)
    pa = position_angle_deg * deg2rad
    x_rot = jnp.sin(pa) * east_AU + jnp.cos(pa) * north_AU
    y_rot = -jnp.cos(pa) * east_AU + jnp.sin(pa) * north_AU

    l_grid, weights = _legendre_nodes(
        x_rot,
        y_rot,
        cos_i,
        sin_i,
        rmax_AU,
        scale_height_fn,
        vertical_exponent,
        n_slices_los,
    )

    y_d = y_rot * cos_i + l_grid * sin_i
    z_d = -y_rot * sin_i + l_grid * cos_i

    xy_sq = x_rot * x_rot + y_d * y_d
    d_star_sq = xy_sq + z_d * z_d
    # Relative guard band on the truncation edges. The dense midplane node lands
    # the deprojected inner-edge pixels within a ULP of rmin_sq, and xy_sq is
    # rounded differently under jit vs eager on x86 (FMA contraction of
    # x_rot*x_rot + y_d*y_d), so a hard >=/<= flips that node's high-density
    # contribution on/off between the two paths -- a platform-dependent jit/eager
    # mismatch. A 1e-9 band sits far above 1-ULP noise and costs a negligible
    # sliver of disk radius.
    edge = 1e-9
    rmin_sq = rmin_AU * rmin_AU * (1.0 - edge)
    rmax_sq = rmax_AU * rmax_AU * (1.0 + edge)
    valid = (xy_sq >= rmin_sq) & (xy_sq <= rmax_sq) & (d_star_sq > 0.0)

    # Sqrt only inside the valid annulus; the caller's density_fn handles
    # any further safe substitution before evaluating its formula.
    xy_sq_safe = jnp.where(valid, xy_sq, 1.0)
    r_AU = jnp.sqrt(xy_sq_safe)
    rho = density_fn(r_AU, z_d, valid)

    # Scattering cosine: private toward-depth / distance = -public sky z / r.
    safe_d_sq = jnp.where(d_star_sq > 0.0, d_star_sq, 1.0)
    safe_d = jnp.sqrt(safe_d_sq)
    cos_phi = l_grid / safe_d
    phase = phase_fn(cos_phi)

    integrand = jnp.where(valid, rho * phase / safe_d_sq, 0.0)
    return (weights * integrand).sum(axis=0)
