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
from hwoutils.conversions import arcsec_to_au
from hwoutils.numerics import COARSEST
from jaxtyping import Array

DensityFn = Callable[[Array, Array, Array], Array]
PhaseFn = Callable[[Array], Array]
ScaleHeightFn = Callable[..., Array | tuple[Array, Array]]

# The vertical profile exp(-(|z| / h)^gamma) is below exp(-16) = 1.1e-7 of
# its midplane value outside 16^(1 / gamma) scale heights (4 for a Gaussian).
_LOG_TRUNCATION = 16.0


def _sincos_deg(degrees, xp):
    """Reduce in degrees so supplementary angles share the same trig arguments."""
    quadrant = xp.rint(degrees / 90.0)
    angle = (degrees - 90.0 * quadrant) * deg2rad
    sine, cosine = xp.sin(angle), xp.cos(angle)
    quarter = (quadrant % 4).astype(np.int32)
    odd = (quarter % 2) == 1
    one = xp.asarray(1.0, dtype=sine.dtype)
    return (
        xp.where(quarter < 2, one, -one) * xp.where(odd, cosine, sine),
        xp.where((quarter == 0) | (quarter == 3), one, -one)
        * xp.where(odd, sine, cosine),
    )


def _midplane_crossing(
    x_rot: Array,
    y_rot: Array,
    cos_i: Array,
    sin_i: Array,
    rmax_AU: Array,
    xp,
) -> tuple[Array, Array]:
    """Depth of the midplane crossing and its radius capped at rmax."""
    safe_cos = xp.where(cos_i == 0, xp.ones_like(cos_i), cos_i)
    l_cross = y_rot * sin_i / safe_cos
    r_cross = xp.sqrt(x_rot**2 + (y_rot / safe_cos) ** 2 + 1e-12)
    return l_cross, xp.minimum(r_cross, rmax_AU)


def _legendre_nodes(
    x_rot: Array,
    y_rot: Array,
    cos_i: Array,
    sin_i: Array,
    rmax_AU: Array,
    scale_height_fn: ScaleHeightFn,
    vertical_exponent: Array | float,
    nodes_weights,
    height_args,
    xp,
) -> tuple[Array, Array]:
    """Depths and path weights of Gauss-Legendre on the exact LOS window."""
    l_cross, r_cross = _midplane_crossing(x_rot, y_rot, cos_i, sin_i, rmax_AU, xp)
    if height_args is None:
        h_cross, slope = jax.jvp(scale_height_fn, (r_cross,), (xp.ones_like(r_cross),))
    else:
        h_cross, slope = scale_height_fn(r_cross, *height_args)
    n_h = _LOG_TRUNCATION ** (1.0 / vertical_exponent)
    # |z| <= n_h h(r), with h bounded by its tangent at the crossing radius.
    # When the denominator closes the ray never leaves the flared layer and
    # the window becomes the whole chord of the outer cylinder.
    closing = xp.abs(cos_i) - n_h * slope * xp.abs(sin_i)
    bounded_layer = (closing > 0) & (cos_i != 0)
    half = n_h * h_cross / xp.where(bounded_layer, closing, xp.ones_like(closing))

    chord_sq = rmax_AU**2 - x_rot**2
    inside = chord_sq > 0.0
    chord = xp.sqrt(xp.where(inside, chord_sq, xp.ones_like(chord_sq)))
    half_chord = xp.where(inside, chord, xp.zeros_like(chord))
    safe_sin = xp.where(sin_i == 0, xp.ones_like(sin_i), sin_i)
    chord_a = (-half_chord - y_rot * cos_i) / safe_sin
    chord_b = (half_chord - y_rot * cos_i) / safe_sin

    # A face-on ray has constant cylindrical radius; only the vertical
    # window limits its depth. At edge-on a flared layer can span the chord.
    chord_low, chord_high = xp.minimum(chord_a, chord_b), xp.maximum(chord_a, chord_b)
    low = xp.where(sin_i == 0, l_cross - half, chord_low)
    high = xp.where(sin_i == 0, l_cross + half, chord_high)
    low = xp.where(bounded_layer, xp.maximum(low, l_cross - half), low)
    high = xp.where(bounded_layer, xp.minimum(high, l_cross + half), high)
    width = xp.maximum(high - low, 0.0)
    u, w = nodes_weights
    u = xp.asarray(0.5 * (u + 1.0), dtype=width.dtype)
    w = xp.asarray(0.5 * w, dtype=width.dtype)
    node_shape = (len(u),) + (1,) * width.ndim
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
    height_args: tuple | None = None,
    sky_offsets_arcsec: tuple | None = None,
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
        height_args: Explicit profile parameters for a scale-height function
            returning both height and radial derivative.
        sky_offsets_arcsec: Original east and north sky offsets in arcseconds and
            distance in parsecs, when the caller converted angular coordinates.

    Returns:
        Scattered-light integral with the broadcast shape of the sky offsets.
        Integral of rho * phase / distance_AU**2 over depth_AU.
    """
    geometry = (incl_deg, position_angle_deg, rmin_AU, rmax_AU, vertical_exponent)

    nodes_weights = np.polynomial.legendre.leggauss(n_slices_los)

    def samples(geometry, east, north, height_args, xp):
        inclination, pa, _, outer, exponent = geometry
        sin_i, cos_i = _sincos_deg(inclination, xp)
        sin_pa, cos_pa = _sincos_deg(pa, xp)
        x_rot = sin_pa * east + cos_pa * north
        y_rot = -cos_pa * east + sin_pa * north
        depth, weights = _legendre_nodes(
            x_rot,
            y_rot,
            cos_i,
            sin_i,
            outer,
            scale_height_fn,
            exponent,
            nodes_weights,
            height_args,
            xp,
        )
        y_d = y_rot * cos_i + depth * sin_i
        z_d = -y_rot * sin_i + depth * cos_i
        xy_sq = x_rot * x_rot + y_d * y_d
        d_sq = xy_sq + z_d * z_d
        return depth, weights, z_d, xy_sq, d_sq

    east, north = jnp.asarray(east_AU), jnp.asarray(north_AU)
    working_dtype = jnp.result_type(east, north, *geometry)
    working_geometry = (
        jnp.asarray(incl_deg, dtype=working_dtype),
        jnp.asarray(position_angle_deg, dtype=working_dtype),
        *geometry[2:],
    )
    l_grid, weights, z_d, xy_sq, d_star_sq = samples(
        working_geometry, east, north, height_args, jnp
    )
    # Radial membership is decided on one float32 grid for every query, host or
    # runtime, in either working precision. A host query skips the device only
    # when its geometry is concrete; the rule is the same.
    sky = (east_AU, north_AU) if sky_offsets_arcsec is None else sky_offsets_arcsec

    def host_values(tree):
        return not any(isinstance(x, jax.Array) for x in jax.tree.leaves(tree))

    # Neither the position angle nor the vertical law affects a face-on radius.
    host_face_on = (
        host_values((incl_deg, rmin_AU, rmax_AU, sky)) and float(incl_deg) % 180 == 0
    )
    host_geometry = height_args is not None and host_values(
        (geometry, sky, height_args)
    )
    xp = np if host_face_on or host_geometry else jnp
    dtype = COARSEST
    coords = tuple(xp.asarray(x, dtype=dtype) for x in sky)
    east, north = coords[:2]
    if sky_offsets_arcsec is not None:
        east, north = arcsec_to_au(east, coords[2]), arcsec_to_au(north, coords[2])

    if host_face_on:
        radial_sq = east * east + north * north
        inner, outer = (xp.asarray(x, dtype=dtype) for x in (rmin_AU, rmax_AU))
        distance_sq = d_star_sq
    else:
        geom, args = jax.tree.map(
            lambda x: xp.asarray(x, dtype=dtype), (geometry, height_args)
        )
        _, _, _, projected_sq, distance_sq = samples(geom, east, north, args, xp)
        inclination, _, inner, outer, _ = geom
        radial_sq = xp.where(inclination % 180 == 0, east**2 + north**2, projected_sq)
    # Compiled and eager float32 arithmetic round the squared radius
    # differently (fused multiply-add), so each edge keeps a band of four
    # float32 epsilons, on host and runtime paths alike.
    band = 4 * float(np.finfo(COARSEST).eps)
    valid = (
        (radial_sq >= inner**2 * (1 - band))
        & (radial_sq <= outer**2 * (1 + band))
        & (distance_sq > 0)
    )

    # Sqrt only inside the valid annulus; the caller's density_fn handles
    # any further safe substitution before evaluating its formula.
    xy_sq_safe = jnp.where(valid, xy_sq, jnp.ones_like(xy_sq))
    r_AU = jnp.sqrt(xy_sq_safe)
    rho = density_fn(r_AU, z_d, valid)

    # Scattering cosine: private toward-depth / distance = -public sky z / r.
    safe_d_sq = jnp.where(d_star_sq > 0.0, d_star_sq, jnp.ones_like(d_star_sq))
    safe_d = jnp.sqrt(safe_d_sq)
    cos_phi = l_grid / safe_d
    phase = phase_fn(cos_phi)

    integrand = rho * phase / safe_d_sq
    integrand = jnp.where(valid, integrand, jnp.zeros_like(integrand))
    return (weights * integrand).sum(axis=0)
