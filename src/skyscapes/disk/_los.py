"""Shared LOS-around-midplane integration kernel for parametric disks.

Coordinates are explicit east/north offsets in AU; sampling belongs to the caller.

Three families of line-of-sight rule share one integrand
(:data:`QUADRATURES`). They differ only in where the nodes sit along the
line of sight and what weight each carries, so they converge to the same
integral:

``"legendre"``
    Gauss-Legendre on the part of the line of sight where the density is
    not negligible. Valid at every inclination, including edge-on.
``"gauss"``, ``"gauss_half"``
    A Gauss rule for the vertical profile ``exp(-|s|^gamma)`` in the
    offset from the midplane crossing. A handful of nodes suffice, but only
    while the radial profile varies slowly across one scale height of the
    layer (:func:`layer_expansion_parameter` below
    :data:`GAUSS_EPS_MAX`).
``"trapezoid"``
    The trapezoid rule on log-spaced nodes that earlier releases used,
    kept unchanged for reproducibility. It converges as ``1 / n^2``.

References:
    - Augereau, J.-C., Lagrange, A.-M., Mouillet, D., Papaloizou, J. C. B.,
      & Grorod, P. A. 1999, A&A, 348, 557 (the radiative transfer).
    - Golub, G. H., & Welsch, J. H. 1969, Math. Comp., 23, 221 (Gauss rules
      from the moments of a weight).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import deg2rad
from jax.scipy.special import gammaln as jax_gammaln
from jaxtyping import Array
from scipy.special import gammaln as scipy_gammaln

DensityFn = Callable[[Array, Array, Array], Array]
PhaseFn = Callable[[Array], Array]
ScaleHeightFn = Callable[[Array], Array]

QUADRATURES = ("legendre", "gauss", "gauss_half", "trapezoid")

# Largest layer expansion parameter at which the midplane-centered Gauss
# rules hold about 1e-3 of the image peak with 7 nodes.
GAUSS_EPS_MAX = 0.45

# The moment matrix of the Golub-Welsch construction loses about one digit
# per node. These caps keep the rule itself accurate to better than 1e-8 in
# float64 for vertical exponents of one and above.
_GAUSS_MAX_NODES = {"gauss": 12, "gauss_half": 8}
# The same construction in float32 (a traced exponent with 64-bit floats
# disabled) keeps about 1e-6 up to these counts and fails soon after.
_GAUSS_MAX_NODES_FLOAT32 = {"gauss": 8, "gauss_half": 4}

# The vertical profile exp(-(|z| / h)^gamma) is below exp(-16) = 1.1e-7 of
# its midplane value outside 16^(1 / gamma) scale heights (4 for a Gaussian).
_LOG_TRUNCATION = 16.0

# Concentration of the log-spaced LOS nodes about the midplane crossing.
# Matches the GRaTeR-JAX reference (lwidth = 100): larger packs more nodes
# near the dense midplane, which is what keeps the integral converged as the
# LOS window (zmax / |cos_i|) stretches toward edge-on.
_LOS_LWIDTH = 100.0

_SMALL = 1e-9


def layer_expansion_parameter(
    log_slope: Array, aspect_ratio: Array, incl_deg: Array
) -> Array:
    """Expansion parameter of a line-of-sight rule centered on the midplane.

    Along a line of sight the vertical profile is a bump of width
    ``h / |cos(i)|`` at the midplane crossing. Across one scale height of
    that bump the radius changes by ``h tan(i)``, so the logarithm of a
    radial power law ``r^alpha`` changes by
    ``eps = |alpha| (h / r) |tan(i)|``. The ``"gauss"`` rules treat the
    radial profile as a slowly varying factor and need ``eps`` below about
    :data:`GAUSS_EPS_MAX`; beyond it use ``"legendre"``.

    Args:
        log_slope: Steepest logarithmic slope ``d ln(rho) / d ln(r)`` of the
            radial profile (sign ignored).
        aspect_ratio: Scale height over radius ``h / r`` where the disk is
            bright, with ``h`` defined by ``exp(-(|z| / h)^gamma)``.
        incl_deg: Inclination [deg].

    Returns:
        ``eps``, dimensionless and nonnegative.
    """
    return jnp.abs(log_slope * aspect_ratio * jnp.tan(incl_deg * deg2rad))


def gauss_rule(
    gamma: Array | float, n: int, *, half_range: bool = False
) -> tuple[Array | np.ndarray, Array | np.ndarray]:
    """Gauss rule for the weight ``exp(-|s|^gamma)``.

    The nodes ``s_j`` and weights ``w_j`` satisfy
    ``sum_j w_j p(s_j) = integral p(s) exp(-|s|^gamma) ds`` for every
    polynomial ``p`` of degree below ``2 n``, over the real line or, with
    ``half_range``, over ``s > 0``. ``gamma = 2`` gives Gauss-Hermite and
    ``gamma = 1`` with ``half_range`` gives Gauss-Laguerre.

    The rule is built from the moments
    ``integral s^k exp(-s^gamma) ds = Gamma((k + 1) / gamma) / gamma`` on
    ``s > 0`` by the method of Golub & Welsch (1969): the Cholesky factor of
    the Hankel moment matrix gives the three-term recurrence, and the
    eigenvalues and first eigenvector components of its Jacobi matrix give
    the nodes and weights. Every step is differentiable, so a traced
    ``gamma`` keeps its gradient.

    Args:
        gamma: Exponent of the weight. A Python number builds the rule with
            NumPy in float64 at trace time; an array builds it with JAX in
            the widest enabled float type.
        n: Number of nodes (static). The moment matrix loses about one
            digit per node: at most 12 (8 with ``half_range``) in float64
            and 8 (4) in float32.
        half_range: Integrate over ``s > 0`` instead of the real line.

    Returns:
        ``(nodes, weights)``, each of shape ``(n,)``, nodes ascending.

    Raises:
        ValueError: If ``n`` exceeds what the float type can carry.
    """
    name = "gauss_half" if half_range else "gauss"
    if isinstance(gamma, (int, float)):
        xp, gammaln = np, scipy_gammaln
        gamma = np.float64(gamma)
        cap = _GAUSS_MAX_NODES[name]
    else:
        xp, gammaln = jnp, jax_gammaln
        dtype = jax.dtypes.canonicalize_dtype(float)
        gamma = jnp.asarray(gamma, dtype=dtype)
        wide = dtype == np.float64
        cap = (_GAUSS_MAX_NODES if wide else _GAUSS_MAX_NODES_FLOAT32)[name]
    if n > cap:
        raise ValueError(
            f"The {name!r} rule supports at most {cap} nodes here, got {n}: its "
            "moment matrix loses about one digit per node. With 64-bit floats "
            "disabled, enable them or fix the vertical exponent as a Python "
            "number to lift the limit to "
            f"{_GAUSS_MAX_NODES[name]}. The error of this rule is set by the "
            "inclination, not the node count; use 'legendre' for more accuracy."
        )
    order = xp.arange(2 * n + 1)
    moments = xp.exp(gammaln((order + 1.0) / gamma)) / gamma
    if not half_range:
        moments = xp.where(order % 2 == 0, 2.0 * moments, 0.0)
    idx = xp.arange(n + 1)
    chol = xp.linalg.cholesky(moments[idx[:, None] + idx[None, :]]).T
    diag = xp.diagonal(chol)
    ratio = xp.diagonal(chol, 1) / diag[:-1]
    center = ratio - xp.concatenate([xp.zeros(1, dtype=ratio.dtype), ratio[:-1]])
    off = (diag[1:] / diag[:-1])[: n - 1]
    jacobi = xp.diag(center) + xp.diag(off, 1) + xp.diag(off, -1)
    nodes, vectors = xp.linalg.eigh(jacobi)
    return nodes, moments[0] * vectors[0] ** 2


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


def _gauss_nodes(
    x_rot: Array,
    y_rot: Array,
    cos_i: Array,
    sin_i: Array,
    rmax_AU: Array,
    scale_height_fn: ScaleHeightFn,
    vertical_exponent: Array | float,
    n: int,
    half_range: bool,
) -> tuple[Array, Array]:
    """Depths and path weights of the Gauss rule for the vertical profile."""
    l_cross, r_cross, safe_cos = _midplane_crossing(x_rot, y_rot, cos_i, sin_i, rmax_AU)
    step = scale_height_fn(r_cross) / jnp.abs(safe_cos)
    s, w = gauss_rule(vertical_exponent, n, half_range=half_range)
    s = jnp.asarray(s, dtype=step.dtype)
    w = jnp.asarray(w, dtype=step.dtype)
    if half_range:
        s = jnp.concatenate([-s[::-1], s])
        w = jnp.concatenate([w[::-1], w])
    # The rule integrates f(s) exp(-|s|^gamma); the kernel's integrand
    # already holds the vertical profile, so divide the weight back out.
    s_abs = jnp.abs(s)
    nonzero = s_abs > 0.0
    w = w * jnp.exp(jnp.where(nonzero, s_abs, 1.0) ** vertical_exponent * nonzero)
    return l_cross + s[:, None, None] * step, w[:, None, None] * step


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
    return low + u[:, None, None] * width, w[:, None, None] * width


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
    quadrature: str = "legendre",
    scale_height_fn: ScaleHeightFn | None = None,
    vertical_exponent: Array | float = 2.0,
    zmax_AU: Array | None = None,
) -> Array:
    """Integrate scattered light along LOS around the disk midplane.

    Private integration chart: (east, north, toward), with depth ``l_grid``
    positive toward the observer (the negative of public sky z). Sky ``x``
    is east (from ``System.positions``), and sky ``y`` is north.
    The line of nodes is (sin(PA), cos(PA)) in (east, north), with
    astronomical position angle PA north through east. The near half for
    inclinations below 90 degrees is (-cos(PA), sin(PA)); above 90 it
    reverses. Inclination zero is face-on. Path lengths are always positive.

    Write ``y_d = y cos(i) + l sin(i)`` and
    ``z_d = -y sin(i) + l cos(i)``. The midplane crossing is at depth
    ``l0 = y tan(i)`` and radius ``r0``, and one scale height ``h`` of the
    layer spans ``h / |cos(i)|`` in depth. Radiative transfer follows
    Augereau et al. (1999), A&A 348, 557.

    **Rules** (``quadrature``, static):

    ``"legendre"`` (default)
        ``n_slices_los`` Gauss-Legendre nodes on the window
        ``|z_d| <= H h(r)`` with ``H = 16^(1 / gamma)`` scale heights (the
        vertical profile is below ``exp(-16)`` outside it). With ``h``
        bounded by its tangent at ``r0`` the window is
        ``l0 +- H h(r0) / (|cos(i)| - H h'(r0) |sin(i)|)``, intersected
        with the chord of the ``rmax_AU`` cylinder. The bound is exact for
        a concave ``h`` (flaring index at most 1); a convex ``h`` can lose
        part of the outer layer. When the denominator closes the ray never
        leaves the flared layer and the window is the whole chord, which is
        finite at edge-on. There is no blend and no inclination threshold:
        the window and the image are smooth through 90 degrees. Convergence
        is spectral where the integrand is smooth. The node count is set by
        the near-edge-on case, a chord holding two narrow ring crossings.
    ``"gauss"``
        ``n_slices_los`` nodes of the Gauss rule for ``exp(-|s|^gamma)``
        (:func:`gauss_rule`) at ``l0 + s_j h(r0) / |cos(i)|``. The rule
        absorbs the vertical profile and treats everything else as a slowly
        varying factor, which holds while
        :func:`layer_expansion_parameter` stays below about
        :data:`GAUSS_EPS_MAX`. Beyond that the error grows without bound
        and no node count repairs it; nothing is raised, because a sampler
        may wander there, so check the prior instead. Best for a smooth
        profile at the midplane (``gamma = 2``).
    ``"gauss_half"``
        Two half-range Gauss rules for ``exp(-s^gamma)`` on ``s > 0``, one
        per side of the midplane, ``n_slices_los`` nodes per side. For
        ``gamma`` other than 2 the flared profile has a cusp at the
        midplane; the symmetric rule straddles it and converges slowly,
        while the half-range pair puts it on a boundary. Same validity
        range as ``"gauss"``.
    ``"trapezoid"``
        The trapezoid rule on log-spaced nodes in a window of half-width
        ``zmax_AU / |cos(i)|`` about ``l0``, blended into an
        outer-radius-bounded window near edge-on (below). Second order:
        the error falls as ``1 / n_slices_los^2`` and is about 2e-2 of the
        peak at 31 nodes and 2e-3 at 101.

    The Gauss rules are built from moments in floating point, which limits
    them to 12 nodes (``"gauss"``) and 8 per side (``"gauss_half"``), or 8
    and 4 for a traced ``vertical_exponent`` with 64-bit floats disabled.

    **Trapezoid window.** The vertical slab ``|z_d| <= Z``
    gives center ``y tan(i)`` and half-width ``Z / |cos(i)|``. The outer
    cylinder implies ``|y_d| <= R`` and thus center ``-y cot(i)`` and
    half-width ``R / |sin(i)|``. The latter is finite at edge-on, where
    it becomes ``[-R, R]``. Both windows contain the intersection of the
    slab and cylinder. A convex blend of their endpoints therefore also
    contains that intersection. We retain the vertical window exactly for
    ``|cos(i)| >= 0.01``, use the radial window for ``|cos(i)| <= 0.009``,
    and blend with a quintic smoothstep between them (C2 endpoints).
    Neither active denominator approaches zero. The radial window includes
    the vertical tails; choose Z to make the omitted slab tails negligible.
    This change of integration coordinates preserves the static node count.

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
        east_AU: East offsets, shape (ny, nx), in AU.
        north_AU: North offsets, shape (ny, nx), in AU.
        n_slices_los: Number of LOS nodes (static); per side of the
            midplane for ``"gauss_half"``.
        quadrature: One of :data:`QUADRATURES` (static).
        scale_height_fn: ``r_AU -> h_AU``, the scale height in
            ``exp(-(|z| / h)^gamma)``, differentiable in ``r``. Required by
            every rule except ``"trapezoid"``.
        vertical_exponent: ``gamma`` of the vertical profile. A Python
            number is a compile-time constant; an array stays
            differentiable. Unused by ``"trapezoid"``.
        zmax_AU: ``"trapezoid"`` only: LOS half-extent in disk-frame z
            [AU]. Should comfortably exceed the disk's largest scale height
            (for flared disks, evaluate the scale-height formula at
            ``rmax_AU``).

    Returns:
        LOS-integrated scattered-light map, shape ``(ny, nx)``.
        Integral of rho * phase / distance_AU**2 over depth_AU.
    """
    check_quadrature(quadrature, n_slices_los)
    if quadrature == "trapezoid":
        if zmax_AU is None:
            raise ValueError("quadrature='trapezoid' needs zmax_AU.")
    elif scale_height_fn is None:
        raise ValueError(f"quadrature={quadrature!r} needs scale_height_fn.")

    incl = incl_deg * deg2rad
    cos_i, sin_i = jnp.cos(incl), jnp.sin(incl)
    pa = position_angle_deg * deg2rad
    x_rot = jnp.sin(pa) * east_AU + jnp.cos(pa) * north_AU
    y_rot = -jnp.cos(pa) * east_AU + jnp.sin(pa) * north_AU

    if quadrature == "trapezoid":
        abs_cos = jnp.abs(cos_i)
        blend = jnp.clip((abs_cos - 0.009) / 0.001, 0.0, 1.0)
        blend = blend**3 * (10.0 + blend * (-15.0 + 6.0 * blend))
        safe_cos = jnp.where(abs_cos > 0.009, cos_i, 1.0)
        safe_sin = jnp.where(abs_cos < 0.01, sin_i, 1.0)
        vertical_mid = y_rot * sin_i / safe_cos
        vertical_half = zmax_AU / jnp.abs(safe_cos)
        radial_mid = -y_rot * cos_i / safe_sin
        radial_half = rmax_AU / jnp.abs(safe_sin)
        l_mid = blend * vertical_mid + (1.0 - blend) * radial_mid
        l_half = blend * vertical_half + (1.0 - blend) * radial_half

        # LOS nodes in [-1, 1], log-spaced and symmetric about the midplane
        # crossing (dense at t = 0). For odd n_slices_los this yields exactly
        # n_slices_los nodes; for even, n_slices_los - 1 (the count is
        # internal -- the LOS is summed away, the output shape is unchanged).
        half = (n_slices_los + 1) // 2
        u = jnp.linspace(0.0, 1.0, half)
        tmp = (jnp.exp(u * jnp.log(_LOS_LWIDTH + 1.0)) - 1.0) / _LOS_LWIDTH
        t = jnp.concatenate([-tmp[:0:-1], tmp])
        l_grid = l_mid[None, :, :] + t[:, None, None] * l_half
    elif quadrature == "legendre":
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
    else:
        l_grid, weights = _gauss_nodes(
            x_rot,
            y_rot,
            cos_i,
            sin_i,
            rmax_AU,
            scale_height_fn,
            vertical_exponent,
            n_slices_los,
            quadrature == "gauss_half",
        )

    y_d = y_rot[None, :, :] * cos_i + l_grid * sin_i
    z_d = -y_rot[None, :, :] * sin_i + l_grid * cos_i

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
    if quadrature == "trapezoid":
        # Trapezoid with non-uniform node spacing (the nodes are log-spaced).
        dl = (t[1:] - t[:-1]) * l_half
        return 0.5 * (dl[:, None, None] * (integrand[:-1] + integrand[1:])).sum(axis=0)
    return (weights * integrand).sum(axis=0)


def check_quadrature(quadrature: str, n_slices_los: int) -> None:
    """Raise ``ValueError`` for an unknown rule or an unusable node count."""
    if quadrature not in QUADRATURES:
        raise ValueError(
            f"quadrature must be one of {QUADRATURES}, got {quadrature!r}."
        )
    if n_slices_los < 1:
        raise ValueError(f"n_slices_los must be positive, got {n_slices_los}.")
    cap = _GAUSS_MAX_NODES.get(quadrature)
    if cap is not None and n_slices_los > cap:
        raise ValueError(
            f"quadrature={quadrature!r} supports at most {cap} nodes, got "
            f"n_slices_los={n_slices_los}: its moment matrix loses about one "
            "digit per node. The error of this rule is set by the inclination, "
            "not the node count; use quadrature='legendre' for more accuracy."
        )
