"""Band integration of scalar-wavelength functions with optional spectral knots."""

from collections.abc import Callable
from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from numpy.polynomial.legendre import leggauss


def _required_intervals(centers_nm, widths_nm, knots_nm):
    """Return the concrete sub-interval cap, or None for traced geometry."""
    if any(
        isinstance(x, jax.core.Tracer)
        for x in jax.tree.leaves((centers_nm, widths_nm, knots_nm))
    ):
        return None
    centers, widths, knots = map(np.asarray, (centers_nm, widths_nm, knots_nm))
    low, high = centers - widths / 2, centers + widths / 2
    counts = np.searchsorted(knots, high, side="left") - np.searchsorted(
        knots, low, side="right"
    )
    return int(np.max(counts, initial=0)) + 1


def _model_max_knots(centers_nm, widths_nm, knots_nm):
    """Infer a model's cap, conservatively retaining legacy traced-bin support."""
    if knots_nm is None:
        return None
    required = _required_intervals(centers_nm, widths_nm, knots_nm)
    return len(knots_nm) + 1 if required is None else required


def bin_mean(
    fn: Callable,
    centers_nm: Array,
    widths_nm: Array,
    *,
    n_nodes: int = 4,
    knots_nm: Array | None = None,
    max_knots: int | None = None,
    weight_fn: Callable | None = None,
) -> Array:
    """Mean of ``fn(wavelength_nm)`` over each bin, shape ``(W, ...)``.

    Args:
        fn: Scalar wavelength [nm] to an array of any fixed shape ``(...)``.
            Evaluation uses ``vmap`` and preserves parameter gradients.
        centers_nm: Bin centers [nm], shape ``(W,)``.
        widths_nm: Nonnegative full bin widths [nm], shape ``(W,)``.
        n_nodes: Static positive Gauss-Legendre order per sub-interval.
        knots_nm: Optional sorted 1D wavelength table [nm]. Split at strictly
            interior knots. A piecewise-linear function is exact for any
            positive order; smooth factors converge with increasing order.
        max_knots: Static positive cap on the number of SUB-INTERVALS per bin
            (interior knots plus one). Inferred when all geometry is concrete;
            required if any geometry is traced. Concrete undersized caps raise
            ValueError. Under tracing the caller must ensure sufficient capacity;
            otherwise some kinks are unresolved. Public star/system methods infer
            the cap from concrete inputs where possible. Selection uses binary
            search and bounded gathers, never a bins-by-table matrix.
        weight_fn: Optional scalar wavelength to scalar weight. Return
            integral(weight * fn) / integral(weight); the denominator must be
            nonzero for a nonzero-width bin.

    Returns:
        Bin means, shape ``(W, ...)``. Zero represented width returns ``fn``
        at the center, including widths smaller than floating-point resolution.
        Without knots, use one rule per bin. With knots, padded intervals have
        zero weight. Work is O(W * (log(len(knots_nm)) + max_knots * n_nodes)),
        apart from the cost of the supplied functions.

    Notes:
        Gauss-Legendre quadrature integrates polynomials through degree 2*n-1
        exactly; see Davis and Rabinowitz (1984), Methods of Numerical
        Integration. Normalize by the represented edge span to retain narrow
        bin accuracy. JIT, vmap and grad are supported with a static cap; knot
        membership changes are discrete, so edge gradients are piecewise smooth.
    """
    n_nodes = index(n_nodes)
    if n_nodes < 1:
        raise ValueError("n_nodes must be a positive integer.")
    if knots_nm is not None:
        required = _required_intervals(centers_nm, widths_nm, knots_nm)
        if max_knots is None:
            if required is None:
                raise ValueError("Traced bin geometry requires a static max_knots.")
            max_knots = required
        max_knots = index(max_knots)
        if max_knots < 1:
            raise ValueError("max_knots must be a positive sub-interval cap.")
        if required is not None and max_knots < required:
            raise ValueError(f"max_knots={max_knots} is too small; need {required}.")

    centers, widths = jnp.asarray(centers_nm), jnp.asarray(widths_nm)
    widths = eqx.error_if(widths, jnp.any(widths < 0), "Bin widths must be >= 0.")
    low, high = centers - widths / 2, centers + widths / 2
    if knots_nm is None or len(knots_nm) == 0:
        edges = jnp.stack((low, high), axis=1)
    else:
        knots = jnp.asarray(knots_nm)
        start = jnp.searchsorted(knots, low, side="right")
        offsets = start[:, None] + jnp.arange(max_knots - 1)
        interior = knots[jnp.minimum(offsets, knots.size - 1)]
        interior = jnp.where(offsets < knots.size, interior, high[:, None])
        interior = jnp.clip(interior, low[:, None], high[:, None])
        edges = jnp.concatenate((low[:, None], interior, high[:, None]), axis=1)

    lo, hi = edges[:, :-1], edges[:, 1:]
    span = high - low
    point_bin = span == 0
    lengths = (hi - lo) / jnp.where(point_bin, 1.0, span)[:, None]
    lengths = jnp.where(point_bin[:, None], 0.0, lengths)
    lengths = lengths.at[:, 0].set(jnp.where(point_bin, 1.0, lengths[:, 0]))
    nodes, weights = leggauss(n_nodes)
    wavelengths = lo[..., None] + (hi - lo)[..., None] * ((jnp.asarray(nodes) + 1) / 2)
    # Evaluate inactive intervals at the center, avoiding 0 * NaN at endpoints.
    wavelengths = jnp.where(
        ((hi == lo) | point_bin[:, None])[..., None],
        centers[:, None, None],
        wavelengths,
    )
    wavelengths = wavelengths.reshape((centers.size, -1))
    weights = (lengths[..., None] * jnp.asarray(weights) / 2).reshape(wavelengths.shape)
    values = jax.vmap(jax.vmap(fn))(wavelengths)
    if weight_fn is not None:
        throughput = jax.vmap(jax.vmap(weight_fn))(wavelengths)
        weights = weights * jnp.where(point_bin[:, None], 1.0, throughput)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    return jnp.einsum("wn,wn...->w...", weights, values)
