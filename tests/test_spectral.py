"""Closed-form and transformation checks for shared spectral integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyscapes.spectral import bin_mean


def test_piecewise_linear_bins():
    """Order one is exact to 2e-14 relative/absolute float64 rounding error.

    Integrate f(x) = 2 + x + 3*max(x-2, 0) - 2*max(x-4, 0).
    Include no interior knot, edges on knots, unequal widths and point bins.
    """
    knots = np.array([0.0, 2.0, 4.0, 7.0])
    centers = np.array([1.0, 3.0, 3.5, 5.0, 2.0])
    widths = np.array([1.0, 2.0, 5.0, 0.0, 0.0])

    def fn(x):
        f = 2 + x + 3 * jnp.maximum(x - 2, 0) - 2 * jnp.maximum(x - 4, 0)
        return jnp.array([[f, 2 * f], [-f, f]])

    def primitive(x):
        return (
            2 * x
            + x**2 / 2
            + 1.5 * np.maximum(x - 2, 0) ** 2
            - np.maximum(x - 4, 0) ** 2
        )

    expected = primitive(centers + widths / 2) - primitive(centers - widths / 2)
    expected /= np.where(widths == 0, 1, widths)
    expected[3:] = [14, 4]
    actual = jax.jit(lambda: bin_mean(fn, centers, widths, n_nodes=1, knots_nm=knots))()
    np.testing.assert_allclose(
        actual,
        expected[:, None, None] * np.array([[1, 2], [-1, 1]]),
        rtol=2e-14,
        atol=2e-14,
    )


def test_weighted_mean_jit_vmap_and_parameter_gradient():
    """Quadratic integrands are exact at order two (rtol/atol 2e-14).

    Weight x and f=a*x give a*(c + width**2/(12*c)). Check a closed-over
    parameter gradient against a central difference with step 1e-5, rtol
    1e-9 and atol 1e-10, allowing cancellation of float64 function values.
    """
    centers = jnp.array([2.0, 3.0])
    widths = jnp.array([1.0, 2.0])
    knots = jnp.array([0.0, 2.0, 4.0])

    def evaluate(a, c, w):
        return bin_mean(
            lambda x: a * x,
            c,
            w,
            n_nodes=2,
            knots_nm=knots,
            max_knots=2,
            weight_fn=lambda x: x,
        )

    eager = evaluate(1.7, centers, widths)
    compiled = jax.jit(evaluate)
    actual = compiled(1.7, centers, widths)
    expected = 1.7 * (centers + widths**2 / (12 * centers))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(actual, eager, rtol=2e-14, atol=2e-14)
    batched = jax.jit(jax.vmap(evaluate, in_axes=(0, None, None)))(
        jnp.array([1.7, 3.4]), centers, widths
    )
    np.testing.assert_allclose(
        batched, jnp.stack([actual, 2 * actual]), rtol=2e-14, atol=2e-14
    )
    derivative = jax.jit(jax.grad(lambda a: evaluate(a, centers, widths).sum()))(1.7)
    fd = (
        compiled(1.7 + 1e-5, centers, widths).sum()
        - compiled(1.7 - 1e-5, centers, widths).sum()
    ) / 2e-5
    np.testing.assert_allclose(derivative, fd, rtol=1e-9, atol=1e-10)


def test_cap_errors():
    """Cap errors are exact: two interior knots require three sub-intervals."""
    centers, widths, knots = np.array([3.0]), np.array([4.0]), np.array([2.0, 4.0])
    with pytest.raises(ValueError, match=r"max_knots.*too small"):
        bin_mean(lambda x: x, centers, widths, knots_nm=knots, max_knots=2)
    with pytest.raises(ValueError, match=r"Traced.*max_knots"):
        jax.jit(lambda c: bin_mean(lambda x: x, c, widths, knots_nm=knots))(centers)
    with pytest.raises(ValueError, match=r"Traced.*max_knots"):
        jax.jit(lambda k: bin_mean(lambda x: x, centers, widths, knots_nm=k))(knots)
