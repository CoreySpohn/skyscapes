"""Temperature-pressure profile components."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractTPProfile


class PowerLawTPProfile(AbstractTPProfile):
    """Power-law T-P profile: ``T(P) = T_eq * P^alpha``.

    The two parameters are PyTree leaves but they are passed to
    :meth:`compute_Tarr` as scalar args rather than being read from
    ``self`` so the atmosphere can vmap over per-planet axes without
    extra plumbing.

    Attributes:
        T_eq_K: Reference temperature at 1 bar [K], shape ``(K,)``.
        T_alpha: Power-law exponent, shape ``(K,)``.
    """

    T_eq_K: Array
    T_alpha: Array

    def compute_Tarr(
        self, rt_engine, T_eq_K_scalar: Array, T_alpha_scalar: Array
    ) -> Array:
        """``T(P) = T_eq * P^alpha`` on the rt_engine's layer pressures."""
        return rt_engine.powerlaw_temperature(T_eq_K_scalar, T_alpha_scalar)


class PolynomialTPProfile(AbstractTPProfile):
    """Polynomial T-P profile: ``T(P) = sum_i a_i (log10 P[bar])**i``.

    The form used to describe an Earth-twin atmosphere in published
    thermal-emission studies. Coefficients are PyTree leaves; an
    isothermal atmosphere is ``a = (T, 0, ..., 0)``.

    Attributes:
        coefficients: Polynomial coefficients ``a_0 ... a_n`` in kelvin,
            shape ``(K, n + 1)``.
    """

    coefficients: Array

    def temperature(self, pressure_bar: Array, k: int = 0) -> Array:
        """Temperature [K] of planet ``k`` at ``pressure_bar``."""
        logp = jnp.log10(pressure_bar)
        powers = logp[..., None] ** jnp.arange(self.coefficients.shape[-1])
        return powers @ self.coefficients[k]

    def compute_Tarr(
        self, rt_engine, coefficients_scalar: Array, unused: Array | None = None
    ) -> Array:
        """Layer temperatures on the rt_engine's pressure grid.

        ``coefficients_scalar`` is one planet's coefficient vector, shape
        ``(n + 1,)``; ``unused`` exists only to match the abstract signature.
        """
        logp = jnp.log10(rt_engine.pressure)
        powers = logp[:, None] ** jnp.arange(coefficients_scalar.shape[-1])
        return powers @ coefficients_scalar
