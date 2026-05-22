"""Temperature-pressure profile components."""

from __future__ import annotations

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
