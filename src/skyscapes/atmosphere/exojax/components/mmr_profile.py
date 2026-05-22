"""Mass-mixing-ratio profiles for atmospheric species.

Each species in a :class:`MolecularSpecies` tuple carries one of these
profile components, replacing the previous "scalar mmr per planet"
representation. Different molecules need different altitude variations:

  - **Well-mixed gases** (N2, O2, CO2, CH4 in our pressure range): use
    :class:`ConstantMmr` -- same mmr at every layer.
  - **O3** is concentrated in the stratosphere with column ~10x larger
    than a uniformly-mixed model would assume: use
    :class:`StratosphericPeakMmr` (Gaussian in log-pressure).
  - **H2O** has a sharp cold-trap drop at the tropopause (~0.1 bar)
    where temperatures fall below ~200 K: use :class:`TroposphericMmr`
    (sigmoid-smoothed step in log-pressure).

The profile contract is just ``evaluate(pressure) -> (n_layers,) mmr``.
Layer pressures come from the rt_engine; each component returns the
per-layer mass-mixing ratio.
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class AbstractMmrProfile(eqx.Module, strict=True):
    """Per-species mass-mixing-ratio profile evaluated at layer pressures."""

    @abstractmethod
    def evaluate(self, pressure: Array) -> Array:
        """Return shape ``(n_layers,)`` mmr values at the layer pressures."""


class ConstantMmr(AbstractMmrProfile):
    """Well-mixed gas: constant mmr at every layer.

    Use for gases with chemical lifetimes longer than mixing timescales
    in the modelled pressure range (CO2, CH4, O2, N2, etc. in
    troposphere + lower stratosphere).

    Attributes:
        log_mmr: Log10 mass-mixing ratio, shape ``(K,)`` (per planet).
    """

    log_mmr: Array

    def evaluate(self, pressure: Array) -> Array:
        """Return ``10**log_mmr`` broadcast to ``pressure.shape``."""
        return (10.0**self.log_mmr) * jnp.ones_like(pressure)


class StratosphericPeakMmr(AbstractMmrProfile):
    """Gaussian-in-log-pressure peak; canonical for O3.

    ``mmr(P) = 10**log_peak_mmr * exp(-0.5 * ((log10(P) -
    log_peak_pressure_bar) / log_sigma_decades)**2)``.

    For Earth's O3: ``log_peak_pressure_bar = log10(0.01) = -2``
    (10 mbar, ~30 km altitude), ``log_sigma_decades = 0.5``.

    Attributes:
        log_peak_mmr: Log10 of the peak mmr, shape ``(K,)``.
        log_peak_pressure_bar: Log10 pressure [bar] at peak, ``(K,)``.
        log_sigma_decades: Gaussian width in log10-pressure, ``(K,)``.
    """

    log_peak_mmr: Array
    log_peak_pressure_bar: Array
    log_sigma_decades: Array

    def evaluate(self, pressure: Array) -> Array:
        """Gaussian peak in log-pressure."""
        log_p = jnp.log10(pressure)
        shape = jnp.exp(
            -0.5 * ((log_p - self.log_peak_pressure_bar) / self.log_sigma_decades) ** 2
        )
        return (10.0**self.log_peak_mmr) * shape


class TroposphericMmr(AbstractMmrProfile):
    """Constant below a step pressure, drops sharply above; canonical for H2O.

    Uses a sigmoid in log-pressure for smooth transition (differentiable
    for HMC retrievals). For Earth's H2O: tropospheric ~3e-3 below
    0.1 bar, ~3e-6 above.

    Attributes:
        log_mmr_below: Log10 mmr at high pressure (below cold trap), ``(K,)``.
        log_mmr_above: Log10 mmr at low pressure (above cold trap), ``(K,)``.
        log_pressure_step_bar: Log10 transition pressure [bar], ``(K,)``.
        log_transition_width_decades: Width of the sigmoid transition
            in log10-pressure, ``(K,)``. Smaller = sharper step.
    """

    log_mmr_below: Array
    log_mmr_above: Array
    log_pressure_step_bar: Array
    log_transition_width_decades: Array

    def evaluate(self, pressure: Array) -> Array:
        """Sigmoid step in log-pressure."""
        log_p = jnp.log10(pressure)
        # Sigmoid weight: ~1 at high pressure (below cold trap),
        # ~0 at low pressure (above cold trap).
        weight_below = jax.nn.sigmoid(
            (log_p - self.log_pressure_step_bar) / self.log_transition_width_decades
        )
        weight_above = 1.0 - weight_below
        return (10.0**self.log_mmr_below) * weight_below + (
            10.0**self.log_mmr_above
        ) * weight_above
