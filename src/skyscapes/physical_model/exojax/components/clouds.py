"""Cloud opacity components."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractClouds, Contribution

# Typical liquid-water terrestrial cloud values: pure scattering
# (ssa = 1, no internal absorption at visible/NIR), strongly forward-
# scattering (g ~ 0.85).
DEFAULT_CLOUD_SSA = 1.0
DEFAULT_CLOUD_ASYMMETRY_G = 0.85
# Cloud vertical distribution width in log10(P) [dex]. 0.3 spreads the
# cloud over ~half a decade in pressure -- a reasonable single-deck width
# at the default 100-layer resolution.
DEFAULT_CLOUD_LOG_SIGMA = 0.3


class GrayCloud(AbstractClouds):
    """Single-layer gray-scattering cloud.

    The total cloud scattering optical depth is distributed across
    layers via a Gaussian in log-pressure (softmax-normalised so the
    weights are well-defined even when the cloud pressure is far
    outside the layer grid -- useful for ablation runs with
    ``log_opt_depth = -inf``).

    Wavelength-grey: the cloud's cross-section is constant across the
    spectrum. A Mie cloud with composition-dependent scattering would
    be a separate component implementing the same contract.

    Attributes (PyTree leaves, fittable):
        log_pressure_bar: Log10 cloud-deck pressure [bar], shape ``(K,)``.
        log_opt_depth: Log10 of the vertically-integrated cloud
            scattering optical depth, shape ``(K,)``.

    Static attributes (configuration):
        ssa: Single-scattering albedo (default 1.0, pure scattering).
        g: Asymmetry parameter (default 0.85, forward-peaked).
        log_sigma: Vertical-distribution width in log10(P) [dex].
    """

    log_pressure_bar: Array
    log_opt_depth: Array
    ssa: float = eqx.field(static=True, default=DEFAULT_CLOUD_SSA)
    g: float = eqx.field(static=True, default=DEFAULT_CLOUD_ASYMMETRY_G)
    log_sigma: float = eqx.field(static=True, default=DEFAULT_CLOUD_LOG_SIGMA)

    def compute(
        self,
        log_pressure_bar_scalar: Array,
        log_opt_depth_scalar: Array,
        pressure: Array,
        n_nu: int,
    ) -> Contribution:
        """Gray cloud contribution at a single pressure level."""
        n_layers = pressure.shape[0]
        log_p_layers = jnp.log10(pressure)
        log_w = -0.5 * ((log_p_layers - log_pressure_bar_scalar) / self.log_sigma) ** 2
        # Softmax stabilizes the normalisation when the cloud pressure
        # is far from any layer (would otherwise give 0/0 underflow).
        weights = jax.nn.softmax(log_w)
        tau_total = 10.0**log_opt_depth_scalar
        dtau_layer = tau_total * weights  # (n_layers,)
        dtau = jnp.broadcast_to(dtau_layer[:, None], (n_layers, n_nu))
        dtau_scatter = self.ssa * dtau
        g_weighted_num = self.g * dtau_scatter
        return Contribution(
            dtau_total=dtau, dtau_scatter=dtau_scatter, g_weighted_num=g_weighted_num
        )


class NoCloud(AbstractClouds):
    """Disable cloud opacity: zero contribution.

    Equivalent to a ``GrayCloud`` with ``log_opt_depth = -inf`` but
    structurally explicit, so that ``isinstance(atm.clouds, NoCloud)``
    documents intent in code that builds cloud-free atmospheres.
    """

    def compute(
        self,
        log_pressure_bar_scalar: Array,
        log_opt_depth_scalar: Array,
        pressure: Array,
        n_nu: int,
    ) -> Contribution:
        """Return zero-everywhere cloud contribution.

        ``log_pressure_bar_scalar`` and ``log_opt_depth_scalar`` are
        accepted for parity with :class:`GrayCloud` but unused.
        """
        zeros = jnp.zeros((pressure.shape[0], n_nu))
        return Contribution(dtau_total=zeros, dtau_scatter=zeros, g_weighted_num=zeros)
