"""Photometric planets: radii and spectral physics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._flux_scale import check_flux_scale
from .._repr import fmt_scalar_or_array, indent
from ..physical_model import AbstractPhysicalModel


class Planet(eqx.Module):
    """Planet radii and their physical model, with no dynamical state.

    Attributes:
        Rp_Rearth: Planet radius [Earth radii], shape ``B + (K,)``.
        physical_model: Spectral physics (reflectivity / emission).
        flux_scale: Dimensionless factor per planet on its contrast and flux,
            shape ``B + (K,)``, default ones. It sits beside ``Rp_Rearth``
            because it means the same thing when the physical model is
            swapped; ``Scene`` applies it once, where it forwards the radius,
            and a physical model never sees it. It is linear, and exactly
            degenerate with the radius squared and the albedo. It is checked
            finite, not positive.
    """

    Rp_Rearth: Array = eqx.field(converter=jnp.asarray)
    physical_model: AbstractPhysicalModel
    flux_scale: Array = eqx.field(converter=jnp.asarray)

    def __init__(self, Rp_Rearth, physical_model, flux_scale=1.0):
        """Store radii and model; a scalar ``flux_scale`` broadcasts to the radii.

        Args:
            Rp_Rearth: Planet radius [Earth radii], shape ``B + (K,)``.
            physical_model: Spectral physics (reflectivity / emission).
            flux_scale: Dimensionless factor per planet on its contrast and
                flux; a scalar broadcasts to the shape of ``Rp_Rearth``.
        """
        self.Rp_Rearth = jnp.asarray(Rp_Rearth)
        self.physical_model = physical_model
        self.flux_scale = jnp.broadcast_to(
            jnp.asarray(flux_scale), jnp.shape(self.Rp_Rearth)
        )

    def __check_init__(self):
        """Require a finite ``flux_scale``."""
        check_flux_scale(self.flux_scale)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the planetary spectrum [nm].

        Returns:
            Breakpoint wavelengths [nm] from the physical model, on the host.
        """
        return self.physical_model.spectral_breaks_nm()

    @property
    def n_planets(self) -> int:
        """Number of planets ``K`` carried by this photometric module."""
        return int(self.Rp_Rearth.shape[-1])

    def __repr__(self) -> str:
        """Nested summary of radii and the physical model.

        Returns:
            A multi-line text summary.
        """
        rp = fmt_scalar_or_array(self.Rp_Rearth)
        scale = ""
        if not isinstance(self.flux_scale, jax.core.Tracer) and bool(
            jnp.any(self.flux_scale != 1.0)
        ):
            scale = f", flux_scale={fmt_scalar_or_array(self.flux_scale)}"
        return (
            f"Planet(K={self.n_planets}, Rp={rp} R_earth{scale})\n"
            f"{indent('physical_model: ' + repr(self.physical_model))}"
        )
