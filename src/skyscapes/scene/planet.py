"""Photometric planets: radii and spectral physics."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._repr import fmt_scalar_or_array, indent
from ..physical_model import AbstractPhysicalModel


class Planet(eqx.Module):
    """Planet radii and their physical model, with no dynamical state.

    Attributes:
        Rp_Rearth: Planet radius [Earth radii], shape ``B + (K,)``.
        physical_model: Spectral physics (reflectivity / emission).
    """

    Rp_Rearth: Array = eqx.field(converter=jnp.asarray)
    physical_model: AbstractPhysicalModel

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the planetary spectrum [nm]."""
        return self.physical_model.spectral_breaks_nm()

    @property
    def n_planets(self) -> int:
        """Number of planets ``K`` carried by this photometric module."""
        return int(self.Rp_Rearth.shape[-1])

    def __repr__(self) -> str:
        """Nested summary of radii and the physical model."""
        rp = fmt_scalar_or_array(self.Rp_Rearth)
        return (
            f"Planet(K={self.n_planets}, Rp={rp} R_earth)\n"
            f"{indent('physical_model: ' + repr(self.physical_model))}"
        )
