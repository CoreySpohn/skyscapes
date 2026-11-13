"""Shared Planck photon spectrum with observer dilution."""

from math import pi, sqrt

import jax.numpy as jnp
from hwoutils.constants import c, h, k_B, nm2m, nm2um, pc2m, um2m
from jaxtyping import Array


def _planck_photon_flux(
    wavelength_nm: Array,
    temperature_K: Array,
    radius: Array,
    dist_pc: Array,
    *,
    radius_unit_m: float = 1.0,
) -> Array:
    """Diluted Planck photon flux density [ph/s/m^2/nm].

    Evaluate pi B_lambda(T) (R/d)**2 divided by photon energy, following
    Rybicki and Lightman (1979), Radiative Processes in Astrophysics.
    Radius is in units of ``radius_unit_m`` meters; inputs broadcast.
    Fold radiance, dilution and per-nm conversion into one moderate scale
    before squaring to preserve float32 second derivatives under JIT.
    """
    wavelength_um = jnp.asarray(wavelength_nm) * nm2um
    exponent = (h * c / (k_B * um2m)) / (wavelength_um * temperature_K)
    amplitude = sqrt(2.0 * pi * c * nm2m) * radius_unit_m / (um2m**2 * pc2m)
    scale = (amplitude * radius / dist_pc) ** 2
    return scale / (wavelength_um**4 * jnp.expm1(exponent))
