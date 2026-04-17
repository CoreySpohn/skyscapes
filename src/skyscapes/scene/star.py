"""Star models for the scene hierarchy.

``AbstractStar`` declares a ``Ms_kg`` / ``dist_pc`` pair and the
``spec_flux_density`` hook. ``SimpleStar`` is a flat-spectrum stand-in
useful for ETC runs. ``SpectrumStar`` wraps an ``interpax.Interpolator2D``
over (wavelength, time) built from Jansky flux data, matching the legacy
``skyscapes._legacy.Star`` semantics.

Note: ``from __future__ import annotations`` is deliberately NOT used
here — it stringifies annotations, which breaks Equinox's metaclass
handling of ``AbstractVar`` type parameters.
"""

from abc import abstractmethod

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from jaxtyping import Array

_Jy = 1e-26  # W m^-2 Hz^-1
_h = 6.62607015e-34  # J s


def _jy_to_photons_per_nm_per_m2(flux_jy: Array, wavelength_nm: Array) -> Array:
    """Convert Jy → ph/s/nm/m²."""
    return flux_jy * _Jy / (wavelength_nm * _h)


class AbstractStar(eqx.Module):
    """Abstract stellar source.

    Attributes:
        Ms_kg: Stellar mass in kilograms.
        dist_pc: Distance to the star in parsecs.
    """

    Ms_kg: AbstractVar[float]
    dist_pc: AbstractVar[float]

    @abstractmethod
    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Return spectral flux density in ph/s/m²/nm."""


class SimpleStar(AbstractStar):
    """Flat-spectrum star — constant flux independent of wavelength or time."""

    Ms_kg: float
    dist_pc: float
    flux_phot_per_nm_m2: float

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Constant flux, broadcast to wavelength_nm's shape.

        ``time_jd`` is part of the AbstractStar interface but ignored here.
        """
        wl = jnp.asarray(wavelength_nm)
        return jnp.full_like(wl, self.flux_phot_per_nm_m2, dtype=wl.dtype)


class SpectrumStar(AbstractStar):
    """Time- and wavelength-dependent star backed by an interpax 2D spline."""

    Ms_kg: float
    dist_pc: float
    ra_deg: float
    dec_deg: float
    midplane_pa_deg: float
    midplane_i_deg: float
    diameter_arcsec: float
    luminosity_lsun: float

    _wavelengths_nm: Array
    _times_jd: Array
    _flux_density_phot: Array
    _flux_interp: interpax.Interpolator2D

    def __init__(
        self,
        *,
        Ms_kg: float,
        dist_pc: float,
        wavelengths_nm: Array,
        times_jd: Array,
        flux_density_jy: Array,
        ra_deg: float = 0.0,
        dec_deg: float = 0.0,
        midplane_pa_deg: float = 0.0,
        midplane_i_deg: float = 0.0,
        diameter_arcsec: float = 0.0,
        luminosity_lsun: float = 1.0,
    ):
        """Store stellar metadata and pre-build the flux-density interpolant."""
        self.Ms_kg = Ms_kg
        self.dist_pc = dist_pc
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.midplane_pa_deg = midplane_pa_deg
        self.midplane_i_deg = midplane_i_deg
        self.diameter_arcsec = diameter_arcsec
        self.luminosity_lsun = luminosity_lsun
        self._wavelengths_nm = wavelengths_nm
        self._times_jd = times_jd

        self._flux_density_phot = jax.vmap(
            _jy_to_photons_per_nm_per_m2, in_axes=(1, None), out_axes=1
        )(flux_density_jy, wavelengths_nm)

        self._flux_interp = interpax.Interpolator2D(
            wavelengths_nm, times_jd, self._flux_density_phot, method="cubic"
        )

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Scalar or array spectral flux density [ph/s/m²/nm]."""
        return self._flux_interp(wavelength_nm, time_jd)
