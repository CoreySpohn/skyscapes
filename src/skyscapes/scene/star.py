"""Star models for the scene hierarchy.

``AbstractStar`` declares a ``Ms_kg`` / ``dist_pc`` pair and the
``spec_flux_density`` hook. ``FlatStar`` is a flat-spectrum stand-in
useful for ETC runs. ``Star`` wraps an ``interpax.Interpolator2D``
over (wavelength, time) built from Jansky flux data, matching the legacy
``skyscapes._legacy.Star`` semantics.

Note: ``from __future__ import annotations`` is deliberately NOT used
here -- it stringifies annotations, which breaks Equinox's metaclass
handling of ``AbstractVar`` type parameters.
"""

from abc import abstractmethod
from typing import Literal

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from jaxtyping import Array

from .spots import StarSpots


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
        """Return spectral flux density in ph/s/m^2/nm."""


class FlatStar(AbstractStar):
    """Flat-spectrum star -- constant flux independent of wavelength or time."""

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

    def __repr__(self) -> str:
        """Compact one-line summary of mass, distance, and flux."""
        m_solar = self.Ms_kg / 1.988409870698051e30
        return (
            f"FlatStar(M={m_solar:.3f} Msun, dist={self.dist_pc:.3g} pc, "
            f"flux={self.flux_phot_per_nm_m2:.3g} ph/s/m^2/nm)"
        )


class Star(AbstractStar):
    """Tabulated stellar flux with optional imported surface metadata.

    A single epoch represents a static spectrum. Spot records are retained
    for later surface modeling; they do not alter the tabulated flux or
    automatically add spatially resolved stellar emission.
    """

    Ms_kg: float
    dist_pc: float
    ra_deg: float
    dec_deg: float
    diameter_arcsec: float
    luminosity_lsun: float
    spots: StarSpots | None
    metadata: dict = eqx.field(static=True)

    _wavelengths_nm: Array
    _times_jd: Array
    _flux_density_phot: Array
    _flux_interp: interpax.Interpolator1D | interpax.Interpolator2D

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
        diameter_arcsec: float = 0.0,
        luminosity_lsun: float = 1.0,
        spots: StarSpots | None = None,
        metadata: dict | None = None,
        interpolation_method: Literal["linear", "cubic"] = "cubic",
    ):
        """Build the flux interpolant; linear interpolation avoids overshoot."""
        if interpolation_method not in ("linear", "cubic"):
            raise ValueError("interpolation_method must be linear or cubic.")
        self.Ms_kg = Ms_kg
        self.dist_pc = dist_pc
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.diameter_arcsec = diameter_arcsec
        self.luminosity_lsun = luminosity_lsun
        self.spots = spots
        self.metadata = {} if metadata is None else dict(metadata)
        self._wavelengths_nm = wavelengths_nm
        self._times_jd = times_jd

        self._flux_density_phot = jax.vmap(
            jy_to_photons_per_nm_per_m2, in_axes=(1, None), out_axes=1
        )(flux_density_jy, wavelengths_nm)

        if times_jd.size == 1:
            self._flux_interp = interpax.Interpolator1D(
                wavelengths_nm,
                self._flux_density_phot[:, 0],
                method=interpolation_method,
            )
        else:
            self._flux_interp = interpax.Interpolator2D(
                wavelengths_nm,
                times_jd,
                self._flux_density_phot,
                method=interpolation_method,
            )

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Scalar or array spectral flux density [ph/s/m^2/nm]."""
        if self._times_jd.size == 1:
            # A single-epoch source is a static spectrum, not a degenerate spline.
            wavelength_nm, _ = jnp.broadcast_arrays(wavelength_nm, time_jd)
            return self._flux_interp(wavelength_nm)
        return self._flux_interp(wavelength_nm, time_jd)

    def __repr__(self) -> str:
        """One-line summary of metadata + wavelength/time grid extent."""
        m_solar = self.Ms_kg / 1.988409870698051e30
        wl_min = float(jnp.min(self._wavelengths_nm))
        wl_max = float(jnp.max(self._wavelengths_nm))
        n_t = int(self._times_jd.shape[0])
        return (
            f"Star(M={m_solar:.3f} Msun, "
            f"dist={self.dist_pc:.3g} pc, "
            f"L={self.luminosity_lsun:.3g} Lsun, "
            f"diam={self.diameter_arcsec:.3g} arcsec, "
            f"RA/Dec=({self.ra_deg:.4f}, {self.dec_deg:.4f}) deg, "
            f"wl={wl_min:.0f}-{wl_max:.0f} nm, n_times={n_t})"
        )
