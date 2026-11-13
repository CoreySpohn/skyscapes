"""Star models for the scene hierarchy.

``AbstractStar`` declares ``Ms_kg``, ``dist_pc``, ``diameter_arcsec``, and the
``spec_flux_density`` hook. ``FlatStar`` is a flat-spectrum stand-in
useful for ETC runs. ``BlackbodyStar`` supplies a diluted Planck spectrum.
``Star`` wraps an ``interpax.Interpolator2D``
over (wavelength, time) built from Jansky flux data, matching the legacy
``skyscapes._legacy.Star`` semantics.

Note: ``from __future__ import annotations`` is deliberately NOT used
here -- it stringifies annotations, which breaks Equinox's metaclass
handling of ``AbstractVar`` type parameters.
"""

from abc import abstractmethod
from math import pi, sqrt
from typing import Literal

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from hwoutils.constants import Msun2kg, c, h, k_B, nm2m, nm2um, pc2m, rad2arcsec, um2m
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from jaxtyping import Array
from numpy.polynomial.legendre import leggauss

from .spots import StarSpots


def _bin_nodes(centers_nm: Array, widths_nm: Array, n_nodes: int):
    """Static Gauss-Legendre setup, followed by traced wavelength mapping."""
    # NumPy generates configuration constants only, never evaluates spectra.
    nodes, weights = leggauss(n_nodes)
    centers = jnp.asarray(centers_nm)
    widths = jnp.asarray(widths_nm)
    widths = eqx.error_if(widths, jnp.any(widths < 0), "Bin widths must be >= 0.")
    return centers[:, None] + widths[:, None] * jnp.asarray(nodes) / 2, jnp.asarray(
        weights
    ) / 2


class AbstractStar(eqx.Module):
    """Abstract stellar source.

    Attributes:
        Ms_kg: Stellar mass in kilograms.
        dist_pc: Distance to the star in parsecs.
        diameter_arcsec: Angular diameter in arcseconds; zero denotes a point star.
    """

    Ms_kg: AbstractVar[float]
    dist_pc: AbstractVar[float]
    diameter_arcsec: AbstractVar[float]

    @abstractmethod
    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Return spectral flux density in ph/s/m^2/nm."""

    def bin_flux_density(
        self, centers_nm: Array, widths_nm: Array, time_jd: Array, *, n_nodes: int = 4
    ) -> Array:
        """Mean photon flux density per bin [ph/s/m^2/nm], shape ``(W,)``.

        Centers and nonnegative full widths have shape ``(W,)``; time is
        scalar. Width zero returns the point value. Analytic spectra use a
        static ``n_nodes``-point Gauss-Legendre rule (default 4). Its error
        is O(width**(2*n_nodes)); see Davis and Rabinowitz (1984), Methods
        of Numerical Integration. For a 5772 K blackbody, centers 400-5000
        nm and widths <= 20 percent of center, the default relative error
        is below 1e-8. Wider bins or Wien tails need a convergence check.
        """
        nodes, weights = _bin_nodes(centers_nm, widths_nm, n_nodes)
        flux = jax.vmap(jax.vmap(lambda w: self.spec_flux_density(w, time_jd)))(nodes)
        mean = flux @ weights
        point = self.spec_flux_density(jnp.asarray(centers_nm), time_jd)
        return jnp.where(jnp.asarray(widths_nm) == 0, point, mean)


class FlatStar(AbstractStar):
    """Flat-spectrum star -- constant flux independent of wavelength or time.

    Attributes:
        Ms_kg: Stellar mass in kilograms.
        dist_pc: Distance to the star in parsecs.
        flux_phot_per_nm_m2: Constant observer flux density in ph/s/m^2/nm.
        diameter_arcsec: Angular diameter in arcseconds, default 0.0 (point star).
    """

    Ms_kg: float
    dist_pc: float
    flux_phot_per_nm_m2: float
    diameter_arcsec: float = 0.0

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
        """Summarize mass, distance, flux, and nonzero angular diameter."""
        m_solar = self.Ms_kg / Msun2kg
        diameter = (
            f", diam={self.diameter_arcsec:.3g} arcsec" if self.diameter_arcsec else ""
        )
        return (
            f"FlatStar(M={m_solar:.3f} Msun, dist={self.dist_pc:.3g} pc, "
            f"flux={self.flux_phot_per_nm_m2:.3g} ph/s/m^2/nm{diameter})"
        )


class BlackbodyStar(AbstractStar):
    """Planck blackbody spectrum diluted by the star's distance.

    Uses the Planck function from Rybicki & Lightman (1979),
    Radiative Processes in Astrophysics, divided by the photon energy.
    The surface flux is pi times the photon spectral radiance.

    Attributes:
        Ms_kg: Stellar mass in kilograms.
        dist_pc: Distance to the star in parsecs.
        teff_K: Effective temperature in kelvin.
        radius_m: Stellar radius in meters.
        diameter_arcsec: Angular diameter derived from radius and distance.
    """

    Ms_kg: float
    dist_pc: float
    teff_K: float
    radius_m: float

    @property
    def diameter_arcsec(self) -> float:
        """Small-angle diameter, 2 R / d, in arcseconds.

        Uses the small-angle relation from Carroll & Ostlie (2007),
        An Introduction to Modern Astrophysics.
        """
        return 2.0 * self.radius_m / (self.dist_pc * pc2m) * rad2arcsec

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_jd: Array,
    ) -> Array:
        """Photon flux density at the observer in ph/s/m^2/nm.

        The result has wavelength_nm's shape. ``time_jd`` is part of
        the AbstractStar interface but ignored here.
        """
        wavelength_um = jnp.asarray(wavelength_nm) * nm2um
        exponent = (h * c / (k_B * um2m)) / (wavelength_um * self.teff_K)
        # Combine radiance and dilution before squaring to preserve float32 HVPs.
        amplitude = sqrt(2.0 * pi * c * nm2m) / (um2m**2 * pc2m)
        scale = (amplitude * self.radius_m / self.dist_pc) ** 2
        return scale / (wavelength_um**4 * jnp.expm1(exponent))

    def __repr__(self) -> str:
        """Compact one-line summary of mass, distance, temperature, and radius."""
        m_solar = self.Ms_kg / Msun2kg
        return (
            f"BlackbodyStar(M={m_solar:.3f} Msun, dist={self.dist_pc:.3g} pc, "
            f"Teff={self.teff_K:.3g} K, radius={self.radius_m:.3g} m)"
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

    _interpolation_method: str = eqx.field(static=True)
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
        self._interpolation_method = interpolation_method
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

    @property
    def wavelengths_nm(self) -> Array:
        """Tabulated wavelength grid in nm, returned without copying."""
        return self._wavelengths_nm

    @property
    def wavelength_range_nm(self) -> tuple[float, float]:
        """Minimum and maximum tabulated wavelengths in nm as Python floats."""
        return (
            float(jnp.min(self._wavelengths_nm)),
            float(jnp.max(self._wavelengths_nm)),
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

    def _bin_weights(self, centers_nm: Array, widths_nm: Array) -> Array:
        """Weights on table knots for exact partial linear-segment integrals.

        Zero-width rows are zero; callers supply their point-value limit.
        Bins beyond the table return NaN instead of silently losing flux.
        """
        centers, widths = jnp.asarray(centers_nm), jnp.asarray(widths_nm)
        widths = eqx.error_if(widths, jnp.any(widths < 0), "Bin widths must be >= 0.")
        low, high = centers - widths / 2, centers + widths / 2
        x0, x1 = self._wavelengths_nm[:-1], self._wavelengths_nm[1:]
        lo = jnp.clip(x0[None, :], low[:, None], high[:, None])
        hi = jnp.clip(x1[None, :], low[:, None], high[:, None])
        fraction = ((lo + hi) / 2 - x0) / (x1 - x0)
        # Divide by the represented span so sub-ULP edge rounding does not
        # spoil the point limit of a very narrow bin.
        span = high - low
        length = (hi - lo) / jnp.where(span == 0, 1.0, span)[:, None]
        left, right = length * (1 - fraction), length * fraction
        weights = jnp.pad(left, ((0, 0), (0, 1))) + jnp.pad(right, ((0, 0), (1, 0)))
        covered = (low >= self._wavelengths_nm[0]) & (high <= self._wavelengths_nm[-1])
        return jnp.where(covered[:, None], weights, jnp.nan)

    def bin_flux_density(
        self, centers_nm: Array, widths_nm: Array, time_jd: Array, *, n_nodes: int = 4
    ) -> Array:
        """Mean photon flux [ph/s/m^2/nm], shape ``(W,)``, at scalar time.

        Linear tables (including atmosphere loaders) are integrated exactly,
        including partial end intervals, using the trapezoid rule on each
        linear segment. Cubic spectra use Gauss-Legendre quadrature as in
        ``AbstractStar``. Width zero returns ``spec_flux_density``.
        """
        if self._interpolation_method != "linear":
            return super().bin_flux_density(
                centers_nm, widths_nm, time_jd, n_nodes=n_nodes
            )
        weights = self._bin_weights(centers_nm, widths_nm)
        flux = (
            self._flux_interp.f
            if self._times_jd.size == 1
            else self.spec_flux_density(self._wavelengths_nm, time_jd)
        )
        mean = weights @ flux
        point = self.spec_flux_density(jnp.asarray(centers_nm), time_jd)
        centers, widths = jnp.asarray(centers_nm), jnp.asarray(widths_nm)
        point_bin = centers - widths / 2 == centers + widths / 2
        return jnp.where(point_bin, point, mean)

    def __repr__(self) -> str:
        """One-line summary of metadata + wavelength/time grid extent."""
        m_solar = self.Ms_kg / Msun2kg
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
