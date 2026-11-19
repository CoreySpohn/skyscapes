"""Star models for the scene hierarchy.

``AbstractStar`` declares the ``photon_flux_density`` hook. ``FlatStar`` is a
flat-spectrum stand-in useful for ETC runs. ``BlackbodyStar`` supplies a
diluted Planck spectrum.
``Star`` converts Jansky flux data to photons once and interpolates over
wavelength alone for a single epoch, or (wavelength, time) for multiple
epochs, with optional surface metadata.

Distance is supplied by the system when evaluating physical spectra;
apparent flux tables and flat spectra retain their measured normalization.
"""

from abc import abstractmethod
from typing import Literal

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import pc2m
from hwoutils.conversions import angular_diameter_arcsec, jy_to_photons_per_nm_per_m2
from hwoutils.planck import planck_photon_radiance
from jaxtyping import Array

from .spots import StarSpots


class AbstractStar(eqx.Module):
    """Abstract stellar source; observer distance is supplied at evaluation.

    All ``time_d`` arguments are days from the scene epoch.
    """

    @abstractmethod
    def photon_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Observer photon flux density [ph s^-1 m^-2 nm^-1].

        Wavelengths are nm arrays of any shape; the result preserves that
        shape, with tabulated stars also broadcasting against ``time_d``.
        Time is days from the scene epoch. Physical spectra use ``dist_pc``;
        apparent spectra accept and ignore it. Concrete models evaluate
        natively in photon units, without a conversion at evaluation time.
        """

    @abstractmethod
    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Apparent stellar diameter in arcseconds at observer distance dist_pc."""

    def spectral_breaks_nm(self) -> np.ndarray:
        """Analytic stellar spectra have no wavelength breaks."""
        return np.empty(0)


class FlatStar(AbstractStar):
    """Flat-spectrum star -- constant flux independent of wavelength or time.

    Attributes:
        flux_phot_per_nm_m2: Constant observer flux density in ph/s/m^2/nm.
        fixed_diameter_arcsec: Angular diameter in arcseconds, default 0.0 (point star).
    """

    flux_phot_per_nm_m2: float
    fixed_diameter_arcsec: float = 0.0

    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Return the stored apparent diameter in arcseconds, ignoring dist_pc."""
        return jnp.asarray(self.fixed_diameter_arcsec)

    def photon_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Constant photon flux [ph s^-1 m^-2 nm^-1], wavelength-shaped.

        ``time_d`` is days from the scene epoch.

        ``time_d`` and ``dist_pc`` are accepted for the common interface
        but ignored: this is an apparent flux model.
        """
        wl = jnp.asarray(wavelength_nm)
        return jnp.full_like(wl, self.flux_phot_per_nm_m2, dtype=wl.dtype)

    def __repr__(self) -> str:
        """Summarize flux and nonzero angular diameter."""
        diameter = (
            f", diam={self.fixed_diameter_arcsec:.3g} arcsec"
            if self.fixed_diameter_arcsec
            else ""
        )
        return f"FlatStar(flux={self.flux_phot_per_nm_m2:.3g} ph/s/m^2/nm{diameter})"


class BlackbodyStar(AbstractStar):
    """Planck blackbody spectrum diluted by the star's distance.

    Uses the Planck function from Rybicki & Lightman (1979),
    Radiative Processes in Astrophysics, divided by the photon energy.
    The surface flux is pi times the photon spectral radiance.

    Attributes:
        teff_K: Effective temperature in kelvin.
        radius_m: Stellar radius in meters.
    """

    teff_K: float
    radius_m: float

    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Small-angle diameter, 2 R / d, in arcseconds.

        Uses the small-angle relation from Carroll & Ostlie (2007),
        An Introduction to Modern Astrophysics.
        """
        return jnp.asarray(angular_diameter_arcsec(self.radius_m, dist_pc))

    def photon_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Photon flux density at the observer [ph s^-1 m^-2 nm^-1].

        ``time_d`` is days from the scene epoch.

        The result has wavelength_nm's shape. ``time_d`` is part of
        the AbstractStar interface but ignored here.
        """
        return (
            planck_photon_radiance(wavelength_nm, self.teff_K)
            * jnp.pi
            * (self.radius_m / (dist_pc * pc2m)) ** 2
        )

    def __repr__(self) -> str:
        """Compact one-line summary of temperature and radius."""
        return f"BlackbodyStar(Teff={self.teff_K:.3g} K, radius={self.radius_m:.3g} m)"


class Star(AbstractStar):
    """Tabulated stellar flux with optional imported surface metadata.

    ``times_d`` stores days from the scene epoch, whose metadata belongs to
    the owning scene. A single epoch represents a static spectrum. Spot
    records are retained for later surface modeling; they do not alter the
    tabulated flux or automatically add spatially resolved stellar emission.
    """

    fixed_diameter_arcsec: float
    luminosity_Lsun: float
    spots: StarSpots | None
    metadata: dict = eqx.field(static=True)

    _interpolation_method: str = eqx.field(static=True)
    _wavelengths_nm: Array
    _times_d: Array
    _flux_density_phot: Array
    _flux_interp: interpax.Interpolator1D | interpax.Interpolator2D

    def __init__(
        self,
        *,
        wavelengths_nm: Array,
        times_d: Array,
        flux_density_jy: Array,
        fixed_diameter_arcsec: float = 0.0,
        luminosity_Lsun: float = 1.0,
        spots: StarSpots | None = None,
        metadata: dict | None = None,
        interpolation_method: Literal["linear", "cubic"] = "cubic",
    ):
        """Build the flux interpolant on ``times_d``, days from the scene epoch.

        Input F_nu [Jy] is converted once to photon density before interpolation.
        F_lambda = F_nu * Jy * c / lambda_m**2; dividing by photon energy
        h*c/lambda_m and multiplying by nm2m gives F_nu*Jy/(h*wavelength_nm)
        in ph s^-1 m^-2 nm^-1. The hwoutils conversion supplies the constants.
        Linear interpolation of photon density avoids overshoot.
        """
        if interpolation_method not in ("linear", "cubic"):
            raise ValueError("interpolation_method must be linear or cubic.")
        self._interpolation_method = interpolation_method
        self.fixed_diameter_arcsec = fixed_diameter_arcsec
        self.luminosity_Lsun = luminosity_Lsun
        self.spots = spots
        self.metadata = {} if metadata is None else dict(metadata)
        self._wavelengths_nm = wavelengths_nm
        self._times_d = times_d

        self._flux_density_phot = jax.vmap(
            jy_to_photons_per_nm_per_m2, in_axes=(1, None), out_axes=1
        )(flux_density_jy, wavelengths_nm)

        if times_d.size == 1:
            self._flux_interp = interpax.Interpolator1D(
                wavelengths_nm,
                self._flux_density_phot[:, 0],
                method=interpolation_method,
            )
        else:
            self._flux_interp = interpax.Interpolator2D(
                wavelengths_nm,
                times_d,
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

    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Return the stored apparent diameter in arcseconds, ignoring dist_pc."""
        return jnp.asarray(self.fixed_diameter_arcsec)

    def photon_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Apparent photon flux [ph s^-1 m^-2 nm^-1]; ``dist_pc`` is ignored.

        Interpolate photon density, broadcasting wavelength and time shapes.
        ``time_d`` is days from the scene epoch.
        """
        if self._times_d.size == 1:
            # A single-epoch source is a static spectrum, not a degenerate spline.
            wavelength_nm, _ = jnp.broadcast_arrays(wavelength_nm, time_d)
            return self._flux_interp(wavelength_nm.reshape(-1)).reshape(
                wavelength_nm.shape
            )
        wave, time = jnp.broadcast_arrays(wavelength_nm, time_d)
        return self._flux_interp(wave.reshape(-1), time.reshape(-1)).reshape(wave.shape)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return table breakpoints [nm] for consumer quadrature.

        Every interpolation method reports the table breakpoints. A linearly
        interpolated photon table is integrated exactly by a Gauss-Legendre
        rule split at the knots, even at order one (Davis and Rabinowitz 1984,
        Methods of Numerical Integration).
        """
        return np.unique(self.wavelengths_nm)

    def __repr__(self) -> str:
        """One-line summary of metadata + wavelength/time grid extent."""
        wl_min = float(jnp.min(self._wavelengths_nm))
        wl_max = float(jnp.max(self._wavelengths_nm))
        n_t = int(self._times_d.shape[0])
        return (
            f"Star("
            f"L={self.luminosity_Lsun:.3g} Lsun, "
            f"diam={self.fixed_diameter_arcsec:.3g} arcsec, "
            f"wl={wl_min:.0f}-{wl_max:.0f} nm, n_times={n_t})"
        )
