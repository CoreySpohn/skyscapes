"""Star models for the scene hierarchy.

``AbstractStar`` declares the ``spec_flux_density`` hook. ``FlatStar`` is a
flat-spectrum stand-in useful for ETC runs. ``BlackbodyStar`` supplies a
diluted Planck spectrum.
``Star`` converts Jansky flux data to photons and interpolates over
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
from hwoutils.constants import pc2m, rad2arcsec
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from jaxtyping import Array

from .._planck import _planck_photon_flux
from ..spectral import _model_max_knots, bin_mean
from .spots import StarSpots


class AbstractStar(eqx.Module):
    """Abstract stellar source; observer distance is supplied at evaluation.

    All ``time_d`` arguments are days from the scene epoch.
    """

    @abstractmethod
    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array | None = None,
    ) -> Array:
        """Return spectral flux density in ph/s/m^2/nm.

        ``time_d`` is days from the scene epoch.
        """

    @property
    def quadrature_knots_nm(self) -> Array | None:
        """Spectral integration knots [nm], or None for an analytic spectrum."""
        return None

    def bin_mean_flux_density(
        self,
        centers_nm: Array,
        widths_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array | None = None,
        n_nodes: int = 4,
    ) -> Array:
        """Mean photon flux density per bin [ph/s/m^2/nm], shape ``(W,)``.

        ``time_d`` is days from the scene epoch.

        Centers and nonnegative full widths have shape ``(W,)``; time is
        scalar. Width zero returns the point value. Analytic spectra use a
        static ``n_nodes``-point Gauss-Legendre rule (default 4). Its error
        is O(width**(2*n_nodes)); see Davis and Rabinowitz (1984), Methods
        of Numerical Integration. For a 5772 K blackbody, centers 400-5000
        nm and widths <= 20 percent of center, the default relative error
        is below 1e-8. Wider bins or Wien tails need a convergence check.
        Linear tables split at ``quadrature_knots_nm`` and are exact. The
        static interval cap is inferred from concrete geometry; traced geometry
        uses a conservative full-table cap to preserve this method's signature.
        Close over concrete bins when compiling to retain bounded work.
        """
        knots = self.quadrature_knots_nm
        return bin_mean(
            lambda w: self.spec_flux_density(w, time_d, dist_pc=dist_pc),
            centers_nm,
            widths_nm,
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=_model_max_knots(centers_nm, widths_nm, knots),
        )


class FlatStar(AbstractStar):
    """Flat-spectrum star -- constant flux independent of wavelength or time.

    Attributes:
        flux_phot_per_nm_m2: Constant observer flux density in ph/s/m^2/nm.
        diameter_arcsec: Angular diameter in arcseconds, default 0.0 (point star).
    """

    flux_phot_per_nm_m2: float
    diameter_arcsec: float = 0.0

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array | None = None,
    ) -> Array:
        """Constant flux, broadcast to wavelength_nm's shape.

        ``time_d`` is days from the scene epoch.

        ``time_d`` and ``dist_pc`` are accepted for the common interface
        but ignored: this is an apparent flux model.
        """
        wl = jnp.asarray(wavelength_nm)
        return jnp.full_like(wl, self.flux_phot_per_nm_m2, dtype=wl.dtype)

    def __repr__(self) -> str:
        """Summarize flux and nonzero angular diameter."""
        diameter = (
            f", diam={self.diameter_arcsec:.3g} arcsec" if self.diameter_arcsec else ""
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

    def diameter_arcsec(self, *, dist_pc: Array) -> Array:
        """Small-angle diameter, 2 R / d, in arcseconds.

        Uses the small-angle relation from Carroll & Ostlie (2007),
        An Introduction to Modern Astrophysics.
        """
        return 2.0 * self.radius_m / (dist_pc * pc2m) * rad2arcsec

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Photon flux density at the observer in ph/s/m^2/nm.

        ``time_d`` is days from the scene epoch.

        The result has wavelength_nm's shape. ``time_d`` is part of
        the AbstractStar interface but ignored here.
        """
        return _planck_photon_flux(wavelength_nm, self.teff_K, self.radius_m, dist_pc)

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

    ra_deg: float
    dec_deg: float
    diameter_arcsec: float
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
        ra_deg: float = 0.0,
        dec_deg: float = 0.0,
        diameter_arcsec: float = 0.0,
        luminosity_Lsun: float = 1.0,
        spots: StarSpots | None = None,
        metadata: dict | None = None,
        interpolation_method: Literal["linear", "cubic"] = "cubic",
    ):
        """Build the flux interpolant on ``times_d``, days from the scene epoch.

        Linear interpolation avoids overshoot.
        """
        if interpolation_method not in ("linear", "cubic"):
            raise ValueError("interpolation_method must be linear or cubic.")
        self._interpolation_method = interpolation_method
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.diameter_arcsec = diameter_arcsec
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

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array | None = None,
    ) -> Array:
        """Scalar or array apparent flux [ph/s/m^2/nm]; ``dist_pc`` is ignored.

        ``time_d`` is days from the scene epoch.
        """
        if self._times_d.size == 1:
            # A single-epoch source is a static spectrum, not a degenerate spline.
            wavelength_nm, _ = jnp.broadcast_arrays(wavelength_nm, time_d)
            return self._flux_interp(wavelength_nm)
        return self._flux_interp(wavelength_nm, time_d)

    @property
    def quadrature_knots_nm(self) -> Array | None:
        """Return linear-table knots [nm]; cubic spectra retain one rule per bin.

        Splitting linear segments makes Gauss-Legendre integration exact even
        at order one (Davis and Rabinowitz 1984, Methods of Numerical Integration).
        """
        if self._interpolation_method == "linear":
            return self.wavelengths_nm
        return None

    def __repr__(self) -> str:
        """One-line summary of metadata + wavelength/time grid extent."""
        wl_min = float(jnp.min(self._wavelengths_nm))
        wl_max = float(jnp.max(self._wavelengths_nm))
        n_t = int(self._times_d.shape[0])
        return (
            f"Star("
            f"L={self.luminosity_Lsun:.3g} Lsun, "
            f"diam={self.diameter_arcsec:.3g} arcsec, "
            f"RA/Dec=({self.ra_deg:.4f}, {self.dec_deg:.4f}) deg, "
            f"wl={wl_min:.0f}-{wl_max:.0f} nm, n_times={n_t})"
        )
