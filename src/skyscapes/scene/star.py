"""Star models for the scene hierarchy.

``AbstractStar`` declares the ``photon_flux_density`` method, concrete over the
private ``_photon_flux_density`` hook, and the ``flux_scale`` leaf applied in
it. ``FlatStar`` is a
flat-spectrum stand-in useful for ETC runs. ``BlackbodyStar`` supplies a
diluted Planck spectrum.
``Star`` converts Jansky flux data to photons once and interpolates over
wavelength alone for a single epoch, or (wavelength, time) for multiple
epochs, with optional surface metadata.

Distance is supplied by the system when evaluating physical spectra;
apparent flux tables and flat spectra retain their measured normalization.

``node_flux_density`` reads a star on the nodes of a band rule. A smooth star
is sampled at the nodes; a tabulated ``Star`` is reduced to node-effective
values by the operator ``optixstuff.spectral.Resolution`` holds for its knots,
so the band integral is exact in the table however rough it is.
"""

from abc import abstractmethod
from typing import Literal

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import pc2m
from hwoutils.conversions import (
    angular_diameter_arcsec,
    jy_to_photons_per_nm_per_m2,
    photons_per_nm_per_m2_to_jy,
)
from hwoutils.leaves import as_float_leaf
from hwoutils.planck import planck_photon_radiance
from hwoutils.precision import like
from jaxtyping import Array
from optixstuff.spectral import Knots, digest_knots

from .._dtype import _working_dtype
from .._flux_scale import check_flux_scale
from .._sample_time import SampleOrigin
from .spots import StarSpots

# Digest names of the interpolations a Star offers; a linear interpolant is
# unique, so every linear table on one grid shares a rule.
_TABLE_METHODS = {"linear": "linear", "cubic": "interpax-cubic"}
# Gauss-Legendre points of the band mean a smooth star is calibrated with.
_CALIBRATION_POINTS = 8


class AbstractStar(eqx.Module):
    """Abstract stellar source; observer distance is supplied at evaluation.

    All ``time_d`` arguments are days from the scene epoch.

    ``flux_scale`` is a dimensionless factor, default one, on all light that
    comes from the star: the leakage, every reflected-light planet and a disk
    defined relative to the star (the disk hook takes ``star=``). It does not
    scale the local zodi, a thermal planet model or the instrument's thermal
    term. It is applied once, in :meth:`photon_flux_density`; a concrete star
    implements :meth:`_photon_flux_density` and never sees it. The scale is
    linear, and the image is bilinear in it and any planet or disk scale: a
    star scale and a planet scale are correlated, not degenerate. It is
    checked finite, not positive.

    Attributes:
        flux_scale: Dimensionless factor on all light from the star, default one.
    """

    flux_scale: eqx.AbstractVar[Array]

    def __check_init__(self):
        """Require a finite ``flux_scale``."""
        check_flux_scale(self.flux_scale)

    def photon_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Observer photon flux density [ph s^-1 m^-2 nm^-1], times ``flux_scale``.

        Wavelengths are nm arrays of any shape; the result preserves that
        shape, with tabulated stars also broadcasting against ``time_d``.
        Time is days from the scene epoch. Physical spectra use ``dist_pc``;
        apparent spectra accept and ignore it.

        Args:
            wavelength_nm: Wavelengths [nm], any shape.
            time_d: Time, days from the scene epoch.
            dist_pc: Observer distance [pc].

        Returns:
            Photon flux density [ph s^-1 m^-2 nm^-1] at the observer.
        """
        return self.flux_scale * self._photon_flux_density(
            wavelength_nm, time_d, dist_pc=dist_pc
        )

    @abstractmethod
    def _photon_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Unscaled observer photon flux density [ph s^-1 m^-2 nm^-1].

        The hook a concrete star implements; :meth:`photon_flux_density`
        applies ``flux_scale`` to it. Concrete models evaluate natively in
        photon units, without a conversion at evaluation time.
        """

    def node_flux_density(
        self,
        quadrature,
        resolution,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Node-effective photon flux density [ph s^-1 m^-2 nm^-1], with ``flux_scale``.

        The values on the band rule's nodes whose weighted sum with any
        response of degree below a piece's order is the band integral of the
        star times that response. A smooth star returns its samples at the
        nodes; a tabulated star returns the reduction of its table by the
        operator ``resolution`` holds for its knots on ``quadrature``.

        Args:
            quadrature: Band rule from ``optixstuff.spectral.Quadrature.build``.
            resolution: ``optixstuff.spectral.Resolution`` built for the rule and
                the scene's tables; a smooth star does not read it.
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            dist_pc: Observer distance [pc].

        Returns:
            Node-effective photon flux density, shape ``(W, Q, T)``.
        """
        return self.flux_scale * self._node_flux_density(
            quadrature, resolution, time_d, dist_pc=dist_pc
        )

    def _node_flux_density(self, quadrature, resolution, time_d, *, dist_pc):
        """Unscaled node values; the default samples a smooth star at the nodes."""
        nodes = quadrature.nodes_nm
        time = jnp.atleast_1d(jnp.asarray(time_d))
        values = self._photon_flux_density(nodes[..., None], time, dist_pc=dist_pc)
        return jnp.broadcast_to(values, nodes.shape + time.shape)

    def calibration_flux_density(self, band, time_d, *, dist_pc: Array) -> Array:
        """Mean of the unscaled photon flux density over a band [ph s^-1 m^-2 nm^-1].

        The calibration a disk under the bolometric luminosity rule divides
        by: a magnitude is a band quantity, so the star is averaged over the
        band instead of read at its center. A smooth star uses an eight-point
        Gauss-Legendre rule; ``flux_scale`` is not applied, as in
        :meth:`_photon_flux_density`.

        Args:
            band: ``optixstuff.spectral.Band`` to average over.
            time_d: Time of the calibration, days from the scene epoch.
            dist_pc: Observer distance [pc].

        Returns:
            The band-mean photon flux density, a scalar.
        """
        x, w = np.polynomial.legendre.leggauss(_CALIBRATION_POINTS)
        nodes = band.center_nm + 0.5 * band.width_nm * like(x, self.flux_scale)
        values = self._photon_flux_density(nodes, jnp.asarray(time_d), dist_pc=dist_pc)
        return jnp.sum(like(w / 2, values) * values)

    @abstractmethod
    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Apparent stellar diameter in arcseconds at observer distance dist_pc.

        Args:
            dist_pc: Observer distance [pc].

        Returns:
            Apparent angular diameter [arcsec].
        """

    def spectral_breaks_nm(self) -> np.ndarray:
        """Analytic stellar spectra have no wavelength breaks.

        Returns:
            An empty array of breakpoint wavelengths [nm].
        """
        return np.empty(0)


class FlatStar(AbstractStar):
    """Flat-spectrum star -- constant flux independent of wavelength or time.

    Attributes:
        flux_phot_per_nm_m2: Constant observer flux density in ph/s/m^2/nm.
        fixed_diameter_arcsec: Angular diameter in arcseconds, default 0.0 (point star).
        flux_scale: Dimensionless factor on the flux, default one; see
            :class:`AbstractStar`.
    """

    flux_phot_per_nm_m2: float
    fixed_diameter_arcsec: float = 0.0
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)

    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Return the stored apparent diameter in arcseconds, ignoring dist_pc.

        Args:
            dist_pc: Observer distance [pc]; ignored.

        Returns:
            The stored apparent diameter [arcsec].
        """
        return like(self.fixed_diameter_arcsec, self.flux_scale)

    def _photon_flux_density(
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
        dtype = jnp.result_type(wl, self.flux_scale, self.flux_phot_per_nm_m2)
        return jnp.full_like(wl, self.flux_phot_per_nm_m2, dtype=dtype)

    def __repr__(self) -> str:
        """Summarize flux and nonzero angular diameter.

        Returns:
            A one-line text summary.
        """
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
        flux_scale: Dimensionless factor on the flux, default one; see
            :class:`AbstractStar`.
    """

    teff_K: float
    radius_m: float
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)

    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Small-angle diameter, 2 R / d, in arcseconds.

        Uses the small-angle relation from Carroll & Ostlie (2007),
        An Introduction to Modern Astrophysics.

        Args:
            dist_pc: Observer distance [pc].

        Returns:
            Apparent angular diameter [arcsec].
        """
        return jnp.asarray(angular_diameter_arcsec(self.radius_m, dist_pc))

    def _photon_flux_density(
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
        dtype = _working_dtype(
            wavelength_nm, dist_pc, self.teff_K, self.radius_m, self.flux_scale
        )
        wave = jnp.asarray(wavelength_nm, dtype=dtype)
        distance = jnp.asarray(dist_pc, dtype=dtype)
        return (
            planck_photon_radiance(wave, self.teff_K)
            * jnp.pi
            * (self.radius_m / (distance * pc2m)) ** 2
        )

    def __repr__(self) -> str:
        """Compact one-line summary of temperature and radius.

        Returns:
            A one-line text summary.
        """
        return f"BlackbodyStar(Teff={self.teff_K:.3g} K, radius={self.radius_m:.3g} m)"


class Star(AbstractStar):
    """Tabulated stellar flux with optional imported spot records.

    ``times_d`` stores days from the scene epoch on construction. Shared-epoch
    imports keep knots relative to the first sample, at ``sample_origin.days``
    on the scene clock. A single epoch represents a static spectrum. Spot
    records are retained for later surface modeling; they do not alter the
    tabulated flux or automatically add spatially resolved stellar emission.
    ``flux_scale`` is a dimensionless factor on the tabulated flux, default one;
    see :class:`AbstractStar`.

    The star is an ``optixstuff.spectral.SpectralTable``: its wavelength grid
    is a static ``Knots`` (float64 host data, never a leaf), ``knot_digest``
    names that grid and the interpolation, and :meth:`node_flux_density` reads
    the operator a ``Resolution`` holds for that digest. The knots must be
    concrete; the flux table may be traced. ``eqx.tree_at`` cannot reach the
    knots: build a new star (or ``dataclasses.replace``) for another grid.

    Attributes:
        knots: Static wavelength grid of the table [nm].
        flux_scale: Dimensionless factor on the tabulated flux, default one.
        fixed_diameter_arcsec: Apparent stellar diameter [arcsec]; zero is a
            point star.
        luminosity_Lsun: Stellar luminosity [solar luminosities].
        spots: Imported spot history, or None.
        sample_origin: First-sample offset of a shared-epoch import, or None.
    """

    knots: Knots = eqx.field(static=True)
    flux_scale: Array = eqx.field(converter=as_float_leaf)
    fixed_diameter_arcsec: Array = eqx.field(converter=as_float_leaf)
    luminosity_Lsun: Array = eqx.field(converter=as_float_leaf)
    spots: StarSpots | None
    sample_origin: SampleOrigin | None

    _interpolation_method: str = eqx.field(static=True)
    _times_d: Array = eqx.field(converter=as_float_leaf)
    _flux_density_phot: Array = eqx.field(converter=as_float_leaf)

    def __init__(
        self,
        *,
        wavelengths_nm: Array,
        times_d: Array,
        flux_density_jy: Array,
        fixed_diameter_arcsec: float = 0.0,
        luminosity_Lsun: float = 1.0,
        spots: StarSpots | None = None,
        interpolation_method: Literal["linear", "cubic"] = "cubic",
        flux_scale: Array | float = 1.0,
    ):
        """Store the photon table on ``times_d``, days from the scene epoch.

        The photon table is the one stored representation and a live leaf; the
        interpolant is built from it on each call. Input F_nu [Jy] is converted
        once to photon density before interpolation.
        F_lambda = F_nu * Jy * c / lambda_m**2; dividing by photon energy
        h*c/lambda_m and multiplying by nm2m gives F_nu*Jy/(h*wavelength_nm)
        in ph s^-1 m^-2 nm^-1. The hwoutils conversion supplies the constants.
        Linear interpolation of photon density avoids overshoot.

        Args:
            wavelengths_nm: Wavelength grid [nm], shape ``(n_wavelengths,)``,
                concrete: a ``Knots`` or a host array, kept in float64.
            times_d: Epochs, days from the scene epoch, shape ``(n_times,)``.
            flux_density_jy: Flux density F_nu [Jy], shape
                ``(n_wavelengths, n_times)``.
            fixed_diameter_arcsec: Apparent stellar diameter [arcsec]; zero is
                a point star.
            luminosity_Lsun: Stellar luminosity [solar luminosities].
            spots: Imported spot history, or None.
            interpolation_method: ``"linear"`` or ``"cubic"`` interpolation of
                the photon table.
            flux_scale: Dimensionless factor on the tabulated flux, default one.
        """
        if interpolation_method not in ("linear", "cubic"):
            raise ValueError("interpolation_method must be linear or cubic.")
        self._interpolation_method = interpolation_method
        self.flux_scale = flux_scale
        self.fixed_diameter_arcsec = fixed_diameter_arcsec
        self.luminosity_Lsun = luminosity_Lsun
        self.sample_origin = None
        self.spots = spots
        self.knots = (
            wavelengths_nm
            if isinstance(wavelengths_nm, Knots)
            else Knots(wavelengths_nm)
        )
        self._times_d = times_d
        # Derived under a trace too, so from the converted local.
        self._flux_density_phot = jax.vmap(
            jy_to_photons_per_nm_per_m2, in_axes=(1, None), out_axes=1
        )(as_float_leaf(flux_density_jy), self.knots.array())

    @property
    def knot_digest(self) -> str:
        """Digest of the static knots and the interpolation method."""
        return digest_knots(self.knots, _TABLE_METHODS[self._interpolation_method])

    @property
    def flux_density_jy(self) -> Array:
        """Recover the tabulated spectral flux density.

        Returns:
            F_nu [Jy], shape ``(n_wavelengths, n_times)``.
        """
        photons = self._flux_density_phot
        return jax.vmap(photons_per_nm_per_m2_to_jy, in_axes=(1, None), out_axes=1)(
            photons, self.knots.array(photons.dtype)
        )

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Tabulated wavelength grid in nm, float64 host data (read-only)."""
        return self.knots.nm

    @property
    def wavelength_range_nm(self) -> tuple[float, float]:
        """Minimum and maximum tabulated wavelengths in nm as Python floats."""
        return float(self.knots.nm[0]), float(self.knots.nm[-1])

    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Return the stored apparent diameter in arcseconds, ignoring dist_pc.

        Args:
            dist_pc: Observer distance [pc]; ignored.

        Returns:
            The stored apparent diameter [arcsec].
        """
        return self.fixed_diameter_arcsec

    def _photon_flux_density(
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
        if self.sample_origin is not None:
            time_d = self.sample_origin.relative(time_d)
        return self._interpolate(wavelength_nm, time_d)

    def _interpolate(self, wavelength_nm, time_d):
        """Read the photon table at times relative to its first-sample origin."""
        method = self._interpolation_method
        dtype = _working_dtype(
            wavelength_nm,
            time_d,
            self._flux_density_phot,
            self._times_d,
            self.flux_scale,
        )
        wavelength_nm = jnp.asarray(wavelength_nm, dtype=dtype)
        time_d = jnp.asarray(time_d, dtype=dtype)
        values = jnp.asarray(self._flux_density_phot, dtype=dtype)
        knots = self.knots.array(dtype)
        outside = jnp.asarray(jnp.nan, dtype=dtype)
        if self._times_d.size == 1:
            # A single-epoch source is a static spectrum, not a degenerate spline.
            wavelength_nm, _ = jnp.broadcast_arrays(wavelength_nm, time_d)
            interp = interpax.Interpolator1D(
                knots, values[:, 0], method=method, extrap=outside
            )
            return interp(wavelength_nm.reshape(-1)).reshape(wavelength_nm.shape)
        wave, time = jnp.broadcast_arrays(wavelength_nm, time_d)
        interp = interpax.Interpolator2D(
            knots,
            jnp.asarray(self._times_d, dtype=dtype),
            values,
            method=method,
            extrap=outside,
        )
        return interp(wave.reshape(-1), time.reshape(-1)).reshape(wave.shape)

    @property
    def spectral_degree(self) -> int:
        """Polynomial degree of the photon table between knots: 1 linear, 3 cubic."""
        return 1 if self._interpolation_method == "linear" else 3

    def spectral_interpolant(self):
        """Return the table's wavelength interpolation as a function of its values.

        The function ``f(values, wavelength_nm)`` interpolates a photon table of
        shape ``(N,)`` on the star's knots with the star's own method; it is
        linear in the values, so ``Resolution.build`` composes it into the
        operator without restating the scheme.

        Returns:
            The interpolation function, on the host knots.
        """
        knots = self.spectral_breaks_nm()
        method = self._interpolation_method

        def interpolant(values, wavelength_nm):
            dtype = _working_dtype(wavelength_nm, values)
            wave = jnp.asarray(wavelength_nm, dtype=dtype)
            interp = interpax.Interpolator1D(
                jnp.asarray(knots, dtype=dtype),
                jnp.asarray(values, dtype=dtype),
                method=method,
                extrap=jnp.asarray(jnp.nan, dtype=dtype),
            )
            return interp(wave.reshape(-1)).reshape(wave.shape)

        return interpolant

    def _table_at(self, time_d):
        """Photon table at the query times, shape ``(N, T)``.

        Interpolating each column in time first and then in wavelength equals
        the two-dimensional interpolant: it is a tensor product.
        """
        if self._times_d.size == 1:
            column = self._flux_density_phot[:, :1]
            return jnp.broadcast_to(column, (column.shape[0], time_d.size))
        interp = interpax.Interpolator1D(
            jnp.asarray(self._times_d, dtype=time_d.dtype),
            jnp.asarray(self._flux_density_phot.T, dtype=time_d.dtype),
            method=self._interpolation_method,
            extrap=jnp.asarray(jnp.nan, dtype=time_d.dtype),
        )
        return interp(time_d).T

    def calibration_flux_density(self, band, time_d, *, dist_pc: Array) -> Array:
        """Refuse: a band mean of a rough table needs the operator, not a point rule.

        Args:
            band: Calibration band.
            time_d: Time of the calibration, days from the scene epoch.
            dist_pc: Observer distance [pc].

        Returns:
            Nothing: the call always raises.

        Raises:
            LookupError: Always. Read the star through ``Scene.sources`` and
                render the disk with ``Sources.node_star``, which carry the
                band mean computed by the table's operator.
        """
        raise LookupError(
            "a tabulated Star has no point-rule band mean: its lines are "
            "narrower than any affordable rule. Compute the calibration through "
            "Scene.sources (build the Resolution with the calibration band's "
            "quadrature from Scene.calibration_bands) and render the disk with "
            "Sources.node_star"
        )

    def _node_flux_density(self, quadrature, resolution, time_d, *, dist_pc):
        """Unscaled node-effective values from the operator for this table."""
        rule = resolution.rule(quadrature, self)
        if self.sample_origin is not None:
            time_d = self.sample_origin.relative(time_d)
        dtype = _working_dtype(
            time_d,
            quadrature.nodes_nm,
            self._times_d,
            self._flux_density_phot,
            self.flux_scale,
        )
        time = jnp.atleast_1d(jnp.asarray(time_d, dtype=dtype))
        if rule.kind == "table":
            return rule.apply(self._table_at(time))
        points = quadrature.nodes_nm if rule.kind == "identity" else rule.fine_nm
        return rule.apply(self._interpolate(points[..., None], time))

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return table breakpoints [nm] for consumer quadrature.

        Every interpolation method reports the table breakpoints. A linearly
        interpolated photon table is integrated exactly by a Gauss-Legendre
        rule split at the knots, even at order one (Davis and Rabinowitz 1984,
        Methods of Numerical Integration).

        Returns:
            The static knots [nm], float64, sorted ascending.
        """
        return self.knots.nm

    def __repr__(self) -> str:
        """One-line summary of luminosity, diameter and wavelength/time extent.

        Returns:
            A one-line text summary.
        """
        wl_min, wl_max = self.wavelength_range_nm
        n_t = int(self._times_d.shape[0])
        return (
            f"Star("
            f"L={float(self.luminosity_Lsun):.3g} Lsun, "
            f"diam={float(self.fixed_diameter_arcsec):.3g} arcsec, "
            f"wl={wl_min:.0f}-{wl_max:.0f} nm, n_times={n_t})"
        )


class _NodeStar(AbstractStar):
    """The host star read at band-rule nodes, built by ``Sources.node_star``.

    ``photon_flux_density`` returns the node-effective value in place of a
    spectrum: a scalar at one node, broadcast to any wavelength shape, or one
    value per node, for a read vectorised over the nodes whose wavelength
    array has the same shape. A source defined relative to the star (a disk)
    rendered with it therefore carries the star's band integral, not its point
    sample. No value is divided, so a node where the star is zero is no
    special case.

    Attributes:
        values: Unscaled node-effective photon flux density
            [ph s^-1 m^-2 nm^-1], a scalar or one value per node.
        diameter_arcsec: Apparent diameter of the host star [arcsec].
        luminosity_Lsun: Luminosity of the host star [solar luminosities], or
            None when the host does not carry one.
        flux_scale: The host star's ``flux_scale``.
        calibration: Unscaled band mean of the star at the scene epoch in each
            of ``calibration_bands`` [ph s^-1 m^-2 nm^-1], shape ``(C,)``.
        calibration_bands: Static calibration bands, in the order of
            ``calibration``.
    """

    values: Array
    diameter_arcsec: Array
    luminosity_Lsun: Array | None
    flux_scale: Array
    calibration: Array = eqx.field(default_factory=lambda: jnp.zeros((0,)))
    calibration_bands: tuple = eqx.field(static=True, default=())

    def angular_diameter_arcsec(self, dist_pc: Array) -> Array:
        """Return the host star's apparent diameter, ignoring dist_pc.

        Args:
            dist_pc: Observer distance [pc]; ignored.

        Returns:
            Apparent diameter [arcsec].
        """
        return self.diameter_arcsec

    def _photon_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Node values in the wavelength array's shape; time and distance are fixed."""
        shape = jnp.shape(wavelength_nm)
        if jnp.ndim(self.values) and jnp.shape(self.values) != shape:
            raise ValueError(
                f"this node star holds values of shape {jnp.shape(self.values)}; "
                f"read it at wavelengths of that shape, not {shape}, or take the "
                "star of one node with Sources.node_star(w, q)"
            )
        return jnp.broadcast_to(self.values, shape)

    def calibration_flux_density(self, band, time_d, *, dist_pc: Array) -> Array:
        """Return the band mean ``Scene.sources`` computed for this calibration band.

        The star is read at the scene epoch whatever ``time_d`` is, as the
        calibration of a disk always is.

        Args:
            band: Calibration band; one of ``Scene.calibration_bands()``.
            time_d: Ignored.
            dist_pc: Observer distance [pc]; ignored.

        Returns:
            The unscaled band-mean photon flux density, a scalar.

        Raises:
            LookupError: ``band`` was not among the scene's calibration bands.
        """
        if band not in self.calibration_bands:
            raise LookupError(
                f"these sources hold no calibration for {band}: it is not one of "
                "Scene.calibration_bands()"
            )
        return self.calibration[self.calibration_bands.index(band)]
