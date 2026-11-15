"""GraterDisk: Augereau 1999 scattered-light disk as an eqx.Module.

The Augereau et al. 1999 model is a two-power-law dust density distribution
with a flared vertical profile, scattering with a single Henyey-Greenstein
phase function. This module exposes the model in skyscapes' eqx.Module
idiom so its parameters are PyTree leaves (fittable through the standard
scene-as-PyTree pattern).

References:
    - Augereau, J.-C., Lagrange, A.-M., Mouillet, D., Papaloizou, J. C. B.,
      & Grorod, P. A. 1999, A&A, 348, 557.
    - `grater-jax <https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX>`_
      (Kondapalli, Lewis, Ashcraft, Millar-Blanchaer).

Notes on convention:
    Disk orientation (``incl_deg`` / ``pa_deg``) is passed in at
    ``surface_brightness`` time rather than stored on the disk -- the
    System's ``midplane_inc_deg`` / ``midplane_pa_deg`` drives every
    component consistently and there is one source of truth.

    Wavelength dependence enters via the HG asymmetry parameter ``g_HG``
    and the albedo-like scaling ``Ag``, both stored as 1-D grids over
    ``wavelengths_nm`` and interpolated at call time with a cubic
    spline. The geometric kernel itself is wavelength-independent, so
    ``surface_brightness`` keeps its scalar-in / ``(ny, nx)``-out shape
    contract; ``vmap`` over wavelength gives a cube without changing
    the interface.

    The line-of-sight rule is a static choice (``quadrature``). The default
    Gauss-Legendre rule integrates over the exact extent of the flared layer
    along each line of sight and is smooth through edge-on, so exactly
    edge-on images and inclination derivatives are supported. Smooth radial
    tapers remove hard sample-crossing jumps.
    Inclinations above 90 view the disk from its other face and return the
    physical, nonnegative map, mirrored across the line of nodes relative to
    ``180 - incl``.
"""

from __future__ import annotations

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from ._los import (
    check_quadrature,
    layer_expansion_parameter,
    los_integrate_scattered,
)
from .base import AbstractDisk


def henyey_greenstein(cos_phi: Array, g: Array) -> Array:
    """Henyey-Greenstein single-scattering phase function.

    Args:
        cos_phi: Cosine of scattering angle, any broadcastable shape.
        g: Asymmetry parameter in (-1, 1). g > 0 is forward-peaked,
            g < 0 is backward-peaked, g = 0 is isotropic.

    Returns:
        Phase function value, same shape as ``cos_phi``.
    """
    one_minus_g2 = 1.0 - g * g
    # Explicit x * sqrt(x) is faster than pow(x, 1.5) on CPU.
    inner = 1.0 + g * g - 2.0 * g * cos_phi
    denom = inner * jnp.sqrt(inner)
    return one_minus_g2 / (4.0 * jnp.pi * denom)


def two_power_law_density(
    r: Array,
    z: Array,
    sma_AU: Array,
    alpha_in: Array,
    alpha_out: Array,
    ksi0_AU: Array,
    gamma: Array,
    beta: Array,
    *,
    fixed_vertical_profile: bool = False,
) -> Array:
    """Augereau 1999 two-power-law radial density with flared vertical profile.

    ``rho = (x^(-2 alpha_in) + x^(-2 alpha_out))^(-1/2) exp(-(|z| / h)^gamma)``
    with ``x = r / sma_AU`` and ``h = ksi0_AU x^beta``.

    Args:
        r: Cylindrical radius in disk frame [AU]. Must be strictly positive;
            callers are responsible for masking r values that are too small
            for the radial power law to evaluate cleanly in the chosen dtype
            (e.g. r_ratio^(-2 alpha_in) overflows float32 below r/sma ~ 1e-3
            at alpha_in = 5).
        z: Height above disk midplane [AU].
        sma_AU: Reference radius (radial profile peak) [AU].
        alpha_in: Inner power-law slope (positive convention).
        alpha_out: Outer power-law slope (negative convention).
        ksi0_AU: Vertical scale height at ``r = sma_AU`` [AU].
        gamma: Vertical profile exponent (2 = Gaussian, 1 = exponential).
        beta: Flaring index (0 = none, 1 = linear).
        fixed_vertical_profile: Static. When True the density is evaluated
            for ``gamma = 2`` and ``beta = 1`` without general powers (one
            logarithm, three exponentials and a reciprocal square root in
            place of five ``pow`` calls), which is 1.5 to 1.8 times faster
            and agrees with the general expression to rounding. ``gamma``
            and ``beta`` are then ignored and carry no gradient.

    Returns:
        Density in arbitrary units (relative), same shape as ``r`` and ``z``.
    """
    if fixed_vertical_profile:
        log_ratio = jnp.log(r / sma_AU)
        radial = jax.lax.rsqrt(
            jnp.exp(-2.0 * alpha_in * log_ratio) + jnp.exp(-2.0 * alpha_out * log_ratio)
        )
        height = z * sma_AU / (ksi0_AU * r)
        return radial * jnp.exp(-height * height)
    r_ratio = r / sma_AU
    radial = (r_ratio ** (-2.0 * alpha_in) + r_ratio ** (-2.0 * alpha_out)) ** (-0.5)
    h_r = ksi0_AU * r_ratio**beta
    vertical = jnp.exp(-((jnp.abs(z) / h_r) ** gamma))
    return radial * vertical


class GraterDisk(AbstractDisk):
    """Augereau 1999 scattered-light disk.

    Density, scattering, and spectral parameters are fittable PyTree leaves.
    Image geometry and LOS integration settings are static configuration.

    **Choosing the line-of-sight rule.** All rules converge to the same
    integral; the choice sets the truncation error and the cost, which is
    proportional to the node count. Errors below are the largest pixel
    error over the image peak for a narrow ring (``alpha = 5``,
    ``h / r = 0.05``) with tapered edges.

    - ``"legendre"`` (default) is accurate at every inclination. Use 24 to
      48 nodes up to 70 degrees (below 1e-4; 48 nodes reach 1e-5 at 60
      degrees and 6e-5 at 70) and 96 from 80 to 90 degrees (4e-4 to 2e-3).
      A broader or thicker disk needs fewer.
    - ``"gauss"`` needs only 7 nodes (2e-5 at 45 degrees, 2e-4 at 60) but
      holds only while :meth:`gauss_rule_eps` stays below about 0.45 over
      the whole prior (up to about 60 degrees for the ring above). Past
      that its error grows without bound whatever the node count, and no
      error is raised. Use ``"gauss_half"`` instead when ``gamma`` is, or
      may be fitted, away from 2 (``n_slices_los`` is then per side; 4 to
      6).
    - ``"trapezoid"`` is the rule of earlier releases, kept bit for bit. It
      is second order: 2e-2 at 31 nodes and 2e-3 at 101, mostly a uniform
      scale error of the image.

    **Fitting.** Set ``cutoff_width_AU > 0``. The hard ``rmin_AU`` /
    ``rmax_AU`` masks of the default make the image piecewise smooth in the
    parameters and add 1e-4 to 7e-4 of the peak to every rule's error. An
    image needs about 1e-4 of the peak for the model error to stay below
    the noise at a peak signal-to-noise ratio of 100 per pixel.

    Attributes:
        sma_AU: Reference radius [AU] (radial profile peak).
        alpha_in: Inner power-law slope (positive).
        alpha_out: Outer power-law slope (negative).
        ksi0_AU: Vertical scale height at ``sma_AU`` [AU].
        gamma: Vertical profile exponent (2 = Gaussian).
        beta: Flaring index (0 = none, 1 = linear).
        rmin_AU: Inner truncation radius [AU].
        rmax_AU: Outer truncation radius [AU].
        cutoff_width_AU: Nonnegative inward radial taper width [AU].
            Zero (default) preserves the hard rmin/rmax mask exactly.
            Positive values use a C2 quintic ramp from zero at the boundary
            to one a width inside the annulus. Widths larger than half the
            annulus overlap.
        wavelengths_nm: 1-D wavelength grid for ``g_HG_grid`` /
            ``Ag_grid``, shape ``(n_wave,)`` [nm]. Must be sorted.
            ``surface_brightness`` calls outside this range return NaN.
        g_HG_grid: Henyey-Greenstein asymmetry at each grid wavelength.
        Ag_grid: Albedo-like scaling at each grid wavelength.

        nx, ny: Output image shape.
        pixel_scale_arcsec: Pixel scale [arcsec/pixel].
        n_slices_los: Number of LOS nodes (per side of the midplane for
            ``"gauss_half"``).
        n_scale_heights: ``"trapezoid"`` only: LOS half-extent in units of
            the scale height at ``rmax_AU`` (default 6.0). The other rules
            size their window from the local scale height and ``gamma``.
        quadrature: Line-of-sight rule (static): ``"legendre"`` (default),
            ``"gauss"``, ``"gauss_half"`` or ``"trapezoid"``.
        fixed_vertical_profile: Static. True declares ``gamma = 2`` and
            ``beta = 1`` fixed: the density is evaluated without general
            powers (1.5 to 1.8 times faster, same numbers to rounding) and
            the Gauss rules are built at trace time, so they also work with
            32-bit floats. ``gamma`` and ``beta`` must hold 2 and 1 and are
            no longer fittable. Leave False (default) to fit either.
    """

    sma_AU: Array
    alpha_in: Array
    alpha_out: Array
    ksi0_AU: Array
    gamma: Array
    beta: Array
    rmin_AU: Array
    rmax_AU: Array
    wavelengths_nm: Array
    g_HG_grid: Array
    Ag_grid: Array

    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    pixel_scale_arcsec: float = eqx.field(static=True)
    n_slices_los: int = eqx.field(static=True)
    n_scale_heights: float = eqx.field(static=True, default=6.0)

    cutoff_width_AU: Array = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    quadrature: str = eqx.field(static=True, default="legendre")
    fixed_vertical_profile: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        """Validate the static LOS settings and the fixed-profile promise."""
        check_quadrature(self.quadrature, self.n_slices_los)
        if not self.fixed_vertical_profile:
            return
        try:
            mismatch = bool(jnp.any((self.gamma != 2.0) | (self.beta != 1.0)))
        except jax.errors.TracerBoolConversionError:
            return
        if mismatch:
            raise ValueError(
                "fixed_vertical_profile=True needs gamma = 2 and beta = 1, got "
                f"gamma={self.gamma}, beta={self.beta}."
            )

    def gauss_rule_eps(self, incl_deg: Array) -> Array:
        """Expansion parameter of the ``"gauss"`` rules at ``incl_deg``.

        ``eps = max(|alpha_in|, |alpha_out|) (ksi0_AU / sma_AU) |tan(i)|``:
        the change in the logarithm of the radial power law across one
        scale height of the layer along the line of sight. ``"gauss"`` and
        ``"gauss_half"`` hold about 1e-3 of the peak or better while this
        stays below about 0.45; evaluate it at the corners of a prior before
        choosing them. Nothing checks it at render time.
        """
        log_slope = jnp.maximum(jnp.abs(self.alpha_in), jnp.abs(self.alpha_out))
        return layer_expansion_parameter(
            log_slope, self.ksi0_AU / self.sma_AU, incl_deg
        )

    def _zmax_AU(self) -> Array:
        """LOS half-extent in disk-frame z, evaluated at ``rmax_AU``.

        For a flared disk, the scale height grows with radius, so the
        LOS depth that captures the vertical tails at the outer edge
        is larger than ``n_scale_heights * ksi0_AU`` at ``sma_AU``.
        """
        return (
            self.n_scale_heights
            * self.ksi0_AU
            * (self.rmax_AU / self.sma_AU) ** self.beta
        )

    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_d: Array,
        incl_deg: Array,
        pa_deg: Array,
        *,
        dist_pc: Array,
    ) -> Array:
        """Render the Augereau 1999 scattered-light disk on a sky-plane grid.

        ``time_d`` is days from the scene epoch.

        ``wavelength_nm`` selects ``g_HG`` and ``Ag`` via cubic-spline
        interpolation over ``wavelengths_nm`` / ``g_HG_grid`` /
        ``Ag_grid``. Queries outside the grid return NaN.
        ``time_d`` is part of the AbstractDisk interface but ignored
        (static disk).
        """
        # Wavelength-dependent scattering coefficients.
        g_HG = interpax.interp1d(
            wavelength_nm,
            self.wavelengths_nm,
            self.g_HG_grid,
            method="cubic",
            extrap=False,
        )
        Ag = interpax.interp1d(
            wavelength_nm,
            self.wavelengths_nm,
            self.Ag_grid,
            method="cubic",
            extrap=False,
        )

        fixed = self.fixed_vertical_profile
        cutoff_width = eqx.error_if(
            jnp.asarray(self.cutoff_width_AU),
            self.cutoff_width_AU < 0,
            "cutoff_width_AU must be nonnegative.",
        )

        # Density closure: substitute sma_AU outside the valid annulus
        # so the (-2 alpha_in) power stays bounded in float32; the kernel
        # masks these substituted contributions back to zero.
        def density_fn(r_AU: Array, z_AU: Array, valid: Array) -> Array:
            r_safe = jnp.where(valid, r_AU, self.sma_AU)
            z_safe = jnp.where(valid, z_AU, jnp.zeros_like(z_AU))
            density = two_power_law_density(
                r_safe,
                z_safe,
                self.sma_AU,
                self.alpha_in,
                self.alpha_out,
                self.ksi0_AU,
                self.gamma,
                self.beta,
                fixed_vertical_profile=fixed,
            )
            width = jnp.where(cutoff_width > 0, cutoff_width, 1.0)
            inner = jnp.clip((r_safe - self.rmin_AU) / width, 0.0, 1.0)
            outer = jnp.clip((self.rmax_AU - r_safe) / width, 0.0, 1.0)
            inner = inner**3 * (10.0 + inner * (-15.0 + 6.0 * inner))
            outer = outer**3 * (10.0 + outer * (-15.0 + 6.0 * outer))
            taper = jnp.where(cutoff_width > 0, inner * outer, 1.0)
            return density * taper

        def phase_fn(cos_phi: Array) -> Array:
            return henyey_greenstein(cos_phi, g_HG)

        def scale_height_fn(r_AU: Array) -> Array:
            r_ratio = r_AU / self.sma_AU
            return self.ksi0_AU * (r_ratio if fixed else r_ratio**self.beta)

        integral = los_integrate_scattered(
            density_fn,
            phase_fn,
            incl_deg=incl_deg,
            pa_deg=pa_deg,
            rmin_AU=self.rmin_AU,
            rmax_AU=self.rmax_AU,
            nx=self.nx,
            ny=self.ny,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            dist_pc=dist_pc,
            n_slices_los=self.n_slices_los,
            quadrature=self.quadrature,
            scale_height_fn=scale_height_fn,
            vertical_exponent=2.0 if fixed else self.gamma,
            zmax_AU=self._zmax_AU() if self.quadrature == "trapezoid" else None,
        )
        return Ag * integral

    def spatial_extent(self) -> tuple[float, float]:
        """Return ``(width_arcsec, height_arcsec)``."""
        return (
            self.nx * self.pixel_scale_arcsec,
            self.ny * self.pixel_scale_arcsec,
        )

    def __repr__(self) -> str:
        """Compact summary of radial/vertical/HG parameters + image/Ag grid."""
        n_wl = int(self.wavelengths_nm.shape[0])
        wl_min = float(self.wavelengths_nm.min())
        wl_max = float(self.wavelengths_nm.max())
        return (
            f"GraterDisk(sma={fmt_scalar_or_array(self.sma_AU)} AU, "
            f"radial: alpha_in={fmt_scalar_or_array(self.alpha_in)}, "
            f"alpha_out={fmt_scalar_or_array(self.alpha_out)}, "
            f"r=[{fmt_scalar_or_array(self.rmin_AU)}, "
            f"{fmt_scalar_or_array(self.rmax_AU)}] AU; "
            f"vertical: ksi0={fmt_scalar_or_array(self.ksi0_AU)} AU, "
            f"beta={fmt_scalar_or_array(self.beta)}, "
            f"gamma={fmt_scalar_or_array(self.gamma)}; "
            f"HG/Ag grid: {n_wl} pts in {wl_min:.0f}-{wl_max:.0f} nm; "
            f"image: {self.ny}x{self.nx} @ {self.pixel_scale_arcsec} arcsec/px, "
            f"n_slices_los={self.n_slices_los}, "
            f"quadrature={self.quadrature!r})"
        )
