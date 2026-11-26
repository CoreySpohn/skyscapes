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

Notes on quadrature:
    Gauss-Legendre integrates over the exact extent of the flared layer
    along each line of sight and is smooth through edge-on, so exactly
    edge-on images and inclination derivatives are supported. Smooth radial
    tapers remove hard sample-crossing jumps.
    Inclinations above 90 view the disk from its other face and return the
    physical, nonnegative map, mirrored across the line of nodes relative to
    ``180 - incl``.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from jaxtyping import Array
from optixstuff.spectral import Knots, as_knots
from zodi import AbstractDustAmplitude

from ._los import (
    los_integrate_scattered,
)
from ._parametric import AbstractParametricDisk


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


class GraterDisk(AbstractParametricDisk):
    """Augereau 1999 scattered-light disk.

    Density, scattering, and spectral parameters are fittable PyTree leaves.
    LOS integration settings are static; the caller supplies the sky grid.

    **Choosing the line-of-sight rule.** All rules converge to the same
    integral; the choice sets the truncation error and the cost, which is
    proportional to the node count. Errors below are the largest pixel
    error over the image peak for a narrow ring (``alpha = 5``,
    ``h / r = 0.05``) with tapered edges.

    Use 24 to 48 Gauss-Legendre nodes up to 70 degrees (below 1e-4;
    48 nodes reach 1e-5 at 60 degrees and 6e-5 at 70) and 96 from 80
    to 90 degrees (4e-4 to 2e-3). A broader or thicker disk needs fewer.

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
        wavelengths_nm: 1-D wavelength grid for ``g_HG_grid`` /
            ``Ag_grid``, shape ``(n_wave,)`` [nm]. Must be sorted.
            ``photon_radiance`` calls outside this range return NaN.
        g_HG_grid: Henyey-Greenstein asymmetry at each grid wavelength.
        Ag_grid: Albedo-like scaling at each grid wavelength.
        profile_id: Class-level profile identifier, ``"grater-1999"``.
        amplitude: Typed density or calibrated surface-brightness amplitude.
        n_slices_los: Number of Gauss-Legendre LOS nodes.
        flux_scale: Dimensionless factor on the radiance, default one; see
            :class:`~skyscapes.disk.AbstractDisk`. Exactly degenerate with
            ``amplitude.level``.
        cutoff_width_AU: Nonnegative inward radial taper width [AU].
            Zero (default) preserves the hard rmin/rmax mask exactly.
            Positive values use a C2 quintic ramp from zero at the boundary
            to one a width inside the annulus. Widths larger than half the
            annulus overlap.
        fixed_vertical_profile: Static. True declares ``gamma = 2`` and
            ``beta = 1`` fixed: the density is evaluated without general
            powers (1.5 to 1.8 times faster, same numbers to rounding).
            ``gamma`` and ``beta`` must hold 2 and 1 and are
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
    wavelengths_nm: Knots = eqx.field(static=True, converter=as_knots)
    g_HG_grid: Array
    Ag_grid: Array

    profile_id: ClassVar[str] = "grater-1999"
    amplitude: AbstractDustAmplitude
    n_slices_los: int = eqx.field(static=True)
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)

    cutoff_width_AU: Array = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    fixed_vertical_profile: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        """Validate the static LOS settings and the fixed-profile promise.

        Returns:
            None.
        """
        if self.n_slices_los < 1:
            raise ValueError("n_slices_los must be positive.")
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

    def _scattered_integral(
        self,
        wavelength_nm,
        east_AU,
        north_AU,
        inclination_deg,
        position_angle_deg,
        *,
        sky_offsets_arcsec=None,
    ):
        """Unit-density LOS integral times the spectral scattering coefficient."""
        # Wavelength-dependent scattering coefficients.
        knots = self.wavelengths_nm.array(
            jnp.result_type(wavelength_nm, self.Ag_grid, self.g_HG_grid)
        )
        g_HG = interpax.interp1d(
            wavelength_nm,
            knots,
            self.g_HG_grid,
            method="cubic",
            extrap=jnp.asarray(jnp.nan, dtype=knots.dtype),
        )
        Ag = interpax.interp1d(
            wavelength_nm,
            knots,
            self.Ag_grid,
            method="cubic",
            extrap=jnp.asarray(jnp.nan, dtype=knots.dtype),
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
            width = jnp.where(
                cutoff_width > 0, cutoff_width, jnp.ones_like(cutoff_width)
            )
            zero, one = jnp.zeros_like(width), jnp.ones_like(width)
            inner = jnp.clip((r_safe - self.rmin_AU) / width, zero, one)
            outer = jnp.clip((self.rmax_AU - r_safe) / width, zero, one)
            inner = inner**3 * (10.0 + inner * (-15.0 + 6.0 * inner))
            outer = outer**3 * (10.0 + outer * (-15.0 + 6.0 * outer))
            taper = jnp.where(cutoff_width > 0, inner * outer, jnp.ones_like(inner))
            return density * taper

        def phase_fn(cos_phi: Array) -> Array:
            return henyey_greenstein(cos_phi, g_HG)

        def scale_height_fn(r_AU, sma, ksi0, beta):
            ratio = r_AU / sma
            if fixed:
                return ksi0 * ratio, ksi0 / sma
            return ksi0 * ratio**beta, (ksi0 * beta / sma) * ratio ** (beta - 1)

        integral = los_integrate_scattered(
            density_fn,
            phase_fn,
            incl_deg=inclination_deg,
            position_angle_deg=position_angle_deg,
            rmin_AU=self.rmin_AU,
            rmax_AU=self.rmax_AU,
            east_AU=east_AU,
            north_AU=north_AU,
            n_slices_los=self.n_slices_los,
            scale_height_fn=scale_height_fn,
            height_args=(self.sma_AU, self.ksi0_AU, self.beta),
            sky_offsets_arcsec=sky_offsets_arcsec,
            vertical_exponent=2.0 if fixed else self.gamma,
        )
        return Ag * integral
