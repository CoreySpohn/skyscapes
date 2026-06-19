"""GraterDisk: Augereau 1999 scattered-light disk as an eqx.Module.

The Augereau et al. 1999 model is a two-power-law dust density distribution
with a flared vertical profile, scattering with a single Henyey-Greenstein
phase function. This module exposes the model in skyscapes' eqx.Module
idiom so its parameters are PyTree leaves (fittable through the standard
scene-as-PyTree pattern).

References:
    Augereau, J.-C., Lagrange, A.-M., Mouillet, D., Papaloizou, J. C. B.,
    & Grorod, P. A. 1999, A&A, 348, 557
    grater-jax (Kondapalli, Lewis, Ashcraft, Millar-Blanchaer):
        https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX

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

    The LOS-around-midplane sampling uses log-spaced nodes concentrated at
    the midplane crossing (after the GRaTeR-JAX reference), so the integral
    stays converged at high inclination. It diverges only at the true edge-on
    singularity (``cos(incl) -> 0``); ``surface_brightness`` raises via
    ``eqx.error_if`` when ``|cos(incl)| < 1e-2`` (within ~89.4 deg of pole-on).
"""

from __future__ import annotations

import equinox as eqx
import interpax
import jax.numpy as jnp
from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from ._los import los_integrate_scattered
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
) -> Array:
    """Augereau 1999 two-power-law radial density with flared vertical profile.

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

    Returns:
        Density in arbitrary units (relative), same shape as ``r`` and ``z``.
    """
    r_ratio = r / sma_AU
    radial = (r_ratio ** (-2.0 * alpha_in) + r_ratio ** (-2.0 * alpha_out)) ** (-0.5)
    h_r = ksi0_AU * r_ratio**beta
    vertical = jnp.exp(-((jnp.abs(z) / h_r) ** gamma))
    return radial * vertical


class GraterDisk(AbstractDisk):
    """Augereau 1999 scattered-light disk.

    Attributes (PyTree leaves, fittable):
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
            ``surface_brightness`` calls outside this range return NaN.
        g_HG_grid: Henyey-Greenstein asymmetry at each grid wavelength.
        Ag_grid: Albedo-like scaling at each grid wavelength.

    Static attributes (compilation-time constants):
        nx, ny: Output image shape.
        pixel_scale_arcsec: Pixel scale [arcsec/pixel].
        dist_pc: System distance [pc].
        n_slices_los: Number of LOS integration slices.
        n_scale_heights: LOS half-extent in units of the local scale
            height at ``rmax_AU`` (default 6.0).
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
    dist_pc: float = eqx.field(static=True)
    n_slices_los: int = eqx.field(static=True)
    n_scale_heights: float = eqx.field(static=True, default=6.0)

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
        time_jd: Array,
        incl_deg: Array,
        pa_deg: Array,
    ) -> Array:
        """Render the Augereau 1999 scattered-light disk on a sky-plane grid.

        ``wavelength_nm`` selects ``g_HG`` and ``Ag`` via cubic-spline
        interpolation over ``wavelengths_nm`` / ``g_HG_grid`` /
        ``Ag_grid``. Queries outside the grid return NaN.
        ``time_jd`` is part of the AbstractDisk interface but ignored
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

        # The LOS-around-midplane sampling diverges only at the true edge-on
        # singularity (cos_i -> 0), where the window half-width zmax/cos_i and
        # the midplane crossing y*tan(i) both blow up. The log-spaced LOS nodes
        # keep the integral converged at all inclinations short of that, so the
        # guard fires only near the genuine singularity (the GRaTeR-JAX
        # reference shares the same 1/cos_i divergence with no handling).
        zmax_AU = self._zmax_AU()
        cos_i = jnp.cos(incl_deg * (jnp.pi / 180.0))
        incl_deg = eqx.error_if(
            incl_deg,
            jnp.abs(cos_i) < 1e-2,
            "GraterDisk: incl_deg too close to edge-on (|cos(incl)| < 1e-2). "
            "The LOS-around-midplane sampling diverges as cos(incl) -> 0; "
            "restrict incl_deg to within ~89.4 deg of pole-on.",
        )

        # Density closure: substitute sma_AU outside the valid annulus
        # so the (-2 alpha_in) power stays bounded in float32; the kernel
        # masks these substituted contributions back to zero.
        def density_fn(r_AU: Array, z_AU: Array, valid: Array) -> Array:
            r_safe = jnp.where(valid, r_AU, self.sma_AU)
            z_safe = jnp.where(valid, z_AU, jnp.zeros_like(z_AU))
            return two_power_law_density(
                r_safe,
                z_safe,
                self.sma_AU,
                self.alpha_in,
                self.alpha_out,
                self.ksi0_AU,
                self.gamma,
                self.beta,
            )

        def phase_fn(cos_phi: Array) -> Array:
            return henyey_greenstein(cos_phi, g_HG)

        integral = los_integrate_scattered(
            density_fn,
            phase_fn,
            incl_deg=incl_deg,
            pa_deg=pa_deg,
            rmin_AU=self.rmin_AU,
            rmax_AU=self.rmax_AU,
            zmax_AU=zmax_AU,
            nx=self.nx,
            ny=self.ny,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            dist_pc=self.dist_pc,
            n_slices_los=self.n_slices_los,
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
            f"dist={self.dist_pc} pc, n_slices_los={self.n_slices_los})"
        )
