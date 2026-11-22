"""ExovistaParametricDisk: Simplified JAX forward model of ExoVista's disk model.

Reproduces the per-component disk-density formula from ExoVista's C++
``Image::disk_imager`` (the kernel underlying ``distribute_diskpoints``)
deterministically using the LOS-around-midplane kernel shared with
``GraterDisk``. Multi-component scenes (warm + cold + halo, etc.) are
built via ``CompositeDisk``.

What's reproduced (Stark 2022 / ``Image.cpp``):

  - Vertical Gaussian with constant opening angle ``hor`` and per-radius
    normalization (mass per unit r preserved).
  - Four-region radial profile:

    - ``r > r0+dr``: ``exp(-0.5) * (r/(r0+dr))^-1.5`` (outer halo).
    - ``|r-r0| < dr``: ``exp(-0.5 (r-r0)^2 / dr^2)`` (Gaussian ring).
    - ``r0-dr > r > 0``: inward decay from Wyatt 1999, eq 3.36,
      ``exp(-0.5) / (1 + 4 eta (1 - sqrt(r/(r0-dr))))``.
    - ``r <= rinner``: additional ``(r/rinner)^3`` cubic ramp.

  - Three-component Henyey-Greenstein phase function
    ``sum_i(w_i * HG(g_i))``, with scalar ``(g_i, w_i)`` parameters.
  - Typed amplitude calibration of the declared unit density profile.
  - 1/r^2 stellar illumination dilution (applied by the LOS kernel via
    division by the squared distance to the star).

What's intentionally simplified vs the full ExoVista pipeline:

  - No Dohnanyi grain-size integration weighted by ``Qsca(s, lambda)``.
    Wavelength dependence is parameterized phenomenologically through
    ``Ag_grid`` instead.
  - No sublimation cutoff ``T < T_sublimate`` (would need stellar Teff
    and rstar).
  - No stellar-surface-radius cutoff ``r > r_star``.
  - ``eta`` is treated as a single grain-size-averaged value, not a
    per-size quantity (the Image.cpp ``tempeta = eta * rdust/rblow``
    produces a color gradient absent here).

The class is intentionally faithful to the ExoVista parameterization
(``r0_AU``, ``dror``, ``rinner_AU``, ``hor``, ``eta``,
three HG ``(g_i, w_i)``) rather than a smaller user-friendly subset.
``CompositeDisk`` is the right way to build full multi-component scenes.

References:
    Stark, C. C. 2022, AJ, 163, 105
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import interpax
import jax.numpy as jnp
from jaxtyping import Array
from zodi import AbstractDustAmplitude

from ._los import (
    los_integrate_scattered,
)
from ._parametric import AbstractParametricDisk
from .grater import henyey_greenstein


def three_component_hg(
    cos_phi: Array,
    g0: Array,
    g1: Array,
    g2: Array,
    w0: Array,
    w1: Array,
    w2: Array,
) -> Array:
    """Three-component Henyey-Greenstein phase function.

    ``pfunc = w0 HG(g0) + w1 HG(g1) + w2 HG(g2)``. The caller is responsible
    for choosing weights that sum to 1 (we do not normalize here).

    Args:
        cos_phi: Cosine of the scattering angle, any shape.
        g0: Asymmetry parameter of the first component, in (-1, 1).
        g1: Asymmetry parameter of the second component, in (-1, 1).
        g2: Asymmetry parameter of the third component, in (-1, 1).
        w0: Weight of the first component.
        w1: Weight of the second component.
        w2: Weight of the third component.

    Returns:
        Phase function value per steradian, broadcast shape of the inputs.
    """
    return (
        w0 * henyey_greenstein(cos_phi, g0)
        + w1 * henyey_greenstein(cos_phi, g1)
        + w2 * henyey_greenstein(cos_phi, g2)
    )


def exovista_density(
    r: Array,
    z: Array,
    *,
    r0_AU: Array,
    dror: Array,
    rinner_AU: Array,
    hor: Array,
    density_multiplier: Array,
    eta: Array,
) -> Array:
    """ExoVista per-component disk density (Image.cpp formula).

    Args:
        r: Disk-frame cylindrical radius [AU]. Must be strictly positive.
        z: Disk-frame height [AU].
        r0_AU: Ring center [AU].
        dror: Fractional ring width, ``dr = dror * r0_AU``.
        rinner_AU: Inner truncation radius [AU]. Below this, an additional
            ``(r/rinner)^3`` cubic ramp suppresses the density.
        hor: Opening angle. The local 1-sigma Gaussian vertical scale at
            radius ``r`` is ``h = r * hor``.
        density_multiplier: Dimensionless multiplier of this density profile.
        eta: Poynting-Robertson drag / collision-timescale ratio for the
            inward-decay component.

    Returns:
        Per-LOS-sample density (relative units), same shape as ``r`` and
        ``z``. The caller multiplies by phase function and divides by
        squared distance to the star.
    """
    dr_AU = dror * r0_AU
    sqrt_two_pi = jnp.sqrt(2.0 * jnp.pi)

    # Vertical Gaussian with per-radius normalization so the column
    # (integrated over z) preserves the radial profile shape.
    h_local = hor * r
    vertical = jnp.exp(-0.5 * (z / h_local) ** 2)
    vertical = vertical / (h_local * sqrt_two_pi)

    n = density_multiplier * vertical

    # Piecewise radial profile. The ring owns its two edges, where it equals
    # the neighboring branch; a strict comparison would leave r = r0 +- dr in
    # no branch and return the fallback value of one there.
    in_ring = jnp.abs(r - r0_AU) <= dr_AU
    outer_halo = r > r0_AU + dr_AU
    inner_pr_drag = r < r0_AU - dr_AU

    ring_factor = jnp.exp(-0.5 * ((r - r0_AU) / dr_AU) ** 2)
    halo_factor = jnp.exp(-0.5) * (r / (r0_AU + dr_AU)) ** -1.5

    # inward-decay component. Guard the sqrt argument
    # against negative values (only relevant just outside r0-dr where
    # the boolean is False, but the formula is still evaluated).
    sqrt_arg = jnp.maximum(r / (r0_AU - dr_AU), 0.0)
    pr_drag_factor = jnp.exp(-0.5) / (1.0 + 4.0 * eta * (1.0 - jnp.sqrt(sqrt_arg)))

    radial = jnp.where(
        in_ring,
        ring_factor,
        jnp.where(
            outer_halo,
            halo_factor,
            jnp.where(inner_pr_drag, pr_drag_factor, 1.0),
        ),
    )
    n = n * radial

    # Cubic inner-truncation ramp (multiplicative, applies on top of the
    # piecewise radial profile).
    inside_rinner = r <= rinner_AU
    inner_ramp = (r / rinner_AU) ** 3
    n = jnp.where(inside_rinner, n * inner_ramp, n)

    return n


class ExovistaParametricDisk(AbstractParametricDisk):
    """ExoVista-style disk component (faithful per-component reproduction).

    Density, scattering, and spectral parameters are fittable PyTree leaves.
    LOS integration settings are static; the caller supplies the sky grid.

    The vertical profile ``exp(-0.5 (z / (hor r))^2)`` is the form
    ``exp(-(|z| / h)^gamma)`` with ``gamma = 2`` and ``h = sqrt(2) hor r``,
    which sets the Gauss-Legendre window. It covers 4 local scale heights
    at every radius, bounded by the ``rmax_AU`` cylinder, including the
    outer halo. Radial slope breaks at ``r0 -+ dr`` and ``rinner_AU`` make
    convergence algebraic rather than spectral.

    Attributes:
        r0_AU: Ring center radius [AU].
        dror: Ring fractional width ``Delta r / r0`` (dimensionless).
        rinner_AU: Inner truncation radius [AU]; below this the density
            is suppressed by ``(r/rinner_AU)^3``. Set to a value smaller
            than ``rmin_AU`` to disable.
        hor: Opening angle (Gaussian vertical scale / radius).
        eta: PR-drag / collision-timescale ratio at ``r0 - dr``.
        g0: Asymmetry parameter of the first HG component.
        g1: Asymmetry parameter of the second HG component.
        g2: Asymmetry parameter of the third HG component.
        w0: Weight of the first HG component. The three weights should sum
            to 1; this is not enforced.
        w1: Weight of the second HG component.
        w2: Weight of the third HG component.
        rmin_AU: LOS-integration inner bound [AU].
        rmax_AU: LOS-integration outer bound [AU].
        wavelengths_nm: 1-D wavelength grid for ``Ag_grid``. Queries
            outside this range return NaN.
        Ag_grid: Phenomenological albedo scaling vs wavelength.
        profile_id: Class-level profile identifier, ``"exovista-parametric"``.
        amplitude: Typed density or calibrated surface-brightness amplitude.
        n_slices_los: Number of Gauss-Legendre LOS nodes.
        inclination_deg: Inclination, zero face-on.
        position_angle_deg: Major axis north through east.
        flux_scale: Dimensionless factor on the radiance, default one; see
            :class:`~skyscapes.disk.AbstractDisk`. Exactly degenerate with
            ``amplitude.level``.
    """

    r0_AU: Array
    dror: Array
    rinner_AU: Array
    hor: Array
    eta: Array
    g0: Array
    g1: Array
    g2: Array
    w0: Array
    w1: Array
    w2: Array
    rmin_AU: Array
    rmax_AU: Array
    wavelengths_nm: Array
    Ag_grid: Array

    profile_id: ClassVar[str] = "exovista-parametric"
    amplitude: AbstractDustAmplitude
    n_slices_los: int = eqx.field(static=True)
    inclination_deg: Array = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    position_angle_deg: Array = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)

    def __check_init__(self):
        """Validate the static LOS settings."""
        if self.n_slices_los < 1:
            raise ValueError("n_slices_los must be positive.")

    def _scattered_integral(
        self, wavelength_nm, east_AU, north_AU, inclination_deg, position_angle_deg
    ):
        """Unit-density LOS integral times the spectral scattering coefficient."""
        Ag = interpax.interp1d(
            wavelength_nm,
            self.wavelengths_nm,
            self.Ag_grid,
            method="cubic",
            extrap=False,
        )

        def density_fn(r_AU: Array, z_AU: Array, valid: Array) -> Array:
            # Substitute r0_AU outside the annulus so the radial-profile
            # transitions evaluate at a well-conditioned radius. The
            # kernel masks these contributions back to zero.
            r_safe = jnp.where(valid, r_AU, self.r0_AU)
            z_safe = jnp.where(valid, z_AU, jnp.zeros_like(z_AU))
            return exovista_density(
                r_safe,
                z_safe,
                r0_AU=self.r0_AU,
                dror=self.dror,
                rinner_AU=self.rinner_AU,
                hor=self.hor,
                density_multiplier=1.0,
                eta=self.eta,
            )

        def phase_fn(cos_phi: Array) -> Array:
            return three_component_hg(
                cos_phi,
                self.g0,
                self.g1,
                self.g2,
                self.w0,
                self.w1,
                self.w2,
            )

        def scale_height_fn(r_AU: Array) -> Array:
            return jnp.sqrt(2.0) * self.hor * r_AU

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
            vertical_exponent=2.0,
        )
        return Ag * integral
