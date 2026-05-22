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
        r > r0+dr:  ``exp(-0.5) * (r/(r0+dr))^-1.5``    (outer halo)
        |r-r0|<dr:  ``exp(-0.5 (r-r0)^2 / dr^2)``       (Gaussian ring)
        r0-dr>r>0:  inward decay from Wyatt 1999, eq 3.36,
                    ``exp(-0.5) / (1 + 4 eta (1 - sqrt(r/(r0-dr))))``
        r <= rinner: additional ``(r/rinner)^3`` cubic ramp
  - Three-component Henyey-Greenstein phase function
    ``sum_i(w_i * HG(g_i))``, with scalar ``(g_i, w_i)`` parameters.
  - ``nzodis`` density scaling (zodi units).
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
(``r0_AU``, ``dror``, ``rinner_AU``, ``hor``, ``nzodis``, ``eta``,
three HG ``(g_i, w_i)``) rather than a smaller user-friendly subset.
``CompositeDisk`` is the right way to build full multi-component scenes.

References:
    Stark, C. C. 2022, AJ, 163, 105
"""

from __future__ import annotations

import equinox as eqx
import interpax
import jax.numpy as jnp
from jaxtyping import Array

from .._repr import fmt_scalar_or_array
from ._los import los_integrate_scattered
from .base import AbstractDisk
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
    nzodis: Array,
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
        nzodis: Density normalization (zodi units).
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

    n = nzodis * vertical

    # Piecewise radial profile.
    in_ring = jnp.abs(r - r0_AU) < dr_AU
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


class ExovistaParametricDisk(AbstractDisk):
    """ExoVista-style disk component (faithful per-component reproduction).

    Attributes (PyTree leaves, fittable):
        r0_AU: Ring center radius [AU].
        dror: Ring fractional width ``Delta r / r0`` (dimensionless).
        rinner_AU: Inner truncation radius [AU]; below this the density
            is suppressed by ``(r/rinner_AU)^3``. Set to a value smaller
            than ``rmin_AU`` to disable.
        hor: Opening angle (Gaussian vertical scale / radius).
        nzodis: Density normalization in zodi units.
        eta: PR-drag / collision-timescale ratio at ``r0 - dr``.
        g0, g1, g2: Three-component HG asymmetry parameters.
        w0, w1, w2: Three-component HG weights. Should sum to 1; not
            enforced.
        rmin_AU: LOS-integration inner bound [AU].
        rmax_AU: LOS-integration outer bound [AU].
        wavelengths_nm: 1-D wavelength grid for ``Ag_grid``. Queries
            outside this range return NaN.
        Ag_grid: Phenomenological albedo scaling vs wavelength.

    Static attributes:
        nx, ny: Output image shape.
        pixel_scale_arcsec: Pixel scale [arcsec/pixel].
        dist_pc: System distance [pc].
        n_slices_los: Number of LOS integration slices.
        n_scale_heights: LOS half-extent in units of the scale height at
            ``rmax_AU`` (default 6.0).
    """

    r0_AU: Array
    dror: Array
    rinner_AU: Array
    hor: Array
    nzodis: Array
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

    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    pixel_scale_arcsec: float = eqx.field(static=True)
    dist_pc: float = eqx.field(static=True)
    n_slices_los: int = eqx.field(static=True)
    n_scale_heights: float = eqx.field(static=True, default=6.0)

    def _zmax_AU(self) -> Array:
        """LOS half-extent in disk-frame z, evaluated at the ring center.

        For ExoVista-style disks the mass is concentrated near ``r0_AU``
        (ring + nearby halo + inward profile), and the radial profile
        decays as ``r^-1.5`` outside. Using ``r0_AU`` rather than the
        LOS-bound ``rmax_AU`` gives an LOS extent matched to where the
        density actually lives, so per-slice dz scales with the scale
        height at the ring rather than with the (much larger) LOS-bound.
        """
        return self.n_scale_heights * self.r0_AU * self.hor

    def surface_brightness(
        self,
        wavelength_nm: Array,
        time_jd: Array,
        incl_deg: Array,
        pa_deg: Array,
    ) -> Array:
        """Render the ExoVista-style disk on a sky-plane grid."""
        Ag = interpax.interp1d(
            wavelength_nm,
            self.wavelengths_nm,
            self.Ag_grid,
            method="cubic",
            extrap=False,
        )

        zmax_AU = self._zmax_AU()
        # ExoVista disks can be intrinsically thick,
        # so the rmax/zmax geometric threshold used
        # in GraterDisk is overly restrictive here. The real numerical
        # problem is only at true edge-on (cos_i -> 0) where l_half
        # diverges. Guard against that and let the LOS-around-midplane
        # parameterization be approximate for puffy disks at moderate
        # inclinations.
        cos_i = jnp.cos(incl_deg * (jnp.pi / 180.0))
        incl_deg = eqx.error_if(
            incl_deg,
            jnp.abs(cos_i) < 1e-2,
            "ExovistaParametricDisk: incl_deg too close to edge-on "
            "(|cos(incl)| < 1e-2). The LOS-around-midplane sampling "
            "diverges; restrict incl_deg to within ~89.4 deg of pole-on.",
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
                nzodis=self.nzodis,
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
        """Compact summary of ring/density/HG parameters + image/Ag grid."""
        n_wl = int(self.wavelengths_nm.shape[0])
        wl_min = float(self.wavelengths_nm.min())
        wl_max = float(self.wavelengths_nm.max())
        return (
            f"ExovistaParametricDisk("
            f"ring: r0={fmt_scalar_or_array(self.r0_AU)} AU, "
            f"dror={fmt_scalar_or_array(self.dror)}; "
            f"PR-drag inner: rinner={fmt_scalar_or_array(self.rinner_AU)} AU; "
            f"density: nzodis={fmt_scalar_or_array(self.nzodis)}, "
            f"eta={fmt_scalar_or_array(self.eta)}, "
            f"hor={fmt_scalar_or_array(self.hor)}; "
            f"3-comp HG: g=[{fmt_scalar_or_array(self.g0)}, "
            f"{fmt_scalar_or_array(self.g1)}, "
            f"{fmt_scalar_or_array(self.g2)}], "
            f"w=[{fmt_scalar_or_array(self.w0)}, "
            f"{fmt_scalar_or_array(self.w1)}, "
            f"{fmt_scalar_or_array(self.w2)}]; "
            f"r=[{fmt_scalar_or_array(self.rmin_AU)}, "
            f"{fmt_scalar_or_array(self.rmax_AU)}] AU; "
            f"Ag grid: {n_wl} pts in {wl_min:.0f}-{wl_max:.0f} nm; "
            f"image: {self.ny}x{self.nx} @ {self.pixel_scale_arcsec} arcsec/px, "
            f"dist={self.dist_pc} pc, n_slices_los={self.n_slices_los})"
        )
