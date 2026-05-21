"""Shared LOS-around-midplane integration kernel for parametric disks.

Lifted out of ``GraterDisk.surface_brightness`` so multiple disk classes
(GraterDisk, ExovistaParametricDisk, ...) can plug in their own density
profiles and phase functions while sharing the geometric integration.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from hwoutils.constants import deg2rad
from hwoutils.transforms import ccw_rotation_matrix
from jaxtyping import Array

DensityFn = Callable[[Array, Array, Array], Array]
PhaseFn = Callable[[Array], Array]


def los_integrate_scattered(
    density_fn: DensityFn,
    phase_fn: PhaseFn,
    *,
    incl_deg: Array,
    pa_deg: Array,
    rmin_AU: Array,
    rmax_AU: Array,
    zmax_AU: Array,
    nx: int,
    ny: int,
    pixel_scale_arcsec: float,
    dist_pc: float,
    n_slices_los: int,
) -> Array:
    """Integrate scattered light along LOS around the disk midplane.

    Sky frame: ``+x`` right, ``+y`` toward N, observer along ``+z``.
    ``pa_deg`` is measured from N toward the disk's projected major
    axis (CCW). ``incl_deg`` is the angle between the disk normal and
    the observer's line of sight (0 = pole-on).

    Args:
        density_fn: ``(r_AU, z_AU, valid) -> rho``. The kernel ensures
            ``r_AU`` is finite everywhere (sqrt is masked) but the
            caller must substitute density-safe values at invalid
            points (e.g. avoid ``r=0`` for ``r^alpha`` profiles).
        phase_fn: ``cos_phi -> phase`` evaluated at every LOS sample.
        incl_deg: Disk inclination [deg].
        pa_deg: Disk position angle [deg] from N, CCW.
        rmin_AU: Inner truncation radius [AU].
        rmax_AU: Outer truncation radius [AU].
        zmax_AU: LOS half-extent in disk-frame z [AU]. Should comfortably
            exceed the disk's largest scale height (for flared disks,
            evaluate the scale-height formula at ``rmax_AU``).
        nx: Output image width [pixels] (static).
        ny: Output image height [pixels] (static).
        pixel_scale_arcsec: Pixel scale [arcsec/pixel] (static).
        dist_pc: System distance [pc] (static).
        n_slices_los: Number of LOS integration slices (static).

    Returns:
        LOS-integrated scattered-light map, shape ``(ny, nx)``.
        Caller multiplies by an albedo / normalization to get contrast.
    """
    px_AU = pixel_scale_arcsec * dist_pc
    x_pix = (jnp.arange(nx) - (nx - 1) / 2.0) * px_AU
    y_pix = (jnp.arange(ny) - (ny - 1) / 2.0) * px_AU
    x_sky, y_sky = jnp.meshgrid(x_pix, y_pix)

    incl = incl_deg * deg2rad
    cos_i = jnp.cos(incl)
    sin_i = jnp.sin(incl)

    r_pa_inv = ccw_rotation_matrix(-pa_deg)
    x_rot = r_pa_inv[0, 0] * x_sky + r_pa_inv[0, 1] * y_sky
    y_rot = r_pa_inv[1, 0] * x_sky + r_pa_inv[1, 1] * y_sky

    # Per-pixel midplane crossing in LOS depth, then sample a fixed
    # disk-frame z extent around it.
    l_mid = y_rot * sin_i / cos_i
    l_half = zmax_AU / cos_i

    t = jnp.linspace(-1.0, 1.0, n_slices_los)
    l_grid = l_mid[None, :, :] + t[:, None, None] * l_half

    y_d = y_rot[None, :, :] * cos_i + l_grid * sin_i
    z_d = -y_rot[None, :, :] * sin_i + l_grid * cos_i

    xy_sq = x_rot * x_rot + y_d * y_d
    d_star_sq = xy_sq + z_d * z_d
    rmin_sq = rmin_AU * rmin_AU
    rmax_sq = rmax_AU * rmax_AU
    valid = (xy_sq >= rmin_sq) & (xy_sq <= rmax_sq) & (d_star_sq > 0.0)

    # Sqrt only inside the valid annulus; the caller's density_fn handles
    # any further safe substitution before evaluating its formula.
    xy_sq_safe = jnp.where(valid, xy_sq, 1.0)
    r_AU = jnp.sqrt(xy_sq_safe)
    rho = density_fn(r_AU, z_d, valid)

    # Scattering angle: observer-frame z / |position| = l_grid / |position|.
    safe_d_sq = jnp.where(d_star_sq > 0.0, d_star_sq, 1.0)
    safe_d = jnp.sqrt(safe_d_sq)
    cos_phi = l_grid / safe_d
    phase = phase_fn(cos_phi)

    integrand = jnp.where(valid, rho * phase / safe_d_sq, 0.0)
    dl = 2.0 * l_half / (n_slices_los - 1)
    return 0.5 * dl * (integrand[:-1] + integrand[1:]).sum(axis=0)
