"""Frame geometry shared by the views: pure NumPy, no plotting imports.

Every direction here comes from the library's own frame conventions rather
than a second derivation of them:

- The sky frame is ``(x, y, z)``: ``x`` the RA offset, ``y`` the Dec
  offset, ``+z`` from the star toward the observer. ``Planet`` positions
  (through orbix, whose phase angle is measured from the ``+z`` observer
  axis) and the parametric disk kernels both use it.
- The disk midplane is placed in the sky frame by
  ``skyscapes.io._frames.rotate_to_sky_coords``, the rotation the ExoVista
  loader applies. Its first two output rows are the images of the disk's
  in-plane axes: the line of nodes, at angle ``pa`` from ``+x`` toward
  ``+y``, and the in-plane axis whose ``+z`` component is ``sin(incl)``,
  which points at the near (forward-scattering) half of the disk. These
  are the same axes the line-of-sight kernel integrates in.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from skyscapes.io._frames import rotate_to_sky_coords


def disk_axes_sky(incl_deg, pa_deg):
    """Sky-frame unit vectors of the disk's axes.

    Args:
        incl_deg: Midplane inclination [deg]; 0 is face-on.
        pa_deg: Midplane position angle [deg].

    Returns:
        ``(nodes, near, normal)``, each shape ``(3,)``: the line of nodes
        (the projected major axis), the in-plane axis toward the near half,
        and the right-handed midplane normal ``nodes x near``.
    """
    basis = np.asarray(
        rotate_to_sky_coords(jnp.eye(3), inc_deg=float(incl_deg), pa_deg=float(pa_deg)),
        dtype=float,
    )
    nodes, near = basis[0], basis[1]
    return nodes, near, np.cross(nodes, near)


def minor_axis_direction(pa_deg):
    """Unit sky ``(x, y)`` direction of the projected minor axis.

    This is the line of nodes turned by +90 degrees in the sky plane, fixed
    for a given ``pa_deg`` whatever the inclination, so a side view built
    on it does not flip while an inclination is swept.

    Args:
        pa_deg: Midplane position angle [deg].

    Returns:
        Shape ``(2,)`` unit vector.
    """
    nodes, _, _ = disk_axes_sky(0.0, pa_deg)
    return np.array([-nodes[1], nodes[0]])


def to_side(points_xyz, minor_xy):
    """Project sky-frame points onto the side-view plane.

    The side-view plane holds the line of sight and the projected minor
    axis; the line of nodes is perpendicular to it.

    Args:
        points_xyz: Sky-frame points, shape ``(..., 3)``.
        minor_xy: Projected minor-axis direction from ``minor_axis_direction``.

    Returns:
        ``(z, u)``: distance toward the observer and offset along the
        projected minor axis, each shape ``(...)``.
    """
    p = np.asarray(points_xyz, dtype=float)
    return p[..., 2], p[..., 0] * minor_xy[0] + p[..., 1] * minor_xy[1]


def ring_sky(radius, incl_deg, pa_deg, n=361):
    """Sky-frame points of a circle of ``radius`` in the disk midplane.

    Args:
        radius: Ring radius, in any length unit (the output shares it).
        incl_deg: Midplane inclination [deg].
        pa_deg: Midplane position angle [deg].
        n: Number of points; the first and last coincide.

    Returns:
        Shape ``(n, 3)``.
    """
    nodes, near, _ = disk_axes_sky(incl_deg, pa_deg)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return radius * (np.cos(t)[:, None] * nodes + np.sin(t)[:, None] * near)


def disk_radii_AU(disk):
    """Inner and outer truncation radii of a disk, when it declares them.

    Parametric disks (``GraterDisk``, ``ExovistaParametricDisk``) carry
    ``rmin_AU`` / ``rmax_AU``; a ``CompositeDisk`` spans its components. A
    pre-rendered disk (``ExovistaDisk``) has no radii, and gives None.

    Args:
        disk: An ``AbstractDisk`` or None.

    Returns:
        ``(r_in_AU, r_out_AU)`` floats, or None.
    """
    if disk is None:
        return None
    components = getattr(disk, "components", None)
    if components is not None:
        radii = [r for r in (disk_radii_AU(c) for c in components) if r is not None]
        if not radii:
            return None
        return min(r[0] for r in radii), max(r[1] for r in radii)
    if hasattr(disk, "rmin_AU") and hasattr(disk, "rmax_AU"):
        return float(disk.rmin_AU), float(disk.rmax_AU)
    return None


def zodi_sightline(ecliptic_lat_deg, solar_lon_deg):
    """Look direction for the two Leinert inputs, observer-centered.

    The frame is ecliptic with ``+x`` pointing from the Sun to the
    observer, so the Sun lies along ``-x`` from the observer, and ``+z`` is
    ecliptic north. The helio-ecliptic longitude difference is taken toward
    ``+y``; the Leinert table is symmetric in its sign.

    Args:
        ecliptic_lat_deg: Ecliptic latitude of the look direction [deg].
        solar_lon_deg: Helio-ecliptic longitude difference between the look
            direction and the Sun [deg], 0 toward the Sun.

    Returns:
        Unit look vector, shape ``(3,)``.
    """
    beta = np.radians(ecliptic_lat_deg)
    dlon = np.radians(solar_lon_deg)
    return np.array(
        [
            -np.cos(beta) * np.cos(dlon),
            np.cos(beta) * np.sin(dlon),
            np.sin(beta),
        ]
    )
