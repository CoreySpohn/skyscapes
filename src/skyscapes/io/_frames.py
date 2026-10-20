"""Frame conversions for ExoVista-style barycentric -> sky-plane rotations.

The proper rotation Rz(pa) Rx(-inc) maps into the right-handed
(east, north, away) frame. It uses the rotations of
``exoverses.util.misc.gen_rotate_to_sky_coords`` (convention ``"exovista"``)
without its final z flip: our z is the negative of that function's
observer-pointing z. Inputs are JAX-friendly degrees-in-floats.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def _rot_x(angle_rad: float) -> Array:
    """3x3 rotation matrix around the X axis."""
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    return jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def _rot_z(angle_rad: float) -> Array:
    """3x3 rotation matrix around the Z axis."""
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    return jnp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotate_to_sky_coords(
    vectors: Array,
    inc_deg: float,
    pa_deg: float,
) -> Array:
    """Rotate ``Nx3`` barycentric vectors into the sky frame.

    The output is right-handed (east, north, away). Apply:

    1. Rotate by ``-inc_deg`` around the X axis.
    2. Rotate by ``+pa_deg`` around the Z axis.

    Args:
        vectors: ``(N, 3)`` array of barycentric position or velocity
            vectors.
        inc_deg: System midplane inclination in degrees.
        pa_deg: System midplane position angle in degrees.

    Returns:
        ``(N, 3)`` array of vectors expressed in (east, north, away).
    """
    inc_rad = jnp.deg2rad(inc_deg)
    pa_rad = jnp.deg2rad(pa_deg)

    # Apply rotations: vectors are row vectors (N, 3); right-multiply by R^T,
    # i.e. v_out = v_in @ R^T == (R @ v_in^T)^T.
    rotated = vectors @ _rot_x(-inc_rad).T
    rotated = rotated @ _rot_z(pa_rad).T

    return rotated
