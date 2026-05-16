"""Frame conversions for ExoVista-style barycentric -> sky-plane rotations.

Ports ``exoverses.util.misc.gen_rotate_to_sky_coords`` (convention
``"exovista"``) to JAX-friendly degrees-in-floats inputs. The math is
identical: rotate by ``-inclination`` around X, then by ``+position_angle``
around Z, then flip the Z sign so that the resulting frame has +Z toward
the observer.
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

    Matches ``exoverses.util.misc.gen_rotate_to_sky_coords`` with
    ``convention="exovista"``:

    1. Rotate by ``-inc_deg`` around the X axis.
    2. Rotate by ``+pa_deg`` around the Z axis.
    3. Flip the Z component sign.

    Args:
        vectors: ``(N, 3)`` array of barycentric position or velocity
            vectors.
        inc_deg: System midplane inclination in degrees.
        pa_deg: System midplane position angle in degrees.

    Returns:
        ``(N, 3)`` array of vectors expressed in the sky frame.
    """
    inc_rad = jnp.deg2rad(inc_deg)
    pa_rad = jnp.deg2rad(pa_deg)

    # Apply rotations: vectors are row vectors (N, 3); right-multiply by R^T,
    # i.e. v_out = v_in @ R^T == (R @ v_in^T)^T.
    rotated = vectors @ _rot_x(-inc_rad).T
    rotated = rotated @ _rot_z(pa_rad).T

    # Flip the Z axis (matches exoverses' final ``vector[:, 2] = -vector[:, 2]``).
    rotated = rotated.at[:, 2].multiply(-1.0)

    return rotated
