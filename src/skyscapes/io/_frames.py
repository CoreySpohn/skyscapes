"""Frame conversions for ExoVista-style barycentric -> sky-plane rotations.

The proper rotation Rz(pa) Rx(-inc) maps into the right-handed
(east, north, away) frame. It uses the rotations of
``exoverses.util.misc.gen_rotate_to_sky_coords`` (convention ``"exovista"``)
without its final z flip: our z is the negative of that function's
observer-pointing z. Inputs are JAX-friendly degrees-in-floats.
"""

from __future__ import annotations

import jax.numpy as jnp
from hwoutils.constants import deg2rad
from jaxtyping import Array
from orbix.system.rotation import from_axis_angle, rotate


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
    tilted = rotate(
        from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -inc_deg * deg2rad), vectors
    )
    return rotate(from_axis_angle(jnp.array([0.0, 0.0, 1.0]), pa_deg * deg2rad), tilted)
