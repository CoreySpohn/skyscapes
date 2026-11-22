"""The ``flux_scale`` leaf shared by every emitter of a scene.

Each emitter base (star, planet, disk, local zodi) owns one dimensionless
``flux_scale``, default one, applied exactly once in the base's public method.
A concrete model implements the private hook and never sees the scale.
"""

import jax
import jax.numpy as jnp


def check_flux_scale(scale) -> None:
    """Require a finite scale when it is concrete; a traced value is not checked.

    Positivity is not required: a detection test profiles a signed amplitude.
    A fit that must keep a scale positive gives it a positive law.

    Args:
        scale: The dimensionless ``flux_scale`` leaf, scalar or array.

    Raises:
        ValueError: If a concrete ``scale`` has a non-finite entry.
    """
    if isinstance(scale, jax.core.Tracer):
        return
    if not bool(jnp.all(jnp.isfinite(scale))):
        raise ValueError("flux_scale must be finite")
