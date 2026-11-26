"""Working precision for forward readers with typed and host inputs."""

import jax.numpy as jnp


def _working_dtype(*values):
    """Combine explicitly typed floating inputs, falling back to the JAX default."""
    typed = (
        x
        for x in values
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact)
    )
    return jnp.result_type(0.0, *typed)
