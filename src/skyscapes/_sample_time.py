"""First-sample offsets for tables and orbits on a shared observation clock."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_float_leaf, is_traced
from hwoutils.precision import like
from jaxtyping import Array
from orbix.kepler import Markley
from orbix.system import System


def _parts(days):
    # Three float32 significands carry all 53 significand bits of a day offset.
    days = np.asarray(days, dtype=np.float64)
    high = np.asarray(days.astype(np.float32), dtype=np.float64)
    residual = days - high
    middle = np.asarray(residual.astype(np.float32), dtype=np.float64)
    return high, middle, residual - middle


class SampleOrigin(eqx.Module):
    """A host float64 day offset stored exactly in either working precision.

    Attributes:
        parts: The offset as three float32-representable summands, last axis 3.
    """

    parts: Array = eqx.field(converter=as_float_leaf)

    def __init__(self, days):
        """Split a host float64 day offset into exact float32 summands.

        Args:
            days: First-sample offset from the scene epoch [d].
        """
        self.parts = np.stack(_parts(days), axis=-1)

    @property
    def days(self):
        """First sample in days from the scene epoch, as host float64."""
        return np.asarray(self.parts, dtype=np.float64).sum(axis=-1)

    def relative(self, time_d):
        """Difference host observations before transfer; preserve traced dates.

        Args:
            time_d: Observation days from the scene epoch.

        Returns:
            Days from the first sample, shape ``(..., T)``.
        """
        host_dates = not isinstance(time_d, jax.Array) and not is_traced(time_d)
        if host_dates and not is_traced(self.parts):
            origin = self.days
            if origin.ndim:
                origin = origin[..., None]
            return like(np.asarray(time_d, dtype=np.float64) - origin, self.parts)
        high, middle, low = jnp.moveaxis(self.parts, -1, 0)
        if high.ndim:
            high, middle, low = high[..., None], middle[..., None], low[..., None]
        if host_dates:
            # Host dates can remain concrete while a transformed scene is traced.
            date_high, date_middle, date_low = _parts(time_d)
            return (
                (like(date_high, high) - high)
                + (like(date_middle, middle) - middle)
                + (like(date_low, low) - low)
            )
        return ((jnp.asarray(time_d) - high) - middle) - low


class SampledSystem(System):
    """An orbix system whose orbit times start at the first recorded sample.

    Attributes:
        sample_origin: First-sample offset from the sky frame's epoch.
    """

    sample_origin: SampleOrigin

    def local_system(self):
        """Underlying orbit propagation takes days from the sample origin.

        Returns:
            The plain orbix system.
        """
        return System(self.star, self.planets, self.orbits, self.frame)

    def solve(self, time_d, *, solver=Markley()):  # noqa: B008
        """Solve at days from the sky frame's epoch, shared across systems.

        Args:
            time_d: Observation days from the sky frame's epoch.
            solver: Kepler solver.

        Returns:
            The solved system state.
        """
        return self.local_system().solve(
            self.sample_origin.relative(time_d), solver=solver
        )


def with_sample_origin(scene, origin):
    """Place the imported star, planet tables and dynamics on one sample clock.

    Args:
        scene: The imported scene.
        origin: First-sample offset, or None to leave the dynamics unshifted.

    Returns:
        The scene with its tables and dynamics on the sample clock.
    """
    fields = scene.dynamics
    if isinstance(fields, SampledSystem):
        fields = fields.local_system()
    if origin is not None:
        fields = SampledSystem(
            fields.star, fields.planets, fields.orbits, fields.frame, origin
        )
    scene = eqx.tree_at(lambda s: s.dynamics, scene, fields)

    def table_clocks(s):
        tables = (s.star, *(p.physical_model for p in s.planets))
        return tuple(t.sample_origin for t in tables if hasattr(t, "sample_origin"))

    return eqx.tree_at(
        table_clocks,
        scene,
        (origin,) * len(table_clocks(scene)),
        is_leaf=lambda x: x is None,
    )
