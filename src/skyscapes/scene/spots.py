"""Imported stellar spot histories, independent of an optical response."""

import equinox as eqx
from jaxtyping import Array


class StarSpots(eqx.Module):
    """Recorded spot geometry and component temperatures.

    Geometry arrays have shape ``(n_times, n_spots)``. ``area_m2`` is
    the total spot area, including the umbra; ``area_ratio`` is total
    spot area divided by umbra area, as used by ExoVista. Longitudes
    and latitudes retain the source's stellar coordinate convention.

    These are sampled source records, not an evolution or rendering
    model. In particular, loading them does not add asymmetric stellar
    leakage to a diameter-only coronagraph map. Temperatures may be
    unknown in files that omit the corresponding header cards.
    """

    times_jd: Array
    latitude_deg: Array
    longitude_deg: Array
    area_m2: Array
    area_ratio: Array
    umbra_temperature_k: float | None = None
    penumbra_temperature_k: float | None = None
