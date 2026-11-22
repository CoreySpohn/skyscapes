"""Imported stellar spot histories, independent of an optical response."""

import equinox as eqx
from jaxtyping import Array


class StarSpots(eqx.Module):
    """Recorded spot geometry and component temperatures.

    ``times_d`` stores days from the scene epoch; the owning scene holds
    the epoch metadata. Geometry arrays have shape ``(n_times, n_spots)``.
    ``area_m2`` is the total spot area, including the umbra; ``area_ratio`` is total
    spot area divided by umbra area, as used by ExoVista. Longitudes
    and latitudes retain the source's stellar coordinate convention.

    These are sampled source records, not an evolution or rendering
    model. In particular, loading them does not add asymmetric stellar
    leakage to a diameter-only coronagraph map. Temperatures may be
    unknown in files that omit the corresponding header cards.

    Attributes:
        times_d: Sample times, days from the scene epoch, shape ``(n_times,)``.
        latitude_deg: Spot latitudes [deg], shape ``(n_times, n_spots)``.
        longitude_deg: Spot longitudes [deg], shape ``(n_times, n_spots)``.
        area_m2: Total spot area including the umbra [m^2], shape
            ``(n_times, n_spots)``.
        area_ratio: Total spot area divided by umbra area, dimensionless, shape
            ``(n_times, n_spots)``.
        umbra_temperature_K: Umbra temperature [K], or None when unknown.
        penumbra_temperature_K: Penumbra temperature [K], or None when unknown.
    """

    times_d: Array
    latitude_deg: Array
    longitude_deg: Array
    area_m2: Array
    area_ratio: Array
    umbra_temperature_K: float | None = None
    penumbra_temperature_K: float | None = None
