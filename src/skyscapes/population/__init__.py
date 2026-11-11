"""Planet populations drawn from occurrence rates.

Public API:

- :func:`load_occurrence_table`, :class:`OccurrenceTable` -- the Dulz et al. (2020)
  nominal occurrence grid in (scaled semi-major axis, mass) cells.
- :func:`load_planet_types`, :class:`PlanetTypes` -- planet types and their
  geometric albedo spectra.
- :func:`draw_population`, :func:`draw_populations`, :class:`PopulationConfig`,
  :class:`PlanetPopulation` --
  the draw, a pure function of a PRNG key and the host stars.
- :func:`sky_elements` -- sky-frame Keplerian elements of a drawn population.
- :func:`earth_like` -- the exoEarth-candidate box of ``get_earth_like_planet_indices``.
"""

from ._draw import (
    PlanetPopulation,
    PopulationConfig,
    SkyElements,
    draw_population,
    draw_populations,
    earth_like,
    sky_elements,
)
from ._tables import (
    OccurrenceTable,
    PlanetTypes,
    load_occurrence_table,
    load_planet_types,
    mass_to_radius,
    occurrence_table_from_grid,
    radius_to_mass,
)

__all__ = [
    "OccurrenceTable",
    "PlanetPopulation",
    "PlanetTypes",
    "PopulationConfig",
    "SkyElements",
    "draw_population",
    "draw_populations",
    "earth_like",
    "load_occurrence_table",
    "load_planet_types",
    "mass_to_radius",
    "occurrence_table_from_grid",
    "radius_to_mass",
    "sky_elements",
]
