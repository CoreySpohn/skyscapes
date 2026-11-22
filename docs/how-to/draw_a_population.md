---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Draw a planet population around a list of stars

{mod}`skyscapes.population` draws the planets of a synthetic universe from
occurrence rates. Give it the luminosities and masses of the stars in a target
list and it returns, for every star, a set of planets with masses, radii,
orbits and albedo types. The draw is a pure function of a JAX PRNG key, so the
same key and the same stars always give the same planets.

## One world

The occurrence grid of Dulz et al. (2020) and the planet types come with the
package. Three stars are enough to see the shape of the result.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

from skyscapes.population import (
    draw_population,
    load_occurrence_table,
    load_planet_types,
)

jax.config.update("jax_enable_x64", True)

occurrence = load_occurrence_table()
types = load_planet_types()

luminosity_Lsun = jnp.array([1.0, 0.5, 0.2])
mass_Msun = jnp.array([1.0, 0.8, 0.5])

world = draw_population(jax.random.key(0), occurrence, types, luminosity_Lsun, mass_Msun)
print("planets per star:", world.present.sum(axis=1))
print("semi-major axes of star 0 [AU]:", world.a_AU[0, :4])
```

Every planet leaf of the returned {class}`~skyscapes.population.PlanetPopulation`
has shape `[n_stars, max_planets]`. Each star owns `max_planets` slots, sorted by
semi-major axis, and `present` marks the slots that hold a planet. An empty slot
has zero mass, radius and semi-major axis, so mask with `present` before
averaging over planets. Neighbouring planets are kept at least
`mutual_hill_min` mutual Hill radii apart, which is why the draw refills slots
until the planets are stable.

## Choose the eccentricity and the inclination

The defaults are those of the ExoVista generator: circular orbits, and planet
inclinations to the system midplane uniform from 0 to 5 degrees.
{class}`~skyscapes.population.PopulationConfig` changes either one with a
`priorities` distribution. {func}`~skyscapes.population.kipping_eccentricity`
and {func}`~skyscapes.population.rayleigh_eccentricity` build the two common
eccentricity distributions, and `inclination_rad` takes any distribution on
$[0, \pi]$, or `None` for coplanar orbits.

```{code-cell} ipython3
from skyscapes.population import PopulationConfig, kipping_eccentricity

config = PopulationConfig(eccentricity=kipping_eccentricity(0.8))
eccentric = draw_population(
    jax.random.key(0), occurrence, types, luminosity_Lsun, mass_Msun, config
)
print("eccentricities of star 0:", eccentric.e[0, :4])
```

## Find the exoEarth candidates

{func}`~skyscapes.population.earth_like` applies the exoEarth-candidate box, a
semi-major axis inside the habitable zone and a radius in the Earth-like range,
to any planets, with the host luminosity broadcast along the slot axis.

```{code-cell} ipython3
from skyscapes.population import earth_like

candidates = earth_like(
    eccentric.a_AU, eccentric.radius_Rearth, luminosity_Lsun[:, None]
) & eccentric.present
print("exoEarth candidates per star:", candidates.sum(axis=1))
```

## Put the planets on the sky

The planet orbits above are written in the frame of the system midplane. Each
star also carries a midplane inclination and position angle on the sky.
{func}`~skyscapes.population.sky_elements` rotates the orbits into the sky frame
and returns the Keplerian elements that orbix expects.

```{code-cell} ipython3
from skyscapes.population import sky_elements

sky = sky_elements(eccentric, mass_Msun)
print("sky inclination of star 0 planets [rad]:", sky.i_rad[0, :3])
print("periods of star 0 planets [d]:", sky.period_d[0, :3])
```

## Many worlds

{func}`~skyscapes.population.draw_populations` draws independent worlds one
after another, so memory stays at one world. The leaves gain a leading world
axis.

```{code-cell} ipython3
from skyscapes.population import draw_populations

worlds = draw_populations(
    jax.random.key(1), 4, occurrence, types, luminosity_Lsun, mass_Msun
)
print(worlds.present.shape)
```

## Draw `Scene` objects with a known density

Fitting and survey codes need the planets as {class}`~skyscapes.Scene` batches,
and the probability density of each draw. {class}`~skyscapes.population.OccurrencePopulation`
holds the same tables and configuration as a point process over
{class}`~skyscapes.population.HostStars`. Its `sample` method returns scenes with
batch shape `(n_worlds, n_stars)`, and `mark_log_density` returns the log
density of every planet against the reference measure of orbix's classical
orbit chart. The density needs an eccentricity distribution, because circular
orbits have no density over all six orbital elements.

```{code-cell} ipython3
import equinox as eqx
from hwoutils.epoch import Epoch

from skyscapes.population import HostStars, OccurrencePopulation
from skyscapes.scene import FlatStar

hosts = HostStars(
    star_id=jnp.arange(3),
    mass_Msun=mass_Msun,
    luminosity_Lsun=luminosity_Lsun,
    dist_pc=jnp.array([5.0, 8.0, 12.0]),
    star=FlatStar(jnp.full(3, 1e9), jnp.zeros(3), flux_scale=jnp.ones(3)),
)
process = OccurrencePopulation(occurrence, types, config, Epoch())

scenes = eqx.filter_jit(process.sample)(jax.random.key(0), hosts, 2)
log_density = process.mark_log_density(scenes, hosts)
print("scene batch:", scenes.valid.shape[:-1], "slots:", scenes.valid.shape[-1])
print("exoEarth candidates:", int(process.earth_like(scenes, hosts).sum()))
print("log density of the first valid planet:", log_density[scenes.valid][0])
```

Each star keeps its planets in separate random streams keyed by its `star_id`,
so selecting or reordering the hosts leaves the planets of the remaining stars
unchanged. `sample` prunes each star independently and keeps the innermost
`max_planets`, which is the process the density describes before pruning, so
treat the density as that of the unpruned process.
