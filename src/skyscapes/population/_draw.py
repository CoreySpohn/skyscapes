"""Draw planet populations for a set of stars from an occurrence-rate grid.

The draw follows the ExoVista planet generator, which draws from the Dulz et
al. (2020) occurrence rates and prunes by mutual Hill stability:

1. The planets of each planet bin are drawn for the whole star sample at once:
   ``round(rate * n_stars)`` planets per bin, placed in the bin's cells in
   proportion to the cell rates. The sample total is therefore fixed.
2. Each planet gets a scaled semi-major axis (log-uniform in its cell, times
   ``sqrt(L)`` of the host), a mass (log-uniform in its cell), a radius from the
   Chen and Kipping (2017) mean relation, a random host star, and uniform orbital
   angles; eccentricity and inclination are uniform in a configured range.
3. Planets of a star are sorted by semi-major axis. Wherever two neighbors are
   closer than `mutual_hill_min` mutual Hill radii the lighter one is removed
   (repeated until none is), then cells left short of their expected count are
   refilled on random stars and the stability pass runs again, until the total
   reaches the expected count or stalls. A star holds at most `max_planets`
   planets; the widest orbits are dropped beyond that.
4. Each planet gets a type from the ExoVista albedo list.

Because the refill places planets on random stars to conserve the sample total,
one call draws every star of a world together. It is a pure function of a key
and the star arrays with fixed output shapes, so ``jax.vmap`` over keys draws
many worlds and ``eqx.filter_jit`` compiles it once per number of stars.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import AU2m, G_si, Mearth2kg, Msun2kg, d2s
from orbix.equations.orbit import state_vector_to_keplerian

from ..habitable_zone import HZ_INNER_EEID, HZ_OUTER_EEID
from ._tables import mass_to_radius

_ABSENT_A = 1.0e20  # sort key of an empty slot
_EARTH_MASSES_PER_SUN = 333000.0  # the generator's mass ratio for the Hill criterion
_NEW_PER_STAR = 48  # most planets one star can receive in one refill pass


class PopulationConfig(eqx.Module):
    """Settings of the draw; the defaults are the ExoVista generator's.

    Attributes:
        e_min: Lower eccentricity bound (uniform draw).
        e_max: Upper eccentricity bound.
        inc_min_deg: Lower bound of the planet inclination to the system midplane.
        inc_max_deg: Upper bound of the planet inclination to the midplane.
        earth_candidate_probability: Probability that a planet inside the exoEarth
            candidate box is an exoEarth-candidate type.
        mutual_hill_min: Smallest allowed separation in mutual Hill radii.
        max_planets: Planet slots per star.
        max_iterations: Refill passes before the draw stops.
        max_stalled: Passes without a gain in stable planets before it stops.
    """

    e_min: float = 0.0
    e_max: float = 0.0
    inc_min_deg: float = 0.0
    inc_max_deg: float = 5.0
    earth_candidate_probability: float = 1.0
    mutual_hill_min: float = 6.0
    max_planets: int = eqx.field(static=True, default=30)
    max_iterations: int = eqx.field(static=True, default=200)
    max_stalled: int = eqx.field(static=True, default=50)


class PlanetPopulation(eqx.Module):
    """One world's planets: `max_planets` slots per star, sorted by semi-major axis.

    Planet leaves have shape ``[n_stars, max_planets]``; an absent slot has
    ``present`` false and zero mass, radius and semi-major axis.

    Attributes:
        present: Whether the slot holds a planet.
        mass_Mearth: Planet mass in Earth masses.
        radius_Rearth: Planet radius in Earth radii.
        a_AU: Semi-major axis in AU.
        e: Eccentricity.
        inc_deg: Inclination to the system midplane.
        node_deg: Longitude of the ascending node in the midplane frame.
        argperi_deg: Argument of periapsis in the midplane frame.
        mean_anomaly_deg: Mean anomaly at the reference epoch.
        type_index: Index into `PlanetTypes` (the albedo spectrum), 0 if absent.
        midplane_inc_deg: ``[n_stars]`` inclination of the system midplane to the sky.
        midplane_pa_deg: ``[n_stars]`` position angle of the system midplane.
    """

    present: jnp.ndarray
    mass_Mearth: jnp.ndarray
    radius_Rearth: jnp.ndarray
    a_AU: jnp.ndarray
    e: jnp.ndarray
    inc_deg: jnp.ndarray
    node_deg: jnp.ndarray
    argperi_deg: jnp.ndarray
    mean_anomaly_deg: jnp.ndarray
    type_index: jnp.ndarray
    midplane_inc_deg: jnp.ndarray
    midplane_pa_deg: jnp.ndarray


_PLANET_FIELDS = (
    "a_AU",
    "mass_Mearth",
    "radius_Rearth",
    "e",
    "inc_deg",
    "node_deg",
    "argperi_deg",
    "mean_anomaly_deg",
    "cell",
)


def earth_like(a_AU, radius_Rearth, luminosity_Lsun):
    """Whether a planet is an exoEarth candidate (``get_earth_like_planet_indices``).

    The scaled semi-major axis ``a / sqrt(L)`` lies in
    ``[HZ_INNER_EEID, HZ_OUTER_EEID)``
    and the radius in ``[0.8 / sqrt(a_scaled), 1.4)`` Earth radii (Stark et al.
    2019, the same box the ExoVista planet-type table uses for exoEarth candidates).
    """
    a_scaled = a_AU / jnp.sqrt(luminosity_Lsun)
    lower_r = 0.8 / jnp.sqrt(a_scaled)
    return (
        (a_scaled >= HZ_INNER_EEID)
        & (a_scaled < HZ_OUTER_EEID)
        & (radius_Rearth >= lower_r)
        & (radius_Rearth < 1.4)
    )


def _sort_by_a(fields, present):
    """Order every slot array of every star by semi-major axis, empty slots last."""
    key = jnp.where(present, fields["a_AU"], _ABSENT_A)
    order = jnp.argsort(key, axis=1)
    take = lambda x: jnp.take_along_axis(x, order, axis=1)  # noqa: E731
    return jax.tree_util.tree_map(take, fields), take(present)


def _remove_unstable(a_AU, mass_Mearth, present, three_star_mass_Mearth, hill_min):
    """Mark planets removed until no pair is closer than `hill_min` mutual Hill radii.

    One star, slots sorted by semi-major axis. The first unstable pair loses its
    lighter member and the scan restarts, which is the generator's sweep.
    """

    def unstable_pair(alive):
        idx = jnp.argsort(~alive, stable=True)
        a, m, ok = a_AU[idx], mass_Mearth[idx], alive[idx]
        a_in, a_out = a[:-1], a[1:]
        m_in, m_out = m[:-1], m[1:]
        both = ok[:-1] & ok[1:]
        safe_m = jnp.where(both, m_in + m_out, 1.0)
        hill = (
            2.0
            * (a_out - a_in)
            / (a_out + a_in)
            * (three_star_mass_Mearth / safe_m) ** (1.0 / 3.0)
        )
        bad = both & (hill < hill_min)
        k = jnp.argmax(bad)
        remove = jnp.where(m_out[k] > m_in[k], idx[k], idx[k + 1])
        return jnp.any(bad), remove

    def cond(alive):
        return unstable_pair(alive)[0]

    def body(alive):
        _, remove = unstable_pair(alive)
        return alive.at[remove].set(False)

    return jax.lax.while_loop(cond, body, present)


def _uniform_between(key, lo, hi, shape):
    return lo + jax.random.uniform(key, shape) * (hi - lo)


def draw_population(key, occurrence, types, luminosity_Lsun, mass_Msun, config=None):
    """Draw one world's planets for the stars with the given luminosities and masses.

    Args:
        key: A JAX PRNG key.
        occurrence: The `OccurrenceTable` to draw from.
        types: The `PlanetTypes` that planets are assigned to.
        luminosity_Lsun: ``[n_stars]`` host luminosities in solar units.
        mass_Msun: ``[n_stars]`` host masses in solar masses.
        config: A `PopulationConfig`; the default is the ExoVista generator's.

    Returns:
        A `PlanetPopulation` with ``[n_stars, config.max_planets]`` planet leaves.
    """
    config = PopulationConfig() if config is None else config
    luminosity_Lsun = jnp.asarray(luminosity_Lsun, float)
    n_stars = luminosity_Lsun.shape[0]
    n_slots = config.max_planets
    n_a, n_m = occurrence.rate.shape
    cap = math.ceil(1.25 * occurrence.total_rate * n_stars) + 64
    sqrt_l = jnp.sqrt(luminosity_Lsun)
    three_ms = 3.0 * jnp.asarray(mass_Msun, float) * _EARTH_MASSES_PER_SUN
    log_a = jnp.log(occurrence.a_edges_AU)
    log_m = jnp.log(occurrence.mass_edges_Mearth)

    key, k_pa, k_inc, k_flip, k_bin = jax.random.split(key, 5)
    midplane_pa = jax.random.uniform(k_pa, (n_stars,)) * 360.0
    cos_inc = jax.random.uniform(k_inc, (n_stars,)) * 2.0 - 1.0
    midplane_inc = jnp.degrees(jnp.arccos(cos_inc))
    midplane_inc = jnp.where(
        jax.random.uniform(k_flip, (n_stars,)) < 0.5, midplane_inc - 180.0, midplane_inc
    )

    # Expected planets per cell. Each (cell, star) entry is a Bernoulli trial at the
    # cell rate; every planet bin is then forced to hold round(rate * n_stars)
    # planets by switching on the entries with the largest random numbers or
    # switching off the hits with the smallest, as the generator does.
    n_cells = n_a * n_m
    bin_flat = occurrence.bin_id.ravel()
    covered = (bin_flat >= 0)[:, None]
    bin_entry = jnp.broadcast_to(
        jnp.where(bin_flat >= 0, bin_flat, 0)[:, None], (n_cells, n_stars)
    )
    rand = jax.random.uniform(k_bin, (n_cells, n_stars))
    hit = (rand < occurrence.rate.ravel()[:, None]) & covered
    per_bin = lambda x: jax.ops.segment_sum(  # noqa: E731
        x.sum(axis=1).astype(float),
        jnp.where(bin_flat >= 0, bin_flat, 0),
        occurrence.n_bins,
    )
    target = jnp.round(
        per_bin(covered * occurrence.rate.ravel()[:, None] * jnp.ones(n_stars))
    )
    hits_per_bin = per_bin(hit)
    diff = target - hits_per_bin
    entries_per_bin = per_bin(covered * jnp.ones(n_stars))

    def rank_in_bin(sort_key, per_bin_count):
        first = jnp.cumsum(per_bin_count) - per_bin_count
        order = jnp.argsort(sort_key.ravel())
        position = (
            jnp.zeros(sort_key.size)
            .at[order]
            .set(jnp.arange(sort_key.size, dtype=float))
        )
        return position.reshape(sort_key.shape) - first[bin_entry]

    big = occurrence.n_bins + 1.0
    add_rank = rank_in_bin(
        jnp.where(covered, bin_entry + (1.0 - rand) * 0.5, big), entries_per_bin
    )
    drop_rank = rank_in_bin(jnp.where(hit, bin_entry + rand * 0.5, big), hits_per_bin)
    n_add, n_drop = diff[bin_entry], -diff[bin_entry]
    hit = (hit | (covered & (n_add > 0) & (add_rank < n_add))) & ~(
        hit & (n_drop > 0) & (drop_rank < n_drop)
    )
    expected = hit.sum(axis=1).astype(float)
    n_expected = jnp.sum(expected)
    k = jnp.arange(cap)

    empty = {name: jnp.zeros((n_stars, n_slots)) for name in _PLANET_FIELDS}
    empty["a_AU"] = jnp.full((n_stars, n_slots), _ABSENT_A)
    empty["cell"] = jnp.full((n_stars, n_slots), -1, jnp.int32)
    present = jnp.zeros((n_stars, n_slots), bool)

    def refill(carry):
        key, fields, present, _, n_stalled, it = carry
        key, k_a, k_m, k_star, k_node, k_peri, k_anom, k_e, k_i = jax.random.split(
            key, 9
        )
        have = (
            jnp.zeros(n_a * n_m)
            .at[jnp.where(present, fields["cell"], 0).ravel()]
            .add(present.ravel().astype(float))
        )
        deficit = jnp.clip(expected - have, 0.0, None)
        cum_def = jnp.cumsum(deficit)
        valid = k < cum_def[-1]
        cell = jnp.clip(jnp.searchsorted(cum_def, k, side="right"), 0, n_a * n_m - 1)
        ia, im = cell // n_m, cell % n_m
        star = jnp.clip(
            (jax.random.uniform(k_star, (cap,)) * n_stars).astype(jnp.int32),
            0,
            n_stars - 1,
        )
        a_scaled = jnp.exp(_uniform_between(k_a, log_a[ia], log_a[ia + 1], (cap,)))
        mass = jnp.exp(_uniform_between(k_m, log_m[im], log_m[im + 1], (cap,)))
        new = {
            "a_AU": a_scaled * sqrt_l[star],
            "mass_Mearth": mass,
            "radius_Rearth": mass_to_radius(mass),
            "e": _uniform_between(k_e, config.e_min, config.e_max, (cap,)),
            "inc_deg": _uniform_between(
                k_i, config.inc_min_deg, config.inc_max_deg, (cap,)
            ),
            "node_deg": jax.random.uniform(k_node, (cap,)) * 360.0,
            "argperi_deg": jax.random.uniform(k_peri, (cap,)) * 360.0,
            "mean_anomaly_deg": jax.random.uniform(k_anom, (cap,)) * 360.0,
            "cell": cell.astype(jnp.int32),
        }
        # Rank of each new planet among those landing on the same star.
        order = jnp.argsort(jnp.where(valid, star, n_stars), stable=True)
        sorted_star = jnp.where(valid, star, n_stars)[order]
        rank_sorted = k - jnp.searchsorted(sorted_star, sorted_star, side="left")
        rank = jnp.zeros(cap, jnp.int32).at[order].set(rank_sorted.astype(jnp.int32))
        keep = valid & (rank < _NEW_PER_STAR)
        row = jnp.where(keep, star, n_stars)
        merged, merged_present = (
            {},
            jnp.concatenate(
                [
                    present,
                    jnp.zeros((n_stars, _NEW_PER_STAR), bool)
                    .at[row, rank]
                    .set(True, mode="drop"),
                ],
                axis=1,
            ),
        )
        for name in _PLANET_FIELDS:
            fill = jnp.zeros((n_stars, _NEW_PER_STAR), fields[name].dtype)
            fill = fill.at[row, rank].set(
                new[name].astype(fields[name].dtype), mode="drop"
            )
            merged[name] = jnp.concatenate([fields[name], fill], axis=1)
        merged, merged_present = _sort_by_a(merged, merged_present)
        fields = {name: value[:, :n_slots] for name, value in merged.items()}
        present = merged_present[:, :n_slots]
        alive = jax.vmap(_remove_unstable, in_axes=(0, 0, 0, 0, None))(
            fields["a_AU"],
            fields["mass_Mearth"],
            present,
            three_ms,
            config.mutual_hill_min,
        )
        fields, present = _sort_by_a(fields, alive)
        n_stable = jnp.sum(present).astype(jnp.int32)
        previous = carry[3]
        n_stalled = jnp.where(n_stable <= previous, n_stalled + 1, 0)
        return key, fields, present, n_stable, n_stalled, it + 1

    def not_done(carry):
        _, _, _, n_stable, n_stalled, it = carry
        return (
            (n_stable < n_expected)
            & (n_stalled < config.max_stalled)
            & (it < config.max_iterations)
        )

    carry = (
        key,
        empty,
        present,
        jnp.zeros((), jnp.int32),
        jnp.zeros((), jnp.int32),
        jnp.zeros((), jnp.int32),
    )
    _, fields, present, _, _, _ = jax.lax.while_loop(not_done, refill, carry)

    # Planet types: those whose ranges contain the planet, by selection weight.
    key, k_type = jax.random.split(key)
    a_scaled = fields["a_AU"] / sqrt_l[:, None]
    radius = fields["radius_Rearth"]
    r_lo = jnp.where(
        types.earth_candidate[None, None, :],
        types.radius_min_Rearth[None, None, :] / jnp.sqrt(a_scaled)[..., None],
        types.radius_min_Rearth[None, None, :],
    )
    fits = (
        (r_lo < radius[..., None])
        & (radius[..., None] < types.radius_max_Rearth[None, None, :])
        & (types.a_min_AU[None, None, :] < a_scaled[..., None])
        & (a_scaled[..., None] < types.a_max_AU[None, None, :])
    )
    any_candidate = jnp.any(fits & types.earth_candidate, axis=-1, keepdims=True)
    p_eec = config.earth_candidate_probability
    scale = jnp.where(types.earth_candidate, p_eec, 1.0 - p_eec)
    weight = types.weight * fits * jnp.where(any_candidate, scale, 1.0)
    cum = jnp.cumsum(weight, axis=-1)
    draw = jax.random.uniform(k_type, radius.shape) * cum[..., -1]
    type_index = jnp.sum(cum < draw[..., None], axis=-1)
    fallback = types.names.index("Mars") if "Mars" in types.names else 0
    type_index = jnp.where(cum[..., -1] > 0.0, type_index, fallback)
    type_index = jnp.where(present, jnp.minimum(type_index, len(types.names) - 1), 0)

    def masked(name):
        return jnp.where(present, fields[name], 0.0)

    return PlanetPopulation(
        present=present,
        mass_Mearth=masked("mass_Mearth"),
        radius_Rearth=masked("radius_Rearth"),
        a_AU=masked("a_AU"),
        e=masked("e"),
        inc_deg=masked("inc_deg"),
        node_deg=masked("node_deg"),
        argperi_deg=masked("argperi_deg"),
        mean_anomaly_deg=masked("mean_anomaly_deg"),
        type_index=type_index,
        midplane_inc_deg=midplane_inc,
        midplane_pa_deg=midplane_pa,
    )


class SkyElements(eqx.Module):
    """Keplerian elements of every planet seen on the sky, at the reference epoch.

    Angles follow orbix: ``i_rad`` is the inclination to the sky plane, ``W_rad`` the
    position angle of the ascending node, ``w_rad`` the argument of periapsis and
    ``M0_rad`` the mean anomaly. Each leaf has shape ``[n_stars, max_planets]``.

    Attributes:
        a_AU: Semi-major axis.
        e: Eccentricity.
        i_rad: Inclination to the sky plane.
        W_rad: Position angle of the ascending node.
        w_rad: Argument of periapsis.
        M0_rad: Mean anomaly at the reference epoch.
        period_d: Orbital period in days (planet mass included in the total mass).
    """

    a_AU: jnp.ndarray
    e: jnp.ndarray
    i_rad: jnp.ndarray
    W_rad: jnp.ndarray
    w_rad: jnp.ndarray
    M0_rad: jnp.ndarray
    period_d: jnp.ndarray


def _rotate_x(v, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c])


def _rotate_z(v, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])


def _kepler_e(mean_anomaly, e):
    """Eccentric anomaly by Newton iteration (the draw's eccentricities are small)."""
    ecc_anomaly = mean_anomaly + e * jnp.sin(mean_anomaly)
    for _ in range(12):
        ecc_anomaly = ecc_anomaly - (
            ecc_anomaly - e * jnp.sin(ecc_anomaly) - mean_anomaly
        ) / (1.0 - e * jnp.cos(ecc_anomaly))
    return ecc_anomaly


def _sky_elements_one(
    a_AU, e, inc, node, peri, anom, mass_Mearth, star_mass_Msun, mid_inc, mid_pa
):
    mu = G_si * (star_mass_Msun * Msun2kg + mass_Mearth * Mearth2kg)
    a_m = a_AU * AU2m
    n = jnp.sqrt(mu / a_m**3)
    ecc_anomaly = _kepler_e(anom, e)
    root = jnp.sqrt(1.0 - e * e)
    denom = 1.0 - e * jnp.cos(ecc_anomaly)
    r = jnp.stack(
        [a_m * (jnp.cos(ecc_anomaly) - e), a_m * root * jnp.sin(ecc_anomaly), 0.0]
    )
    v = jnp.stack(
        [
            -a_m * n * jnp.sin(ecc_anomaly) / denom,
            root * a_m * n * jnp.cos(ecc_anomaly) / denom,
            0.0,
        ]
    )

    def to_sky(x):
        x = _rotate_z(x, peri)
        x = _rotate_x(x, inc)
        x = _rotate_z(x, node)
        x = _rotate_x(x, mid_inc)
        x = _rotate_z(x, mid_pa)
        return jnp.stack(
            [-x[0], x[1], -x[2]]
        )  # (west, north, toward) to (east, north, away)

    _, e_sky, i, big_w, w, m = state_vector_to_keplerian(to_sky(r), to_sky(v), mu)
    return e_sky, i, big_w, w, m, 2.0 * jnp.pi / n / d2s


def sky_elements(population, star_mass_Msun):
    """Sky-frame Keplerian elements of a `PlanetPopulation`.

    The planet state at the reference epoch is built in the system midplane
    frame, rotated by the midplane inclination and position angle in the
    generator's order and handed to orbix's state-vector conversion; this is the
    chain by which ExoVista writes planet states that skyscapes then reads back.
    Absent slots return finite filler values.

    Args:
        population: One world's `PlanetPopulation`.
        star_mass_Msun: ``[n_stars]`` host masses in solar masses.
    """
    rad = jnp.pi / 180.0
    present = population.present
    safe = lambda x, fill: jnp.where(present, x, fill)  # noqa: E731
    one = jax.vmap(
        jax.vmap(_sky_elements_one, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None)),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    e, i, big_w, w, m, period = one(
        safe(population.a_AU, 1.0),
        population.e,
        population.inc_deg * rad,
        population.node_deg * rad,
        population.argperi_deg * rad,
        population.mean_anomaly_deg * rad,
        population.mass_Mearth,
        jnp.asarray(star_mass_Msun, float),
        population.midplane_inc_deg * rad,
        population.midplane_pa_deg * rad,
    )
    return SkyElements(
        a_AU=population.a_AU,
        e=e,
        i_rad=i,
        W_rad=big_w,
        w_rad=w,
        M0_rad=m,
        period_d=period,
    )


def draw_populations(
    key, n_worlds, occurrence, types, luminosity_Lsun, mass_Msun, config=None
):
    """Draw `n_worlds` independent worlds, one after another to bound memory.

    Equivalent to calling `draw_population` with ``n_worlds`` split keys; the
    leaves of the result gain a leading world axis. Use ``eqx.filter_jit`` on a
    wrapper for repeated calls with the same number of stars.
    """
    keys = jax.random.split(key, n_worlds)
    return jax.lax.map(
        lambda k: draw_population(
            k, occurrence, types, luminosity_Lsun, mass_Msun, config
        ),
        keys,
    )
