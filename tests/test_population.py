"""Planet population draw: stability, fixed total, determinism, sky elements, tables."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyscapes.population import (
    PopulationConfig,
    draw_population,
    earth_like,
    load_occurrence_table,
    load_planet_types,
    mass_to_radius,
    radius_to_mass,
    sky_elements,
)

N_STARS = 8
LUMINOSITY = jnp.linspace(0.3, 1.4, N_STARS)
MASS = 0.6 + 0.4 * LUMINOSITY**0.25


@pytest.fixture(scope="module")
def tables():
    return load_occurrence_table(), load_planet_types()


@pytest.fixture(scope="module")
def draw(tables):
    """Compiled draw for the module's one program: eight stars, default config."""
    occurrence, types = tables
    run = eqx.filter_jit(draw_population)
    return lambda seed, config=None: run(
        jax.random.key(seed), occurrence, types, LUMINOSITY, MASS, config
    )


@pytest.fixture(scope="module")
def world(draw):
    return draw(0)


def test_planets_are_hill_stable_sorted_and_zero_when_absent(world):
    """Every neighboring pair is at least 6 mutual Hill radii apart and sorted by a.

    Fails if the stability pass is skipped or compares the wrong neighbors, if the
    slots are not sorted by semi-major axis, or if an absent slot keeps values.
    """
    present = np.asarray(world.present)
    a, m = np.asarray(world.a_AU), np.asarray(world.mass_Mearth)
    for s in range(N_STARS):
        n = present[s].sum()
        assert present[s, :n].all() and not present[s, n:].any()
        assert np.all(np.diff(a[s, :n]) > 0.0)
        three_ms = 3.0 * float(MASS[s]) * 333000.0
        hill = (
            2.0
            * (a[s, 1:n] - a[s, : n - 1])
            / (a[s, 1:n] + a[s, : n - 1])
            * (three_ms / (m[s, 1:n] + m[s, : n - 1])) ** (1.0 / 3.0)
        )
        assert np.all(hill >= 6.0)
    for name in ("a_AU", "mass_Mearth", "radius_Rearth", "inc_deg"):
        assert np.all(np.asarray(getattr(world, name))[~present] == 0.0)


def test_sample_total_is_the_binned_expectation(tables, world):
    """The stable planets of the sample match sum of round(rate * n_stars) per bin.

    The count lands within 5 percent: the bin adjustment can switch on an entry that
    is already a hit. Fails if the refill pass stops short (the count then falls to
    the first-pass survivors, 30 percent or more below) or a bin's count is not forced.
    """
    occurrence, _ = tables
    rate, bins = (
        np.asarray(occurrence.rate).ravel(),
        np.asarray(occurrence.bin_id).ravel(),
    )
    per_bin = np.bincount(bins[bins >= 0], weights=rate[bins >= 0])
    target = np.round(per_bin * N_STARS).sum()
    assert abs(int(world.present.sum()) - target) <= 0.05 * target


def test_draw_is_deterministic_in_the_key(draw, world):
    """The same key gives the same world and another key a different one.

    Fails if the draw reads global random state or ignores the key.
    """
    again, other = draw(0), draw(1)
    assert np.array_equal(np.asarray(again.a_AU), np.asarray(world.a_AU))
    assert not np.array_equal(np.asarray(other.a_AU), np.asarray(world.a_AU))


def test_orbit_ranges_follow_the_config(draw):
    """Eccentricity and inclination fall inside the configured uniform ranges.

    Fails if the config bounds are swapped, ignored or applied in the wrong units.
    """
    config = PopulationConfig(e_min=0.1, e_max=0.2, inc_min_deg=1.0, inc_max_deg=3.0)
    p = draw(0, config)
    present = np.asarray(p.present)
    e, inc = np.asarray(p.e)[present], np.asarray(p.inc_deg)[present]
    assert e.min() >= 0.1 and e.max() <= 0.2
    assert inc.min() >= 1.0 and inc.max() <= 3.0


@pytest.mark.parametrize(
    ("a_AU", "radius_Rearth", "luminosity", "expected"),
    [
        (0.95, 1.0, 1.0, True),  # inner edge is inclusive
        (1.67, 1.0, 1.0, False),  # outer edge is exclusive
        (1.2, 1.4, 1.0, False),  # upper radius edge is exclusive
        (1.0, 0.79, 1.0, False),  # below 0.8 / sqrt(a) at a = 1
        (0.5, 1.0, 0.25, True),  # scaled by sqrt(L): a / sqrt(L) = 1
    ],
)
def test_earth_like_box_edges(a_AU, radius_Rearth, luminosity, expected):
    """The exoEarth-candidate box has the edges of get_earth_like_planet_indices.

    Fails if an edge flips inclusivity or the luminosity scaling is dropped.
    """
    assert bool(earth_like(a_AU, radius_Rearth, luminosity)) is expected


def test_sky_elements_match_orbit_geometry(world):
    """Sky elements obey Kepler's third law and the midplane and eccentricity limits.

    A massless 1 AU planet around a solar-mass star has a one-year period; a coplanar
    planet's sky inclination has the cosine of the midplane inclination; an input
    eccentricity is returned. Fails if a unit or rotation in the chain is wrong.
    """
    present = world.present
    base = eqx.tree_at(
        lambda p: (p.a_AU, p.mass_Mearth, p.inc_deg, p.e),
        world,
        (
            jnp.where(present, 1.0, 0.0),
            jnp.zeros_like(world.mass_Mearth),
            jnp.zeros_like(world.inc_deg),
            jnp.full_like(world.e, 0.25),
        ),
    )
    sky = sky_elements(base, jnp.ones(N_STARS))
    p = np.asarray(present)
    np.testing.assert_allclose(np.asarray(sky.period_d)[p], 365.2564, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(sky.e)[p], 0.25, atol=1e-9)
    cos_i = np.abs(np.cos(np.asarray(sky.i_rad)))
    cos_mid = np.abs(np.cos(np.radians(np.asarray(world.midplane_inc_deg))))
    np.testing.assert_allclose(
        cos_i[p], np.broadcast_to(cos_mid[:, None], p.shape)[p], atol=1e-9
    )


def test_planet_types_cover_every_drawn_planet_and_albedo_is_tabulated(tables, world):
    """Present planets get a type index and the albedo table carries the named spectra.

    Fails if the albedo table and the type list are misaligned (Jupiter's albedo
    at 1000 nm is 0.108) or a planet falls outside every type range.
    """
    _, types = tables
    assert np.all(np.asarray(types.albedo) >= 0.0)
    index = np.asarray(world.type_index)[np.asarray(world.present)]
    assert index.min() >= 0 and index.max() < len(types.names)
    jupiter = types.names.index("Jupiter")
    assert float(types.geometric_albedo(jupiter, 1000.0)) == pytest.approx(
        0.1077, rel=1e-3
    )


def test_occurrence_grid_total_and_mass_radius_inverse(tables):
    """The grid holds 11.0 planets per star in 25 bins; the radius law inverts.

    Fails if the table build changes the integrated rate or the radius edges drift.
    """
    occurrence, _ = tables
    assert occurrence.total_rate == pytest.approx(11.0002, abs=1e-3)
    assert occurrence.n_bins == 25
    radius = np.linspace(0.5, 9.0, 30)
    np.testing.assert_allclose(
        mass_to_radius(radius_to_mass(radius)), radius, rtol=1e-9
    )
    radius_jax = jnp.asarray(radius)
    mass_jax = jax.jit(jax.vmap(radius_to_mass))(radius_jax)
    assert isinstance(radius_to_mass(radius_jax), jax.Array)
    np.testing.assert_allclose(mass_jax, radius_to_mass(radius), rtol=1e-12)
    np.testing.assert_allclose(jax.jit(mass_to_radius)(mass_jax), radius, rtol=1e-9)
