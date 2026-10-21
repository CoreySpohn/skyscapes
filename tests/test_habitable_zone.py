"""Habitable-zone edges: EEID scaling and Kopparapu effective-flux limits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyscapes.habitable_zone import (
    HZ_INNER_EEID,
    HZ_OUTER_EEID,
    KOPPARAPU_SEFF_COEFFICIENTS,
    OPTIMISTIC_HZ_INNER_EEID,
    OPTIMISTIC_HZ_OUTER_EEID,
    eeid_au,
    habitable_zone_au,
    kopparapu_effective_flux,
    kopparapu_habitable_zone_au,
)


def test_solar_twin_gives_the_published_edges_exactly():
    """A solar twin's conservative zone is 0.95 to 1.67 AU, exactly."""
    inner_au, outer_au = habitable_zone_au(1.0)
    assert float(inner_au) == 0.95
    assert float(outer_au) == 1.67


def test_solar_twin_optimistic_edges():
    """A solar twin's optimistic zone is 0.75 to 1.77 AU, exactly."""
    inner_au, outer_au = habitable_zone_au(1.0, optimistic=True)
    assert float(inner_au) == 0.75
    assert float(outer_au) == 1.77


def test_named_edges_are_the_published_values():
    """The named EEID edges carry the published values."""
    assert (HZ_INNER_EEID, HZ_OUTER_EEID) == (0.95, 1.67)
    assert (OPTIMISTIC_HZ_INNER_EEID, OPTIMISTIC_HZ_OUTER_EEID) == (0.75, 1.77)


@pytest.mark.parametrize("luminosity_lsun", [0.01, 0.25, 4.0, 9.0])
def test_edges_scale_with_root_luminosity(luminosity_lsun):
    """Both edges scale with sqrt(L / L_sun)."""
    inner_au, outer_au = habitable_zone_au(luminosity_lsun)
    scale = np.sqrt(luminosity_lsun)
    np.testing.assert_allclose(float(eeid_au(luminosity_lsun)), scale, rtol=1e-15)
    np.testing.assert_allclose(float(inner_au), 0.95 * scale, rtol=1e-15)
    np.testing.assert_allclose(float(outer_au), 1.67 * scale, rtol=1e-15)


def test_quadrupling_luminosity_doubles_the_edges():
    """Four times the luminosity puts each edge twice as far out."""
    inner_1, outer_1 = habitable_zone_au(1.0)
    inner_4, outer_4 = habitable_zone_au(4.0)
    assert float(inner_4) == 2.0 * float(inner_1)
    assert float(outer_4) == 2.0 * float(outer_1)


def test_vmap_over_stars_matches_broadcasting():
    """A vmap over a vector of stars agrees with plain broadcasting."""
    luminosities_lsun = jnp.array([0.05, 0.5, 1.0, 2.0, 7.5])
    inner_v, outer_v = jax.vmap(habitable_zone_au)(luminosities_lsun)
    inner_b, outer_b = habitable_zone_au(luminosities_lsun)
    assert inner_v.shape == luminosities_lsun.shape
    np.testing.assert_array_equal(inner_v, inner_b)
    np.testing.assert_array_equal(outer_v, outer_b)
    np.testing.assert_allclose(inner_v, 0.95 * np.sqrt(luminosities_lsun))


def test_jit_and_grad():
    """The edges trace under jit and differentiate under grad."""
    inner_au = jax.jit(lambda lum: habitable_zone_au(lum)[0])(4.0)
    assert float(inner_au) == pytest.approx(1.9)
    # d(0.95 sqrt(L))/dL = 0.95 / (2 sqrt(L))
    slope = jax.grad(lambda lum: habitable_zone_au(lum)[0])(4.0)
    assert float(slope) == pytest.approx(0.95 / 4.0)


@pytest.mark.parametrize("limit", sorted(KOPPARAPU_SEFF_COEFFICIENTS))
def test_kopparapu_flux_at_the_solar_temperature_is_the_solar_coefficient(limit):
    """At Teff = 5780 K the polynomial reduces to its constant term."""
    s_sun = KOPPARAPU_SEFF_COEFFICIENTS[limit][0]
    assert float(kopparapu_effective_flux(5780.0, limit)) == s_sun


# S_eff at 3000, 4500 and 7000 K, evaluated by hand from Table 1 of
# Kopparapu et al. (2014) for a 1 Earth-mass planet.
KOPPARAPU_TABLE_1_CHECKS = {
    "recent_venus": (1.4791538008, 1.5637132906, 2.0432451855),
    "runaway_greenhouse": (0.9219746116, 0.9746303702, 1.2736568438),
    "maximum_greenhouse": (0.2329794468, 0.2850033676, 0.4267713723),
    "early_mars": (0.2094049062, 0.2561806832, 0.3836158471),
}


@pytest.mark.parametrize("limit", sorted(KOPPARAPU_TABLE_1_CHECKS))
def test_kopparapu_flux_away_from_the_solar_temperature(limit):
    """The quartic matches hand evaluations of Table 1 at non-solar Teff."""
    flux = kopparapu_effective_flux(jnp.array([3000.0, 4500.0, 7000.0]), limit)
    np.testing.assert_allclose(flux, KOPPARAPU_TABLE_1_CHECKS[limit], rtol=1e-9)


def test_kopparapu_solar_edges():
    """The effective-flux limits reproduce the published solar distances."""
    # Kopparapu et al. (2014): 0.95 / 1.68 AU conservative, 0.75 / 1.77 AU
    # optimistic for a 1 Earth-mass planet around the Sun.
    inner_au, outer_au = kopparapu_habitable_zone_au(1.0, 5780.0)
    assert float(inner_au) == pytest.approx(0.95, abs=0.005)
    assert float(outer_au) == pytest.approx(1.68, abs=0.005)
    inner_au, outer_au = kopparapu_habitable_zone_au(1.0, 5780.0, optimistic=True)
    assert float(inner_au) == pytest.approx(0.75, abs=0.005)
    assert float(outer_au) == pytest.approx(1.77, abs=0.005)


def test_kopparapu_edges_scale_with_root_luminosity_at_fixed_temperature():
    """At fixed Teff the flux-based edges scale with sqrt(L / L_sun)."""
    inner_1, outer_1 = kopparapu_habitable_zone_au(1.0, 4500.0)
    inner_9, outer_9 = kopparapu_habitable_zone_au(9.0, 4500.0)
    np.testing.assert_allclose(float(inner_9), 3.0 * float(inner_1), rtol=1e-14)
    np.testing.assert_allclose(float(outer_9), 3.0 * float(outer_1), rtol=1e-14)


def test_cooler_hosts_move_the_flux_limits_down():
    """Each limit sits at a lower stellar flux around a cooler host."""
    # Redder starlight is absorbed more efficiently by H2O and CO2 and
    # Rayleigh-scatters less, so a planet around a cooler star reaches each
    # limit at a lower stellar flux (Kopparapu et al. 2013).
    teff_k = jnp.array([7200.0, 5780.0, 4500.0, 3500.0, 2600.0])
    for limit in KOPPARAPU_SEFF_COEFFICIENTS:
        flux = kopparapu_effective_flux(teff_k, limit)
        assert bool(jnp.all(jnp.diff(flux) < 0)), limit


def test_kopparapu_vmap_over_stars_matches_broadcasting():
    """A vmap over (L, Teff) pairs agrees with plain broadcasting."""
    luminosities_lsun = jnp.array([0.02, 0.3, 1.0, 3.0])
    teffs_k = jnp.array([3200.0, 4700.0, 5780.0, 6800.0])
    inner_v, outer_v = jax.vmap(kopparapu_habitable_zone_au)(luminosities_lsun, teffs_k)
    inner_b, outer_b = kopparapu_habitable_zone_au(luminosities_lsun, teffs_k)
    np.testing.assert_allclose(inner_v, inner_b, rtol=1e-14)
    np.testing.assert_allclose(outer_v, outer_b, rtol=1e-14)
    assert bool(jnp.all(inner_v < outer_v))


def test_unknown_kopparapu_limit_raises():
    """An unknown limit name is rejected."""
    with pytest.raises(KeyError):
        kopparapu_effective_flux(5780.0, "moist_greenhouse")
