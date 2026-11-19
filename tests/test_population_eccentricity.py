"""Eccentricity distributions of populations and their use by the draw."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from priorities.testing import check_distribution
from scipy import stats

from skyscapes.population import (
    PopulationConfig,
    draw_population,
    kipping_eccentricity,
    load_occurrence_table,
    load_planet_types,
    rayleigh_eccentricity,
)


def test_constructor_passes_the_distribution_contract():
    """A truncated Rayleigh from the constructor satisfies the priorities contract.

    Fails if a constructor builds a distribution whose draws, cdf or density
    disagree. The Beta record has its own contract test in priorities.
    """
    law = rayleigh_eccentricity(0.3, e_max=0.6)
    check_distribution(law, jax.random.key(0), n_draws=1_000, rtol=1e-6)


def test_kipping_is_beta_and_truncation_conditions_it():
    """Kipping is Beta(0.867, 3.03) and e_max conditions it, matching scipy.

    Fails if the shapes are swapped or the cut is not renormalized. Tolerance
    1e-10 on log densities, floating-point.
    """
    e = jnp.array([0.05, 0.3, 0.9])
    reference = stats.beta(0.867, 3.03)
    np.testing.assert_allclose(
        kipping_eccentricity().log_prob(e), reference.logpdf(e), rtol=1e-10
    )
    cut = kipping_eccentricity(e_max=0.95).log_prob(e)
    np.testing.assert_allclose(
        cut[:2], reference.logpdf(e[:2]) - np.log(reference.cdf(0.95)), rtol=1e-10
    )
    assert np.isneginf(kipping_eccentricity(e_max=0.5).log_prob(e[2]))


@pytest.fixture(scope="module")
def tables():
    return load_occurrence_table(), load_planet_types()


def _draw(tables, config):
    occurrence, types = tables
    lum = jnp.linspace(0.3, 1.4, 8)
    return eqx.filter_jit(draw_population)(
        jax.random.key(0), occurrence, types, lum, 0.6 + 0.4 * lum**0.25, config
    )


def test_default_population_is_circular(tables):
    """With no eccentricity distribution every planet has e = 0.

    Fails if the None default draws eccentric orbits.
    """
    p = _draw(tables, None)
    assert np.all(np.asarray(p.e) == 0.0)


def test_draw_follows_the_eccentricity_law(tables):
    """Drawn eccentricities are consistent with the truncated Kipping law.

    Fails if the draw ignores the distribution or its cut. Kolmogorov-Smirnov
    distance below 2.69 / sqrt(n), a 1e-6 false-rejection rate, monte-carlo.
    """
    law = kipping_eccentricity(e_max=0.5)
    p = _draw(tables, PopulationConfig(eccentricity=law))
    e = np.sort(np.asarray(p.e)[np.asarray(p.present)])
    assert e.max() <= 0.5
    unit = np.asarray(law.cdf(jnp.asarray(e)))
    n = e.size
    rank = np.arange(n)
    distance = max(np.max((rank + 1) / n - unit), np.max(unit - rank / n))
    assert distance < 2.69 / np.sqrt(n)
