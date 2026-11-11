"""The population draw matches the ExoVista planet generator in distribution.

Both draw the same star list: 50 ExoVista worlds against 200 worlds of the draw,
2000 and 8000 star-draws. Every comparison is a two-sample test at a fixed seed with
a failure threshold of p < 1e-3, so a pass means no difference is visible at that
sample size, not that the two are identical. The ExoVista generator needs the
`exovista` package, which is not on PyPI; the module is skipped where it is absent.
"""

import contextlib
import importlib.util
import io
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from skyscapes.population import (
    draw_populations,
    earth_like,
    load_occurrence_table,
    load_planet_types,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("exovista") is None,
        reason="the exovista package is not installed",
    ),
]

N_STARS, N_EXOVISTA, N_DRAW, P_MIN = 40, 50, 200, 1e-3
LUMINOSITY = np.linspace(0.25, 1.6, N_STARS)
MASS = 0.6 + 0.4 * LUMINOSITY**0.25


def _two_sample_chi2(a, b, min_count=20):
    """Two-sample chi-square p-value on counts, merging bins below `min_count`."""
    merged_a, merged_b, acc_a, acc_b = [], [], 0, 0
    for x, y in zip(a, b, strict=True):
        acc_a, acc_b = acc_a + x, acc_b + y
        if acc_a + acc_b >= min_count:
            merged_a.append(acc_a)
            merged_b.append(acc_b)
            acc_a = acc_b = 0
    merged_a[-1] += acc_a
    merged_b[-1] += acc_b
    a, b = np.array(merged_a, float), np.array(merged_b, float)
    k = np.sqrt(b.sum() / a.sum())
    chi2 = ((a * k - b / k) ** 2 / (a + b)).sum()
    return stats.chi2.sf(chi2, len(a) - 1)


@pytest.fixture(scope="module")
def exovista_worlds():
    from exovista import Settings, generate_planets
    from exovista.constants import pllabel

    names = list(load_planet_types().names)
    planets, types = [], []
    for w in range(N_EXOVISTA):
        stars = pd.DataFrame({"Lstar": LUMINOSITY, "mass": MASS})
        settings = Settings.Settings(seed=500 + w)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            # the generator assigns through a chained pandas indexer
            warnings.simplefilter("ignore")
            pl, albedos = generate_planets(stars, settings, usebins=True)
        type_index = np.full(pl.shape[:2], -1)
        for i, row in enumerate(albedos):
            for j, name in enumerate(row):
                type_index[i, j] = names.index(name)
        planets.append(pl)
        types.append(type_index)
    cols = {name: pllabel.index(name) for name in pllabel}
    return np.array(planets), np.array(types), cols


@pytest.fixture(scope="module")
def drawn():
    occurrence, types = load_occurrence_table(), load_planet_types()
    run = eqx.filter_jit(draw_populations)
    worlds = run(
        jax.random.key(0),
        N_DRAW,
        occurrence,
        types,
        jnp.asarray(LUMINOSITY),
        jnp.asarray(MASS),
    )
    return jax.tree_util.tree_map(np.asarray, worlds)


def _exovista_arrays(exovista_worlds):
    planets, types, cols = exovista_worlds
    present = planets[..., cols["R"]] > 0
    a_scaled = planets[..., cols["a"]] / np.sqrt(LUMINOSITY)[None, :, None]
    return {
        "present": present,
        "a_scaled": a_scaled[present],
        "mass": planets[..., cols["M"]][present],
        "radius": planets[..., cols["R"]][present],
        "inc": planets[..., cols["i"]][present],
        "e": planets[..., cols["e"]][present],
        "type": types[present],
        "class1": np.asarray(
            earth_like(
                planets[..., cols["a"]],
                planets[..., cols["R"]],
                LUMINOSITY[None, :, None],
            )
        )
        & present,
    }


def test_grid_equals_the_exovista_occurrence_grid():
    """The vendored tables rebuild ExoVista's occurrence grid cell for cell.

    Fails if the table build departs from the generator's convolution or bin edges.
    """
    from exovista.generate_planets import load_occurrence_rates

    rate, m_edge, _, a_edge, *_ = load_occurrence_rates(usebins=True)
    occurrence = load_occurrence_table()
    np.testing.assert_allclose(np.asarray(occurrence.rate), rate, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(occurrence.mass_edges_mearth), m_edge, rtol=1e-14
    )
    np.testing.assert_allclose(np.asarray(occurrence.a_edges_au), a_edge, rtol=1e-14)


def test_distributions_match_the_exovista_generator(exovista_worlds, drawn):
    """Planets per star, a, mass, inclination, eccentricity, type and class-1 match.

    The sample is 2000 ExoVista and 8000 draw star-draws; failure is p < 1e-3.
    Fails if the draw's occurrence weighting, bin forcing, stability pass or
    refill changes the population the generator produces.
    """
    ev = _exovista_arrays(exovista_worlds)
    occurrence = load_occurrence_table()
    present = drawn.present
    a_scaled = drawn.a_au / np.sqrt(LUMINOSITY)[None, :, None]
    ev_count, my_count = ev["present"].sum(2).ravel(), present.sum(2).ravel()
    counts = [np.bincount(c, minlength=31) for c in (ev_count, my_count)]
    assert _two_sample_chi2(*counts) > P_MIN

    for ev_x, my_x in (
        (np.log(ev["a_scaled"]), np.log(a_scaled[present])),
        (np.log(ev["mass"]), np.log(drawn.mass_mearth[present])),
        (ev["inc"], drawn.inc_deg[present]),
    ):
        assert stats.ks_2samp(ev_x, my_x).pvalue > P_MIN

    # planets per planet bin
    def bin_counts(a, m):
        ia = np.clip(np.searchsorted(np.asarray(occurrence.a_edges_au), a) - 1, 0, None)
        im = np.clip(
            np.searchsorted(np.asarray(occurrence.mass_edges_mearth), m) - 1, 0, None
        )
        b = np.asarray(occurrence.bin_id)[ia, im]
        return np.bincount(b[b >= 0], minlength=occurrence.n_bins)

    assert (
        _two_sample_chi2(
            bin_counts(ev["a_scaled"], ev["mass"]),
            bin_counts(a_scaled[present], drawn.mass_mearth[present]),
        )
        > P_MIN
    )

    assert ev["e"].max() == 0.0 and drawn.e.max() == 0.0
    types = [
        np.bincount(t, minlength=15) for t in (ev["type"], drawn.type_index[present])
    ]
    assert _two_sample_chi2(*types) > P_MIN

    ev_class1 = ev["class1"].sum(2).ravel()
    my_class1 = (
        (
            np.asarray(
                earth_like(drawn.a_au, drawn.radius_rearth, LUMINOSITY[None, :, None])
            )
            & present
        )
        .sum(2)
        .ravel()
    )
    sigma = np.sqrt(ev_class1.var() / ev_class1.size + my_class1.var() / my_class1.size)
    assert abs(ev_class1.mean() - my_class1.mean()) < 4.0 * sigma
