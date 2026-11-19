"""Draw and reference-measure density of the per-star occurrence process."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import priorities
import pytest
from hwoutils.epoch import Epoch

from skyscapes.population import (
    HostStars,
    OccurrencePopulation,
    PlanetTypes,
    PopulationConfig,
    occurrence_table_from_grid,
)
from skyscapes.scene import FlatStar


def _star_relative_a(s):
    from skyscapes.population._process import _relative_orbits

    r, v, mass = _relative_orbits(s)
    from orbix.system.core import GM_SUN_AU3_PER_D2

    mu = GM_SUN_AU3_PER_D2 * (s.star_mass_Msun + mass)
    return 1 / (2 / jnp.linalg.norm(r, axis=-1) - jnp.sum(v * v, axis=-1) / mu)


_read_a = jax.jit(jax.vmap(jax.vmap(_star_relative_a)))


def _hosts():
    return HostStars(
        jnp.array([11, 27, 39], jnp.int32),
        jnp.array([1.0, 0.8, 1.2]),
        jnp.array([1.0, 0.5, 2.0]),
        jnp.array([10.0, 12.0, 15.0]),
        FlatStar(jnp.ones(3), jnp.zeros(3)),
    )


@pytest.fixture(scope="module")
def process():
    table = occurrence_table_from_grid(
        [[0.8, 0.2], [0.4, 0.6]], [0.5, 2.0, 8.0], [0.5, 2.0, 8.0], [[0, 0], [0, 0]]
    )
    types = PlanetTypes(
        names=("Mars", "bright", "candidate"),
        radius_min_Rearth=jnp.array([0.0, 0.0, 0.8]),
        radius_max_Rearth=jnp.array([100.0, 100.0, 1.4]),
        a_min_AU=jnp.array([0.0, 0.0, 0.95]),
        a_max_AU=jnp.array([100.0, 100.0, 1.67]),
        weight=jnp.array([1.0, 3.0, 2.0]),
        earth_candidate=jnp.array([False, False, True]),
        wavelength_nm=jnp.array([400.0, 600.0, 800.0]),
        albedo=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.3, 0.2], [0.2, 0.25, 0.3]]),
    )
    return OccurrencePopulation(
        table,
        types,
        PopulationConfig(
            eccentricity=priorities.Uniform(0.1, 0.4),
            max_planets=12,
            mutual_hill_min=0.0,
            earth_candidate_probability=0.4,
        ),
        Epoch(),
    )


def test_mark_density_gradient(process):
    """Fails if density is disconnected from orbit leaves or uses the wrong measure.

    A central difference at h=1e-5 permits 1e-7 relative roundoff/truncation.
    """
    host = jax.tree.map(lambda x: x[:1], _hosts())
    scene = eqx.filter_jit(process.mark_sample)(jax.random.key(0), 8, host)

    def loss(ex):
        changed = eqx.tree_at(lambda s: s.dynamics.orbits.ex, scene, ex)
        return process.log_intensity(changed, host).sum()

    x = scene.dynamics.orbits.ex
    derivative = jax.jit(jax.grad(loss))(x)
    step = jnp.full_like(x, 1e-5)
    difference = (loss(x + step) - loss(x - step)) / 2e-5
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative.sum(), difference, rtol=1e-7)


@pytest.fixture(scope="module")
def marks(process):
    host = jax.tree.map(lambda x: x[:1], _hosts())
    scene = eqx.filter_jit(process.mark_sample)(jax.random.key(0), 1536, host)
    return host, scene


def test_mark_marginals_and_normalization(process, marks):
    """Fails if cell weights, isotropy, mass Jacobian, or normalization drift.

    Four 8-bin chi-square tests use false-rejection probability 1e-4 each.
    N=1536 resolves order 4/sqrt(N)=0.102 marginal errors. The independent
    uniform-classical-box importance estimate uses its measured 4-sigma
    standard error (not the sampler itself as a proposal).
    """
    from hwoutils.constants import Mearth2kg, Msun2kg
    from orbix.system.charts import ClassicalChart
    from orbix.system.core import GM_SUN_AU3_PER_D2
    from orbix.system.frame_orbit import FrameOrbit
    from scipy.stats import chi2

    host, scene = marks
    elem = scene.dynamics.orbits.to_classical()
    mass = np.asarray(scene.dynamics.planets.mass_Msun[:, 0]) * Msun2kg / Mearth2kg
    values = [
        (np.log(np.asarray(elem["a_AU"][:, 0])), np.log([0.5, 8.0])),
        (np.log(mass), np.log([0.5, 8.0])),
        (np.cos(np.asarray(elem["i_rad"][:, 0])), [-1.0, 1.0]),
        (np.asarray(elem["e"][:, 0]), [0.1, 0.4]),
    ]
    for k, (value, bounds) in enumerate(values):
        observed = np.histogram(value, np.linspace(*bounds, 9))[0]
        probabilities = np.full(8, 1 / 8)
        if k == 1:
            probabilities = np.repeat([0.6 / 4, 0.4 / 4], 4)
        expected = len(value) * probabilities
        assert np.sum((observed - expected) ** 2 / expected) < chi2.ppf(1 - 1e-4, 7)
    from skyscapes.population import mass_to_radius
    from skyscapes.population._draw import _type_weights

    probabilities = _type_weights(
        process.types,
        process.config,
        elem["a_AU"][:, 0],
        mass_to_radius(jnp.asarray(mass)),
    )
    probabilities = np.asarray(
        probabilities / probabilities.sum(axis=-1, keepdims=True)
    )
    labels = np.asarray(scene.planets[0].physical_model.type_index[:, 0])
    for index in range(3):
        expected = probabilities[:, index].sum()
        sigma = np.sqrt(np.sum(probabilities[:, index] * (1 - probabilities[:, index])))
        assert abs(np.sum(labels == index) - expected) < 4 * sigma
    # A proper sub-box of the first occurrence cell, with uniform proposal in
    # a,e,i,W,w,M,m, not log-scaled or isotropic coordinates.
    rng = np.random.default_rng(0)
    low = np.array([0.7, 0.15, 0.3, 0.2, 0.1, 0.4, 0.7])
    high = np.array([1.7, 0.35, 2.5, 5.5, 5.8, 5.9, 1.7])
    proposal = rng.uniform(low, high, (1536, 7))
    a, e, inc, node, peri, mean, m = proposal.T
    orbit = FrameOrbit.from_classical(
        a[:, None],
        e[:, None],
        inc[:, None],
        node[:, None],
        peri[:, None],
        mean[:, None],
        mu_AU3_per_d2=GM_SUN_AU3_PER_D2 * (1 + m[:, None] * Mearth2kg / Msun2kg),
    )
    box = eqx.tree_at(
        lambda s: (s.dynamics.orbits, s.dynamics.planets.mass_Msun),
        scene,
        (orbit, jnp.asarray(m[:, None] * Mearth2kg / Msun2kg)),
    )

    # Sum counting measure over all type labels; low-probability types cannot
    # be silently omitted from the normalization check.
    def total_density(box):
        def at_type(index):
            typed = eqx.tree_at(
                lambda s: s.planets[0].physical_model.type_index,
                box,
                jnp.full_like(box.valid, index, dtype=jnp.int32),
            )
            return jnp.exp(process.mark_log_density(typed, host))

        return jax.vmap(at_type)(jnp.arange(len(process.types.names))).sum(axis=0)

    density = jax.jit(total_density)(box)[:, 0]
    x = ClassicalChart().from_orbit(orbit)
    # Pull dnu back to ordinary a,e,i,W,w,M: exp(logV)/(a*e*(1-e))
    # times sin(i)/(1-cos(i)^2), with the coordinate Jacobians explicit.
    measure = (
        jnp.exp(ClassicalChart().log_volume(x))[:, 0]
        / (a * e * (1 - e))
        * np.sin(inc)
        / (1 - np.cos(inc) ** 2)
    )
    weights = np.asarray(density * measure) * np.prod(high - low)
    known = (
        0.8
        / 2
        * np.log(1.7 / 0.7) ** 2
        / np.log(4) ** 2
        * (0.35 - 0.15)
        / 0.3
        * (np.cos(0.3) - np.cos(2.5))
        / 2
        * np.prod(high[3:6] - low[3:6])
        / (2 * np.pi) ** 3
    )
    assert abs(weights.mean() - known) < 4 * weights.std(ddof=1) / np.sqrt(len(weights))
    assert np.isfinite(eqx.filter_jit(process.mark_log_density)(scene, host)).all()


def test_chart_change_of_variables(process, marks):
    """Fails if a chart Jacobian changes the reference density.

    The independent six-dimensional Jacobian permits 2e-12 absolute error
    from float64 rotations/inverse angles and determinant evaluation.
    """
    from orbix.system.charts import ClassicalChart, EccentricityVectorChart

    host, scenes = marks
    scene = jax.tree.map(lambda x: x[:8], scenes)
    classical, vector = ClassicalChart(), EccentricityVectorChart()
    orbit = scene.dynamics.orbits
    x, y = classical.from_orbit(orbit), vector.from_orbit(orbit)

    def change(z):
        return classical.from_orbit(vector.to_orbit(z, mu_AU3_per_d2=1.0))

    jacobian = jax.jit(jax.vmap(jax.jacfwd(change)))(y[:, 0])
    determinant = jnp.linalg.slogdet(jacobian)[1][:, None]
    density = eqx.filter_jit(process.mark_log_density)(scene, host)
    q_classical = density + classical.log_volume(x)
    q_vector = q_classical + determinant
    np.testing.assert_allclose(
        q_vector - vector.log_volume(y), density, atol=2e-12, rtol=0
    )

    # Non-negligible planet masses make direct-Jacobi evaluation visibly wrong.
    table = eqx.tree_at(
        lambda t: t.mass_edges_Mearth,
        process.occurrence,
        process.occurrence.mass_edges_Mearth * 100,
    )
    massive = eqx.tree_at(lambda p: p.occurrence, process, table)
    rows = dict(
        a_AU=jnp.array([1.0, 4.0]),
        mass=jnp.array([100.0, 400.0]),
        e=jnp.array([0.2, 0.3]),
        i_rad=jnp.array([0.6, 0.8]),
        W_rad=jnp.array([0.2, 0.4]),
        w_rad=jnp.array([0.7, 0.9]),
        M0_rad=jnp.array([0.3, 0.5]),
        type_index=jnp.array([0, 1]),
    )
    star = jax.tree.map(lambda x: x[0], host.star)

    def build(rows):
        joint = massive._system(rows, jnp.ones(2, bool), star, 1.0, 10.0)
        separate = jax.vmap(
            lambda row: massive._system(
                jax.tree.map(lambda x: x[None], row), jnp.ones(1, bool), star, 1.0, 10.0
            )
        )(rows)
        return massive.mark_log_density(joint, host), massive.mark_log_density(
            separate, host
        )

    joint, separate = jax.jit(build)(rows)
    np.testing.assert_allclose(joint, separate[:, 0], rtol=0, atol=2e-12)


def test_support_errors_and_type_photometry(process, marks):
    """Fails if support, padding, degeneracy, or type spectra are ignored.

    Albedo and photometry use the same interpolation and permit 16 float64
    eps for fused phase/radius arithmetic; support and masks are exact.
    Zero-width tilt is tested because isotropic mixing removes its singularity.
    """
    from skyscapes.physical_model import LambertianPhysicalModel

    host, scenes = marks
    scene = jax.tree.map(lambda x: x[:8], scenes)
    outside = eqx.tree_at(
        lambda s: s.dynamics.orbits.a_AU,
        scene,
        jnp.full_like(scene.dynamics.orbits.a_AU, 100.0),
    )
    density = eqx.filter_jit(process.mark_log_density)
    assert np.isneginf(density(outside, host)).all()
    padded = eqx.tree_at(lambda s: s.valid, outside, jnp.zeros_like(outside.valid))
    np.testing.assert_array_equal(density(padded, host), 0.0)
    np.testing.assert_array_equal(process.log_intensity(padded, host), 0.0)
    degenerate = eqx.tree_at(lambda p: p.config, process, PopulationConfig())
    with pytest.raises(ValueError, match=r"eccentricity is None"):
        degenerate.mark_log_density(scene, host)
    with pytest.raises(ValueError, match=r"eccentricity is None"):
        eqx.filter_jit(degenerate.mark_log_density)(scene, host)
    zero_tilt = eqx.tree_at(lambda p: p.config.inc_max_deg, process, 0.0)
    np.testing.assert_array_equal(
        zero_tilt.mark_log_density(scene, host), process.mark_log_density(scene, host)
    )
    single = jax.tree.map(lambda x: x[0], scene)
    assert process.log_intensity(single, host).shape == (1,)
    np.testing.assert_allclose(
        process.log_intensity(single, host),
        process.mark_log_density(single, host) + process.log_rate(host)[0],
    )
    albedo = process.type_albedo(scene, 600.0)
    actual = jax.jit(lambda s: s.contrasts(jnp.asarray(600.0), jnp.array([0.0])))(scene)
    # Verify the physical-model hook independently of the population accessor.
    geom = single.ephemeris(jnp.array([0.0]))
    expected = LambertianPhysicalModel(albedo[0]).contrast(
        geom.phase_angle_rad(jnp.array([1]), 0),
        geom.distance_AU(jnp.array([1]), 0),
        jnp.asarray(600.0),
        single.planets[0].Rp_Rearth,
    )
    np.testing.assert_allclose(actual[0], expected, rtol=16 * np.finfo(float).eps)
    assert np.isfinite(actual).all()


@pytest.fixture(scope="module")
def worlds(process):
    run = eqx.filter_jit(lambda p, key, stars: p.sample(key, stars, 1024))
    unpruned = run(process, jax.random.key(0), _hosts())
    pruned_process = eqx.tree_at(lambda p: p.config.mutual_hill_min, process, 6.0)
    pruned = run(pruned_process, jax.random.key(0), _hosts())
    return unpruned, pruned


def test_poisson_counts(process, worlds):
    """Fails if counts are fixed totals, tied across hosts, or rate is scaled by L.

    N=1024 worlds gives standard error sqrt(rate/N); the acceptance band is
    four Poisson sigma per star/cell. Off-diagonal covariances permit four
    standard errors sqrt(rate_i*rate_j/N). K=12 has negligible loss at rate=2.
    """
    scene, _ = worlds
    counts = np.asarray(scene.valid.sum(axis=-1))
    expected = np.asarray(process.rate(_hosts()))
    assert np.all(abs(counts.mean(axis=0) - expected) < 4 * np.sqrt(expected / 1024))
    np.testing.assert_allclose(
        np.exp(process.log_rate(_hosts())), expected, rtol=2 * np.finfo(float).eps
    )
    assert np.all(counts.var(axis=0) > 0.7 * expected)

    covariance = np.cov(counts, rowvar=False)
    off_diagonal = ~np.eye(3, dtype=bool)
    assert np.all(abs(covariance[off_diagonal]) < 4 * 2 / np.sqrt(1024))
    from hwoutils.constants import Mearth2kg, Msun2kg

    a = (
        np.asarray(_read_a(scene))
        / np.sqrt(np.asarray(_hosts().luminosity_Lsun))[None, :, None]
    )
    m = np.asarray(scene.dynamics.planets.mass_Msun) * Msun2kg / Mearth2kg
    cell = 2 * (a >= 2).astype(int) + (m >= 2).astype(int)
    per_cell = np.stack(
        [np.sum((cell == i) & np.asarray(scene.valid), axis=-1) for i in range(4)],
        axis=-1,
    )
    rates = np.asarray(process.occurrence.rate).ravel()
    assert np.all(abs(per_cell.mean(axis=0) - rates) < 4 * np.sqrt(rates / 1024))
    covariance = np.cov(per_cell.reshape(-1, 4), rowvar=False)
    tolerance = 4 * np.sqrt(rates[:, None] * rates[None, :] / 3072)
    assert np.all(
        abs(covariance[~np.eye(4, dtype=bool)]) < tolerance[~np.eye(4, dtype=bool)]
    )


def _equal(a, b, *, bits=False):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if bits:
            x, y = np.asarray(x), np.asarray(y)
            assert x.dtype == y.dtype
            np.testing.assert_array_equal(
                np.ascontiguousarray(x).view(np.uint8),
                np.ascontiguousarray(y).view(np.uint8),
            )
        else:
            np.testing.assert_array_equal(x, y)


def test_star_and_world_identity(process):
    """Fails if array position or requested world count enters a host's key.

    These are identical random draws: comparisons require bitwise equality.
    """
    run = eqx.filter_jit(process.sample)
    stars = _hosts()
    three = run(jax.random.key(0), stars, 3)
    five = run(jax.random.key(0), stars, 5)
    _equal(three, jax.tree.map(lambda x: x[:3], five), bits=True)
    selected = jnp.array([2, 0])
    subset = run(jax.random.key(0), jax.tree.map(lambda x: x[selected], stars), 3)
    _equal(subset, jax.tree.map(lambda x: x[:, selected], three), bits=True)
    reverse = run(jax.random.key(0), jax.tree.map(lambda x: x[::-1], stars), 3)
    _equal(reverse, jax.tree.map(lambda x: x[:, ::-1], three), bits=True)


def test_structure_padding_and_compile(process, marks):
    """Fails if mass is photometric only, padding is active, or values recompile.

    One fixed (S,K) program is reused exactly. Recovered two-body radii allow
    64 eps for orbital state conversions; padding values must match exactly.
    """
    from hwoutils.constants import Mearth2kg, Msun2kg

    from skyscapes.population import mass_to_radius
    from skyscapes.scene._system_batch import _PAD_ORBIT

    run = jax.jit(lambda key, stars: process.sample(key, stars, 4))
    scene = run(jax.random.key(0), _hosts())
    run(jax.random.fold_in(jax.random.key(0), 1), _hosts())
    assert run._cache_size() == 1
    assert scene.batch_shape == (4, 3) and scene.capacity == 12
    valid = np.asarray(scene.valid)
    assert np.all(np.diff(valid.astype(int), axis=-1) <= 0)
    a = np.asarray(scene.dynamics.orbits.a_AU)
    assert np.all(np.diff(a, axis=-1)[valid[..., 1:]] > 0)
    from orbix.system.rotation import rotate

    normal = np.asarray(rotate(scene.dynamics.orbits.quat, jnp.array([0.0, 0.0, 1.0])))
    dots = np.einsum("...ik,...jk->...ij", normal, normal)
    pairs = valid[..., :, None] & valid[..., None, :]
    # Every pair is at most twice the largest tilt from their shared midplane.
    assert np.all(dots[pairs] >= np.cos(np.deg2rad(10.0)) - 64 * np.finfo(float).eps)
    mass = scene.dynamics.planets.mass_Msun * Msun2kg / Mearth2kg
    np.testing.assert_allclose(
        np.asarray(mass_to_radius(mass))[valid],
        np.asarray(scene.planets[0].Rp_Rearth)[valid],
        rtol=64 * np.finfo(float).eps,
    )
    np.testing.assert_array_equal(np.asarray(mass)[~valid], 0.0)
    np.testing.assert_array_equal(np.asarray(scene.planets[0].Rp_Rearth)[~valid], 0.0)
    for path, value in jax.tree_util.tree_flatten_with_path(scene.dynamics.orbits)[0]:
        np.testing.assert_array_equal(
            np.asarray(value)[~valid],
            np.broadcast_to(_PAD_ORBIT[path[-1].name], np.asarray(value)[~valid].shape),
        )
    # Kinematic rates must include each enclosed planet, not just the host.
    enclosed = scene.star_mass_Msun[..., None] + jnp.cumsum(
        scene.dynamics.planets.mass_Msun, axis=-1
    )
    from orbix.system.core import GM_SUN_AU3_PER_D2

    np.testing.assert_allclose(
        np.asarray(scene.dynamics.orbits.mu_AU3_per_d2)[valid],
        np.asarray(enclosed * GM_SUN_AU3_PER_D2)[valid],
        rtol=64 * np.finfo(float).eps,
    )

    from skyscapes.scene import Scene

    host, singles = marks
    single = jax.tree.map(lambda x: x[0], singles)
    packed = Scene.from_systems([single], capacity=3)
    restored = packed.to_systems()[0]
    _equal(restored, single)
    density = eqx.filter_jit(process.mark_log_density)
    np.testing.assert_allclose(
        density(packed, host)[0, :1], density(single, host), atol=2e-12
    )
    read = jax.jit(lambda s: s.contrasts(jnp.asarray(600.0), jnp.array([0.0])))
    np.testing.assert_allclose(
        read(packed)[0, :1], read(single), rtol=16 * np.finfo(float).eps
    )


def test_pruning_residual(process, worlds):
    """Fails if Hill pruning refills or causes large unexplained marginal changes.

    On this small rate=2 table, loose bounds are 35% removed and 0.1 maximum
    CDF shift in ln(a); these characterize pruning, not Monte Carlo precision.
    """
    from scipy.stats import ks_2samp

    from skyscapes.population._draw import _remove_unstable

    unpruned, pruned = worlds
    before, after = int(unpruned.valid.sum()), int(pruned.valid.sum())
    removed = 1 - after / before

    # The stored Jacobi semimajor axes differ slightly from star-relative a;
    # recover the mark's scale for the declared residual.
    a, b = _read_a(unpruned), _read_a(pruned)
    shift = ks_2samp(
        np.log(np.asarray(a)[unpruned.valid]), np.log(np.asarray(b)[pruned.valid])
    ).statistic
    print(f"pruning removed={removed:.6f}; max ln(a) CDF shift={shift:.6f}")
    assert 0 < removed < 0.35 and shift < 0.1
    # The specified outer-member mass tie must survive reuse of the helper.
    alive = _remove_unstable(
        jnp.array([1.0, 1.001]), jnp.ones(2), jnp.ones(2, bool), 999000.0, 6.0
    )
    np.testing.assert_array_equal(alive, [True, False])


def test_capacity_keeps_innermost(process):
    """Fails if capacity selection is random or happens before the stability sweep.

    Identical candidates imply matching selected star-relative marks; 2e-13
    accounts for Jacobi inverse roundoff when different tails are omitted.
    """
    key = jax.random.key(0)
    large = process
    small = eqx.tree_at(
        lambda p: p.config, process, replace(process.config, max_planets=1)
    )
    run = eqx.filter_jit(lambda p: p.sample(key, _hosts(), 24))
    a, b = run(large), run(small)
    assert np.any(a.valid.sum(axis=-1) > 1)
    np.testing.assert_array_equal(b.valid[..., 0], a.valid[..., 0])
    np.testing.assert_allclose(
        b.dynamics.orbits.a_AU[..., 0], a.dynamics.orbits.a_AU[..., 0], rtol=2e-13
    )
    np.testing.assert_array_equal(
        b.dynamics.planets.mass_Msun[..., 0], a.dynamics.planets.mass_Msun[..., 0]
    )

    unpruned_mass = a.dynamics.planets.mass_Msun[..., 0]
    pruned_large = eqx.tree_at(lambda p: p.config.mutual_hill_min, large, 6.0)
    pruned_small = eqx.tree_at(lambda p: p.config.mutual_hill_min, small, 6.0)
    a, b = run(pruned_large), run(pruned_small)
    assert np.any(a.dynamics.planets.mass_Msun[..., 0] != unpruned_mass)
    np.testing.assert_allclose(
        b.dynamics.orbits.a_AU[..., 0], a.dynamics.orbits.a_AU[..., 0], rtol=2e-13
    )
    np.testing.assert_array_equal(
        b.dynamics.planets.mass_Msun[..., 0], a.dynamics.planets.mass_Msun[..., 0]
    )


def test_mark_coordinate_round_trip(process, marks):
    """Fails if rebuilding loses mass, type, epoch state, or batch axes.

    State and density allow 256 ulps at unit scale (256*eps*(1+abs(x)))
    for Cartesian/frame transformations; integer labels and masks are exact.
    Scalar, vector and two-axis batches use the same eight sampled marks.
    """
    host, scenes = marks
    small = jax.tree.map(lambda x: x[:8], scenes)

    @eqx.filter_jit
    def rebuild(scene):
        return process.mark_build(host, *process.mark_read(scene))

    for shape in ((), (8,), (2, 4)):
        scene = (
            jax.tree.map(lambda x: x[0], small)
            if not shape
            else jax.tree.map(
                lambda x, shape=shape: x.reshape((*shape, *x.shape[1:])), small
            )
        )
        rebuilt = rebuild(scene)
        assert rebuilt.batch_shape == shape
        for before, after in zip(
            jax.tree.leaves(scene), jax.tree.leaves(rebuilt), strict=True
        ):
            if jnp.issubdtype(jnp.asarray(before).dtype, jnp.inexact):
                np.testing.assert_allclose(
                    before,
                    after,
                    rtol=256 * np.finfo(float).eps,
                    atol=256 * np.finfo(float).eps,
                )
            else:
                np.testing.assert_array_equal(before, after)
        np.testing.assert_allclose(
            process.mark_log_density(scene, host),
            process.mark_log_density(rebuilt, host),
            rtol=0,
            atol=256 * np.finfo(float).eps,
        )
        padded = eqx.tree_at(lambda s: s.valid, scene, jnp.zeros_like(scene.valid))
        read = eqx.filter_jit(process.mark_read)(padded)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(read))
        assert read[2].dtype == jnp.int32


def test_earth_like_flags(process, worlds):
    """Fails if the scene flag departs from the candidate box or marks padding.

    The flag is compared exactly with the box function applied to the
    star-relative semi-major axis, the mass-radius image and the host
    luminosity read independently from the same scenes.
    """
    from skyscapes.population import earth_like, mass_to_radius
    from skyscapes.population._process import _MASS_RATIO, _relative_orbits

    hosts, scenes = _hosts(), worlds[1]
    flag = np.asarray(eqx.filter_jit(process.earth_like)(scenes, hosts))
    a = np.asarray(_read_a(scenes))
    mass = np.asarray(
        jax.vmap(jax.vmap(lambda s: _relative_orbits(s)[2]))(scenes) / _MASS_RATIO
    )
    valid = np.asarray(scenes.valid)
    lum = np.asarray(hosts.luminosity_Lsun)[None, :, None]
    radius = np.asarray(mass_to_radius(np.where(valid, mass, 1.0)))
    expected = valid & np.asarray(earth_like(np.where(valid, a, 1.0), radius, lum))
    np.testing.assert_array_equal(flag, expected)
    assert flag.shape == valid.shape and flag.any() and not flag[~valid].any()
    assert (valid & ~flag).any()
