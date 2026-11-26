"""Scene.sources: node-effective star, planet and zodi values on one band rule.

Every truth here is a band integral of a source times a cubic response R,
computed independently of the operator: Gauss-Legendre points on every cell
between the merged table knots, which is exact for a piecewise polynomial of
the degrees involved, or ``scipy.integrate.quad`` for a Planck spectrum. A
cubic response is the highest degree a four-node piece integrates exactly, so
a wrong distribution of the source over the nodes shows; a flat response would
check only the total. The PHOENIX 5800 K table is the offline test fixture
(native 1427-knot sampling is block-averaged there to 57 knots in 500 to
555 nm and 36 in 735 to 785 nm), which still puts many knots inside every
piece.
"""

from pathlib import Path

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rearth2m, c, h, k_B, pc2m
from hwoutils.leaves import is_param
from hwoutils.model_paths import assign
from hwoutils.numerics import roundoff
from numpy.polynomial.legendre import leggauss
from optixstuff.grids import PixelGrid, SkyOrientation
from optixstuff.spectral import Band, Knots, Quadrature, Resolution
from scipy.integrate import quad
from zodi import DensityDust

from skyscapes.disk import GraterDisk
from skyscapes.io import from_phoenix
from skyscapes.physical_model import (
    BlackbodyPhysicalModel,
    GridPhysicalModel,
    LambertianPhysicalModel,
    PrecomputedPhysicalModel,
    SumPhysicalModel,
    TimeSeriesPhysicalModel,
)
from skyscapes.scene import BlackbodyStar, Planet, PrecomputedZodi, Scene, Star
from tests.test_physical_model_exojax import make_physical_model

DATA = Path(__file__).parent / "data"
BANDS = ((500.0, 555.0), (735.0, 785.0))
DIST_PC = 10.0
TIMES = jnp.zeros(1)
# A 1 nm albedo table with a 50 nm absorption band at 760 nm.
ALBEDO_NM = np.arange(450.0, 851.0, 1.0)
ALBEDO = 0.3 - 0.2 * np.clip(1 - np.abs(ALBEDO_NM - 760.0) / 25.0, 0, None)
# A second, coarser table whose knots fall between the first's.
RAMP_NM = np.arange(452.5, 850.0, 5.0)
RAMP = 0.05 + 0.02 * np.sin(RAMP_NM / 7.0)


def _fetch(teff_K):
    return str(DATA / f"phoenixm00_{teff_K}_test.fits")


def _phoenix(method="linear", flux_scale=1.0):
    star = from_phoenix(
        5800.0,
        4.5,
        dist_pc=DIST_PC,
        radius_m=6.957e8,
        wavelength_range_nm=(400.0, 900.0),
        fetch=_fetch,
    )
    return Star(
        wavelengths_nm=star.wavelengths_nm,
        times_d=jnp.zeros(1),
        flux_density_jy=star.flux_density_jy,
        luminosity_Lsun=star.luminosity_Lsun,
        interpolation_method=method,
        flux_scale=flux_scale,
    )


def _table_planet(knots, values):
    return TimeSeriesPhysicalModel(knots, np.zeros(1), values[None, :, None])


def _scene(star, models, **kwargs):
    count = len(models)
    return Scene.from_classical(
        star,
        tuple(Planet(jnp.ones(1), model) for model in models),
        star_mass_Msun=1.0,
        planet_mass_Msun=0.0,
        dist_pc=DIST_PC,
        a_AU=jnp.arange(1.0, count + 1.0),
        e=0.0,
        i_rad=0.5,
        W_rad=0.0,
        w_rad=0.0,
        M0_rad=0.3,
        **kwargs,
    )


@pytest.fixture(scope="module")
def rule():
    """Both E8 bands; the break at 520 nm gives the second band padded nodes."""
    return Quadrature.build([Band(*band) for band in BANDS], breaks_nm=(520.0,))


def _response(x, band):
    """A positive cubic across the band: degree Q - 1 of a four-node piece."""
    low, high = BANDS[band]
    u = (np.asarray(x) - (low + high) / 2) / ((high - low) / 2)
    return 1 + 0.3 * u - 0.2 * u**2 + 0.1 * u**3


def _cells(band, *knots, n=8):
    """Gauss points and weights, n per cell between the knots inside the band."""
    low, high = BANDS[band]
    inner = [
        np.asarray(k)[(np.asarray(k) > low) & (np.asarray(k) < high)] for k in knots
    ]
    edges = np.union1d(np.concatenate([[low, high], *inner]), [])
    x, w = leggauss(n)
    length = np.diff(edges)[:, None]
    return (edges[:-1, None] + length * (x + 1) / 2).ravel(), (length * w / 2).ravel()


def _band_sum(rule, values, band):
    """Weighted node sum of values times the cubic response in one band."""
    nodes = np.asarray(rule.nodes_nm)[band]
    weights = np.asarray(rule.weights_nm)[band]
    return np.sum(weights * np.asarray(values)[band] * _response(nodes, band))


def _responses(nodes):
    """The cubic response of each band on its row of nodes, shape (W, Q)."""
    return jnp.stack([jnp.asarray(_response(nodes[b], b)) for b in range(len(BANDS))])


def _star_at(star, x):
    return np.asarray(
        star.photon_flux_density(jnp.asarray(x), jnp.zeros(()), dist_pc=DIST_PC)
    )


@vv.case("stellar-photometry", "code-verification")
@pytest.mark.parametrize("method", ["linear", "cubic"])
@pytest.mark.float64_only(reason="float64 spectral quadrature reference")
def test_star_node_values_integrate_a_phoenix_table_exactly(rule, method):
    """Fails if the star's node values are point samples or misplaced on the nodes.

    Truth: the PHOENIX photon table times a cubic response, integrated with
    eight Gauss points on every cell between knots, exact for the degree-4
    (linear table) or degree-6 (cubic table) integrand on each cell; the star
    is evaluated through its own point interpolant, independently of the
    operator. Tolerance 1e-12 relative, basis floating-point (measured 2e-16).
    The scale (1.7) must be applied once. Mutation controls on the same
    values, each measured on this fixture: point samples (1e-2 and 3e-3),
    the band mean at every node (5e-3 and 2e-3) and node rows reversed within
    each piece (3e-4 and 5e-3) all miss by more than 1e-4.
    """
    star = _phoenix(method, flux_scale=1.7)
    scene = _scene(star, ())
    resolution = Resolution.build(rule, *scene.spectral_tables())
    sources = scene.sources(rule, resolution, TIMES)
    knots = np.asarray(star.wavelengths_nm)
    star_nodes = np.asarray(sources.star)[..., 0]
    nodes = np.asarray(rule.nodes_nm)
    for band in range(len(BANDS)):
        x, wx = _cells(band, knots)
        truth = np.sum(wx * _star_at(star, x) * _response(x, band))
        np.testing.assert_allclose(_band_sum(rule, star_nodes, band), truth, rtol=1e-12)
        point = _star_at(star, nodes)
        mean = np.full_like(
            star_nodes, np.sum(wx * _star_at(star, x)) / np.ptp(BANDS[band])
        )
        rows = star_nodes.copy()
        for start in range(0, 8 if band == 0 else 4, 4):
            rows[band, start : start + 4] = star_nodes[band, start : start + 4][::-1]
        for wrong in (point, mean, rows):
            assert abs(_band_sum(rule, wrong, band) / truth - 1) > 1e-4
    np.testing.assert_allclose(sources.unscaled_star * 1.7, sources.star, rtol=1e-15)
    # A padded node copies the first real node of its band.
    assert star_nodes[1, 4] == star_nodes[1, 0]


@vv.case("planet-phase", "code-verification")
@pytest.mark.float64_only(reason="float64 spectral quadrature reference")
def test_tabulated_planet_flux_is_reduced_with_the_star(rule):
    """Fails if a tabulated planet is its node albedo times the reduced star.

    Two planets on a 1 nm albedo table with an absorption band (the second
    twice as bright) and a composite of that table with a 5 nm table whose
    knots interleave, under the PHOENIX star, at two times. Truth: star times
    albedo times a cubic response, eight Gauss points on every cell between
    the merged star and albedo knots, exact for the degree-5 integrand. Rate
    and band contrast to 1e-10 relative, basis floating-point (measured
    4e-16). The node-albedo shortcut (the albedo sampled at the nodes times
    the reduced star) misses the absorbing band by 2.4e-2 on this fixture.
    """
    star = _phoenix()
    absorber = _table_planet(ALBEDO_NM, ALBEDO * 1e-9)
    pair = TimeSeriesPhysicalModel(
        ALBEDO_NM, np.zeros(1), np.stack([ALBEDO, 2 * ALBEDO])[:, :, None] * 1e-9
    )
    composite = SumPhysicalModel((absorber, _table_planet(RAMP_NM, RAMP * 1e-9)))
    scene = Scene.from_classical(
        star,
        (Planet(jnp.ones(2), pair), Planet(jnp.ones(1), composite)),
        star_mass_Msun=1.0,
        planet_mass_Msun=0.0,
        dist_pc=DIST_PC,
        a_AU=jnp.array([1.0, 2.0, 3.0]),
        e=0.0,
        i_rad=0.5,
        W_rad=0.0,
        w_rad=0.0,
        M0_rad=0.3,
    )
    times = jnp.array([0.0, 30.0])
    resolution = Resolution.build(rule, *scene.spectral_tables())
    sources = scene.sources(rule, resolution, times)
    contrasts = scene.band_contrasts(
        rule, times, transmission=_responses, resolution=resolution
    )
    knots = np.asarray(star.wavelengths_nm)
    planets = np.asarray(sources.planets)
    for band in range(len(BANDS)):
        x, wx = _cells(band, knots, ALBEDO_NM, RAMP_NM)
        weighted = wx * _star_at(star, x) * _response(x, band)
        albedo = np.interp(x, ALBEDO_NM, ALBEDO) * 1e-9
        ramp = np.interp(x, RAMP_NM, RAMP) * 1e-9
        truths = [np.sum(weighted * albedo), 2 * np.sum(weighted * albedo)]
        truths.append(np.sum(weighted * (albedo + ramp)))
        for k, truth in enumerate(truths):
            for t in range(times.size):
                np.testing.assert_allclose(
                    _band_sum(rule, planets[:, :, k, t], band), truth, rtol=1e-10
                )
                np.testing.assert_allclose(
                    contrasts[band, k, t], truth / np.sum(weighted), rtol=1e-10
                )
    nodes = np.asarray(rule.nodes_nm)
    shortcut = (
        np.asarray(sources.star)[..., 0] * np.interp(nodes, ALBEDO_NM, ALBEDO) * 1e-9
    )
    x, wx = _cells(1, knots, ALBEDO_NM)
    albedo = np.interp(x, ALBEDO_NM, ALBEDO) * 1e-9
    truth = np.sum(wx * _star_at(star, x) * albedo * _response(x, 1))
    assert abs(_band_sum(rule, shortcut, 1) / truth - 1) > 1e-2


@vv.case("planet-thermal-emission", "code-verification")
@pytest.mark.float64_only(reason="float64 spectral quadrature reference")
def test_thermal_planet_node_flux_ignores_the_star(rule):
    """Fails if a thermal planet's node flux is scaled by the star's reduction.

    A 3000 K blackbody planet of one Earth radius at 10 pc under the PHOENIX
    star. Truth: ``scipy.integrate.quad`` of pi times the Planck photon
    radiance times (Rp / d)^2 times a cubic response, at 1e-13 relative.
    Tolerance 1e-9, basis spike: order-4 Gauss on the smooth thermal flux
    (measured 5e-11). Doubling the star's table and scale leaves the thermal
    flux bitwise unchanged.
    """
    temperature = 3000.0
    scene = _scene(_phoenix(), (BlackbodyPhysicalModel(jnp.array([temperature])),))
    brighter = _scene(
        _phoenix(flux_scale=2.0), (BlackbodyPhysicalModel(jnp.array([temperature])),)
    )
    brighter = eqx.tree_at(
        lambda s: s.star._flux_density_phot,
        brighter,
        2 * brighter.star._flux_density_phot,
    )
    resolution = Resolution.build(rule, *scene.spectral_tables())
    flux = np.asarray(scene.sources(rule, resolution, TIMES).planets)[:, :, 0, 0]
    again = np.asarray(brighter.sources(rule, resolution, TIMES).planets)[:, :, 0, 0]
    np.testing.assert_array_equal(flux, again)
    dilution = np.pi * (Rearth2m / (DIST_PC * pc2m)) ** 2
    for band, (low, high) in enumerate(BANDS):

        def integrand(wave_nm, band=band):
            lam = wave_nm * 1e-9
            radiance = 2 * c / lam**4 / np.expm1(h * c / (lam * k_B * temperature))
            return radiance * 1e-9 * dilution * _response(wave_nm, band)

        truth = quad(integrand, low, high, epsrel=1e-13, limit=200)[0]
        np.testing.assert_allclose(_band_sum(rule, flux, band), truth, rtol=1e-9)


def test_tabulated_source_without_a_rule_is_refused(rule):
    """Fails if a table without an operator falls back to a point read.

    A PHOENIX scene on a resolution built for its local-zodi table only, and
    a composite planet on one built for the star only, raise a LookupError at
    trace time, eagerly and under ``jax.jit``. With every rule the zodi node
    radiance integrates its linear table against a cubic response within 32
    relative roundings for interpolation,
    cubic response and tree reductions over 100 cells and eight Gauss nodes.
    """
    star = _phoenix()
    zodi_nm = np.linspace(400.0, 900.0, 101)
    zodi = PrecomputedZodi(zodi_nm, 1e-6 * (1 + 0.5 * np.cos(zodi_nm / 13.0)))
    composite = SumPhysicalModel(
        (
            _table_planet(ALBEDO_NM, ALBEDO * 1e-9),
            LambertianPhysicalModel(jnp.array([0.3])),
        )
    )
    scene = _scene(star, (composite,), local_zodi=zodi)
    for partial, missing in (
        (Resolution.build(rule, zodi), "Star"),
        (Resolution.build(rule, star), "SumPhysicalModel"),
    ):
        with pytest.raises(LookupError, match=f"no rule for {missing}"):
            scene.sources(rule, partial, TIMES)
        with pytest.raises(LookupError, match=f"no rule for {missing}"):
            jax.jit(lambda s, r=partial: s.sources(rule, r, TIMES).star)(scene)
    resolution = Resolution.build(rule, *scene.spectral_tables())
    radiance = np.asarray(scene.sources(rule, resolution, TIMES).local_zodi)[..., 0]
    for band in range(len(BANDS)):
        x, wx = _cells(band, zodi_nm)
        truth = np.sum(
            wx
            * np.interp(x, zodi_nm, zodi.radiance_phot_per_s_m2_nm_sr)
            * _response(x, band)
        )
        np.testing.assert_allclose(
            _band_sum(rule, radiance, band),
            truth,
            rtol=float(roundoff(1.0, n_ops=32, dtype=radiance.dtype)),
        )


def test_knots_are_static_and_a_new_grid_has_no_rule(rule):
    """Fails if traced knots are accepted, a knot is a leaf, or an old rule is reused.

    Exact. A star built inside ``jax.jit`` from traced knots is refused (the
    knots must be concrete). The knots are static, so a fit request that
    names them (``hwoutils.model_paths.assign``, which a fit uses) is refused
    as not a leaf, the former knot path no longer exists, and no knot grid is
    among the scene's leaves. A star rebuilt on a
    grid moved by 0.01 nm has another digest, so the ``Resolution`` built for
    the old grid has no rule for it and ``sources`` refuses (truth: the
    lookup by static digest).
    """
    star = _phoenix()
    scene = _scene(star, (_table_planet(ALBEDO_NM, ALBEDO * 1e-9),))
    resolution = Resolution.build(rule, *scene.spectral_tables())

    def traced_star(wavelengths_nm, flux_density_jy):
        return Star(
            wavelengths_nm=wavelengths_nm,
            times_d=jnp.zeros(1),
            flux_density_jy=flux_density_jy,
            interpolation_method="linear",
        ).flux_density_jy

    with pytest.raises(TypeError, match="concrete"):
        jax.jit(traced_star)(jnp.asarray(star.wavelengths_nm), star.flux_density_jy)
    with pytest.raises(TypeError, match="not a leaf"):
        assign(scene, "star.knots", Knots(star.wavelengths_nm + 0.01))
    with pytest.raises(AttributeError):
        assign(scene, "star._wavelengths_nm", star.wavelengths_nm + 0.01)
    for knots in (star.knots.nm, ALBEDO_NM):
        assert not any(
            np.array_equal(np.asarray(leaf), knots) for leaf in jax.tree.leaves(scene)
        )
    moved = Star(
        wavelengths_nm=star.wavelengths_nm + 0.01,
        times_d=jnp.zeros(1),
        flux_density_jy=star.flux_density_jy,
        interpolation_method="linear",
    )
    with pytest.raises(LookupError, match="no rule"):
        eqx.tree_at(lambda s: s.star, scene, moved).sources(rule, resolution, TIMES)


@vv.case("stellar-photometry", "code-verification")
@pytest.mark.float64_only(reason="finite-difference derivative")
def test_sources_are_one_program_with_table_gradients(rule):
    """Fails if a new table value retraces or a table gradient is lost.

    A two-epoch PHOENIX star (the second epoch brighter in the red) with the
    absorbing planet. One jitted program serves a scaled table without a
    retrace, and its band rate (star plus planet, cubic response) has
    gradients with respect to ``flux_scale``, one table value inside the
    second band and the observation time equal to central differences to 1e-6
    relative (basis floating-point: the rate is linear in the first two and,
    with a linear table, in time between epochs). A batched pair of scenes
    returns the stacked single-scene values.
    """
    base = _phoenix()
    table = base.flux_density_jy[:, 0]
    knots = np.asarray(base.wavelengths_nm)
    red = 1 + 0.2 * (knots - 400.0) / 500.0
    star = Star(
        wavelengths_nm=base.wavelengths_nm,
        times_d=jnp.array([0.0, 20.0]),
        flux_density_jy=jnp.stack([table, table * red], axis=1),
        interpolation_method="linear",
    )
    scene = _scene(star, (_table_planet(ALBEDO_NM, ALBEDO * 1e-9),))
    resolution = Resolution.build(rule, *scene.spectral_tables())
    response = _responses(rule.nodes_nm)
    traces = []

    @jax.jit
    def rate(system, time_d):
        traces.append(None)
        sources = system.sources(rule, resolution, time_d[None])
        light = sources.star[..., 0] + sources.planets[..., 0, 0]
        return rule.integrate(light * response).sum()

    time = jnp.asarray(7.0)
    value = rate(scene, time)
    scaled = eqx.tree_at(
        lambda s: s.star._flux_density_phot, scene, 3 * scene.star._flux_density_phot
    )
    np.testing.assert_allclose(rate(scaled, time), 3 * value, rtol=1e-13)
    assert len(traces) == 1

    inside = int(np.flatnonzero((knots > 750.0) & (knots < 760.0))[0])
    leaves = (
        lambda s: s.star.flux_scale,
        lambda s: s.star._flux_density_phot,
    )
    for where in leaves:
        leaf = where(scene)
        grad = where(eqx.filter_grad(rate)(scene, time))
        step = 1e-4 * (jnp.abs(leaf).max())
        bump = jnp.zeros_like(leaf).at[(inside, 0) if leaf.ndim else ()].set(step)
        plus = rate(eqx.tree_at(where, scene, leaf + bump), time)
        minus = rate(eqx.tree_at(where, scene, leaf - bump), time)
        picked = grad[(inside, 0)] if leaf.ndim else grad
        np.testing.assert_allclose(picked, (plus - minus) / (2 * step), rtol=1e-6)
    fd = (rate(scene, time + 0.5) - rate(scene, time - 0.5)) / 1.0
    np.testing.assert_allclose(jax.grad(rate, argnums=1)(scene, time), fd, rtol=1e-6)

    single = _scene(base, (_table_planet(ALBEDO_NM, ALBEDO * 1e-9),))
    pair = Scene.from_scenes(
        [single, eqx.tree_at(lambda s: s.star.flux_scale, single, 2.0)]
    )
    batched = pair.sources(rule, resolution, TIMES)
    alone = single.sources(rule, resolution, TIMES)
    np.testing.assert_allclose(batched.star[1], 2 * alone.star, rtol=1e-14)
    np.testing.assert_allclose(batched.planets[0], alone.planets, rtol=1e-14)


def test_smooth_sources_keep_the_point_read(rule):
    """Fails if a smooth scene moves or the node star is indexed wrongly.

    A blackbody star of scale 1.3 with a gray and a thermal planet:
    ``band_contrasts`` with an empty resolution equals the band ratio of the
    point reads at the nodes to 32 relative roundings (Planck factors, scales, the
    eight-term node and line-of-sight sums and division), so the reflected
    planet carries the star's scale once, and the star's node values are its
    point samples. A disk rendered against the node star reproduces the disk
    at each node: all nodes at once, and band 1 node 2 alone (a transposed
    index reads another node).
    """
    star = BlackbodyStar(5772.0, 6.957e8, flux_scale=1.3)
    disk = GraterDisk(
        sma_AU=jnp.array(3.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-3.0),
        ksi0_AU=jnp.array(0.1),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(1.5),
        rmax_AU=jnp.array(5.0),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([0.4, 0.4]),
        Ag_grid=jnp.array([0.3, 0.3]),
        n_slices_los=8,
        amplitude=DensityDust(
            level=1.0, profile_id="grater-1999", calibration_id="unit-profile"
        ),
    )
    scene = _scene(
        star,
        (
            LambertianPhysicalModel(jnp.array([0.3])),
            BlackbodyPhysicalModel(jnp.array([900.0])),
        ),
        disk=disk,
    )
    empty = Resolution()
    nodes = rule.nodes_nm
    planet_points = jax.vmap(jax.vmap(lambda w: scene.planet_flux_densities(w, TIMES)))(
        nodes
    )
    star_points = scene.star.photon_flux_density(
        nodes[..., None], TIMES, dist_pc=scene.dist_pc
    )
    point_path = rule.integrate(planet_points) / rule.integrate(star_points)[:, None, :]
    tol = float(roundoff(1.0, n_ops=32, dtype=nodes.dtype))
    np.testing.assert_allclose(
        scene.band_contrasts(rule, TIMES, resolution=empty),
        point_path,
        rtol=tol,
    )
    sources = scene.sources(rule, empty, TIMES)
    np.testing.assert_allclose(
        sources.star[..., 0],
        _star_at(star, nodes),
        rtol=tol,
    )
    grid = PixelGrid.regular((4, 4), 0.1, "arcsec", orientation=SkyOrientation(0.0))
    east, north = grid.coordinates()
    every = scene.photon_radiance_at(nodes, 0.0, east, north, star=sources.node_star())
    np.testing.assert_allclose(
        every,
        scene.photon_radiance_at(nodes, 0.0, east, north),
        rtol=tol,
    )
    one = scene.photon_radiance_at(
        nodes[1, 2], 0.0, east, north, star=sources.node_star(1, 2)
    )
    np.testing.assert_allclose(one, every[1, 2], rtol=tol)


def test_the_node_star_builds_under_jit_and_a_concrete_nonfinite_scale_still_raises(
    rule,
):
    """Fails if a concrete scale inside jit trips the finite check, or nan passes.

    A concrete scale met while tracing is checked with NumPy, so building the
    node star in a jitted function works; a nonfinite concrete scale is still
    refused, inside jit as outside it.
    """
    from skyscapes._flux_scale import check_flux_scale

    star = BlackbodyStar(5772.0, 6.957e8, flux_scale=1.3)
    scene = _scene(star, (LambertianPhysicalModel(jnp.array([0.3])),))
    sources = scene.sources(rule, Resolution(), TIMES)
    eager = sources.node_star().photon_flux_density(rule.nodes_nm, 0.0, dist_pc=10.0)
    jitted = jax.jit(
        lambda: sources.node_star().photon_flux_density(
            rule.nodes_nm, 0.0, dist_pc=10.0
        )
    )()
    np.testing.assert_allclose(jitted, eager, rtol=1e-14)
    jax.jit(lambda: check_flux_scale(2.0))()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            jax.jit(lambda bad=bad: check_flux_scale(bad))()


def _tabulated_sources():
    """One of each tabulated source and break declarer, built from NumPy input."""
    exojax = make_physical_model(K=1, n_nu=40, n_layers=6)
    knots = np.array([450.0, 600.0, 650.0])
    grid = GridPhysicalModel(
        knots, np.array([0.0, 90.0, 180.0]), np.full((1, 3, 3), 1e-10)
    )
    disk = GraterDisk(
        sma_AU=jnp.array(3.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-3.0),
        ksi0_AU=jnp.array(0.1),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(1.5),
        rmax_AU=jnp.array(5.0),
        wavelengths_nm=np.array([400.0, 700.0, 1000.0]),
        g_HG_grid=jnp.array([0.4, 0.4, 0.4]),
        Ag_grid=jnp.array([0.3, 0.3, 0.3]),
        n_slices_los=8,
        amplitude=DensityDust(
            level=1.0, profile_id="grater-1999", calibration_id="unit-profile"
        ),
    )
    return {
        "star": _phoenix(),
        "time series": _table_planet(knots, np.array([0.1, 0.2, 0.3])),
        "grid": grid,
        "sum": SumPhysicalModel((grid, LambertianPhysicalModel(jnp.array([0.3])))),
        "zodi": PrecomputedZodi(np.array([450.0, 550.0, 650.0]), np.full(3, 1e12)),
        "exojax": exojax,
        "cached": PrecomputedPhysicalModel.from_physical_model(exojax),
        "parametric disk": disk,
    }


def test_no_parameter_leaf_moves_the_knots():
    """Fails if any parameter leaf of a tabulated source sets its breaks.

    Exact (bitwise). For each tabulated source and break declarer (star, time
    series, grid, a sum over a grid, the tabulated zodi, the ExoJAX model, the
    cached model and the parametric disk), every inexact leaf is scaled by
    1.07 and offset by 0.013 in turn, and ``spectral_breaks_nm()`` must be
    bitwise unchanged: the knots are static, never a leaf a fit can free.
    The ExoJAX knots are the native wavenumber grid converted once in
    float64 (truth: ``1e7 / nu`` sorted, computed here with NumPy), and the
    grid the opacity engine reads is the input grid bit for bit.
    """
    for name, source in _tabulated_sources().items():
        base = np.array(source.spectral_breaks_nm())
        assert base.size >= 2, name
        flat, treedef = jax.tree_util.tree_flatten(source)
        moved_any = False
        for i, leaf in enumerate(flat):
            if not is_param(leaf):
                continue
            moved_any = True
            leaves = list(flat)
            leaves[i] = leaf * 1.07 + 0.013
            moved = jax.tree_util.tree_unflatten(treedef, leaves)
            np.testing.assert_array_equal(
                moved.spectral_breaks_nm(), base, err_msg=name
            )
        assert moved_any, name
    nu = np.asarray(jnp.linspace(1.0e4, 2.5e4, 40))
    exojax = make_physical_model(K=1, n_nu=40, n_layers=6)
    np.testing.assert_array_equal(exojax.nu_grid.values, nu)
    np.testing.assert_array_equal(
        exojax.knots.nm, np.sort(1.0e7 / nu.astype(np.float64))
    )


def test_a_composite_derives_its_knots_from_its_components():
    """Fails if a sum stores its own copy of the knots.

    Exact. Replacing the gridded component of a sum with a model on another
    grid (``eqx.tree_at`` on the component) changes the sum's knots and digest
    to the new grid's; a stored copy would keep the old grid.
    """
    total = _tabulated_sources()["sum"]
    other = GridPhysicalModel(
        np.array([455.0, 610.0, 660.0]),
        np.array([0.0, 90.0, 180.0]),
        np.full((1, 3, 3), 1e-10),
    )
    swapped = eqx.tree_at(lambda m: m.components[0], total, other)
    np.testing.assert_array_equal(swapped.knots.nm, [455.0, 610.0, 660.0])
    assert swapped.knot_digest != total.knot_digest


@pytest.mark.parametrize("scalar_time", [False, True], ids=["array", "scalar"])
def test_numpy_and_jax_input_build_one_scene(rule, precision, scalar_time):
    """Fails if array types change the scene or a scalar date loses its time axis.

    The same star, tabulated planet and tabulated zodi built from NumPy or JAX
    arrays have equal knots, the same tree, and bitwise equal eager ``sources``.
    Both time inputs keep the same size-one time axis, including under JIT.
    """
    base = _phoenix()
    wavelengths_nm = np.asarray(base.wavelengths_nm, dtype=precision)

    def build(array):
        star = Star(
            wavelengths_nm=array(wavelengths_nm),
            times_d=array([0.0]),
            flux_density_jy=array(np.asarray(base.flux_density_jy)),
            interpolation_method="linear",
        )
        planet = _table_planet(array(ALBEDO_NM), array(ALBEDO * 1e-9))
        zodi = PrecomputedZodi(array([450.0, 650.0, 850.0]), array(np.full(3, 1e12)))
        return _scene(star, (planet,), local_zodi=zodi)

    host, device = build(np.array), build(jnp.array)
    assert jax.tree.structure(host) == jax.tree.structure(device)
    resolution = Resolution.build(rule, *host.spectral_tables())
    time_d = 0.0 if scalar_time else TIMES
    eager = host.sources(rule, resolution, time_d)
    expected = device.sources(rule, resolution, TIMES)
    compiled = eqx.filter_jit(Scene.sources)(
        host, rule, resolution, jnp.asarray(time_d)
    )
    for eager_leaf, expected_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(expected),
        jax.tree.leaves(compiled),
        strict=True,
    ):
        assert (
            eager_leaf.dtype == expected_leaf.dtype == compiled_leaf.dtype == precision
        )
        assert eager_leaf.shape == expected_leaf.shape == compiled_leaf.shape
        np.testing.assert_array_equal(eager_leaf, expected_leaf)
        # JIT can fuse the interpolation and reduction's floating-point operations.
        np.testing.assert_array_max_ulp(compiled_leaf, expected_leaf, maxulp=8)


def test_a_batch_refuses_twin_star_grids_by_name():
    """Fails if two star grids that differ only by rounding are stacked silently.

    Exact. A star on [400, 450.1, 520.3, 700] nm and one on its float32 copy
    agree within the wavelength tolerance but differ in digest, so
    ``Scene.from_scenes`` refuses them, naming both scenes, both digests and
    the fix (one shared ``Knots``); the same stars on one shared ``Knots``
    stack.
    """
    grid = np.array([400.0, 450.1, 520.3, 700.0])

    def scene(knots):
        star = Star(
            wavelengths_nm=knots,
            times_d=np.zeros(1),
            flux_density_jy=np.ones((4, 1)),
            interpolation_method="linear",
        )
        return _scene(star, (LambertianPhysicalModel(jnp.array([0.3])),))

    twin = Knots(grid.astype(np.float32))
    with pytest.raises(
        ValueError, match="build both stars on one shared Knots"
    ) as info:
        Scene.from_scenes([scene(Knots(grid)), scene(twin)])
    assert "scene 0 and scene 1" in str(info.value)
    assert twin.digest in str(info.value)
    shared = Knots(grid)
    assert Scene.from_scenes([scene(shared), scene(shared)]).batch_shape == (2,)


def test_a_batch_refuses_twin_planet_grids_by_name():
    """Fails if two planet-model grids that differ only by rounding stack with
    a generic structure error instead of naming the scenes and the fix.

    Exact. Time-series planets on [400, 450.1, 520.3, 700] nm and on its
    float32 copy agree within the wavelength tolerance but differ in digest;
    a different grid is named as different; one shared grid stacks. A scene
    with a local zodi never reaches the check: a batch refuses it.
    """
    grid = np.array([400.0, 450.1, 520.3, 700.0])
    star = Star(
        wavelengths_nm=grid,
        times_d=np.zeros(1),
        flux_density_jy=np.ones((4, 1)),
        interpolation_method="linear",
    )

    def scene(planet_grid):
        model = TimeSeriesPhysicalModel(
            planet_grid, np.zeros(1), np.full((1, 4, 1), 0.1)
        )
        return _scene(star, (model,))

    with pytest.raises(ValueError, match="planet models of group 0") as info:
        Scene.from_scenes([scene(grid), scene(grid.astype(np.float32))])
    assert "scene 0 and scene 1" in str(info.value)
    assert "differ in digest" in str(info.value)
    with pytest.raises(ValueError, match="have different knots"):
        Scene.from_scenes([scene(grid), scene(grid + 1.0)])
    assert Scene.from_scenes([scene(grid), scene(grid)]).batch_shape == (2,)


def test_a_one_wavelength_table_names_itself_when_integrated():
    """Fails if the refusal of a one-wavelength time series omits the table.

    Exact. The time series is valid as a point read; as a table of a scene it
    is refused by ``spectral_tables`` and by its ``knots``, naming the class
    and the wavelength count.
    """
    star = Star(
        wavelengths_nm=[400.0, 700.0],
        times_d=np.zeros(1),
        flux_density_jy=np.ones((2, 1)),
        interpolation_method="linear",
    )
    model = TimeSeriesPhysicalModel([500.0], np.zeros(1), np.full((1, 1, 1), 0.1))
    scene = _scene(star, (model,))
    for build in (lambda: model.knots, scene.spectral_tables):
        with pytest.raises(
            ValueError, match=r"TimeSeriesPhysicalModel with 1 wavelength\(s\)"
        ):
            build()


def _product_scene(offsets_tol):
    """A line-by-line planet over a coarse star with nearby or aligned knots.

    The planet is a time-series table at R = 1.5e5 over 498 to 552 nm with
    random ordinates; the star is a linear table of 13 knots, each moved off a
    planet knot by ``offsets_tol`` wavelength tolerances. Truth is computed
    from the raw arrays: the ratio of the exact integrals over 500 to 550 nm
    of (contrast times star) and of the star, taken cell by cell on the union
    of both raw grids and the band edges with Simpson's rule, which is exact
    for the quadratic product of two linear interpolants.
    """
    from optixstuff.spectral import merge_tolerance_nm

    rng = np.random.default_rng(7)
    ratio = 1.0 + 1.0 / 1.5e5
    planet = 498.0 * ratio ** np.arange(int(np.log(552 / 498) / np.log(ratio)) + 1)
    first, last = np.searchsorted(planet, [499.0, 551.0])
    pick = planet[np.linspace(first, last, 13).astype(int)]
    sign = np.where(np.arange(13) % 2, 1.0, -1.0)
    star_grid = pick + sign * np.asarray(offsets_tol) * merge_tolerance_nm(pick)
    contrast = rng.uniform(0.05, 1.0, planet.size)
    flux = rng.uniform(0.5, 2.0, 13)
    star = Star(
        wavelengths_nm=star_grid,
        times_d=np.zeros(1),
        flux_density_jy=flux[:, None],
        interpolation_method="linear",
    )
    model = TimeSeriesPhysicalModel(planet, np.zeros(1), contrast[None, :, None])
    system = _scene(star, (model,))
    return system, planet, star_grid, contrast


@pytest.mark.parametrize("aligned", [False, True], ids=["near", "aligned"])
def test_product_knots_keep_distinct_host_wavelengths(aligned):
    """Inside the common band, every distinct host wavelength stays an edge."""
    offsets = np.zeros(13) if aligned else np.linspace(0.1, 0.9, 13)
    system, planet, star, _ = _product_scene(offsets)
    union = np.union1d(planet, star)
    edges = system.spectral_tables()[-1].knots.values
    np.testing.assert_array_equal(
        edges[(edges >= 500.0) & (edges <= 550.0)],
        union[(union >= 500.0) & (union <= 550.0)],
    )


@pytest.mark.parametrize("aligned", [False, True], ids=["near", "aligned"])
@pytest.mark.float64_only(
    reason="V&V exact cell-integral reference resolves sub-float32-ULP knot shifts"
)
def test_product_integrates_near_but_distinct_knots_exactly(aligned):
    """The independent Simpson integral is exact for each quadratic product.

    Merging distinct knots biases the integral by about 1e-6 on this fixture.
    The 1e-12 relative float64 reference resolves that signal above rounding
    across 15,000 cells. The raw knot union is tested exactly in both lanes.
    """
    offsets = np.zeros(13) if aligned else np.linspace(0.1, 0.9, 13)
    system, planet, star_grid, contrast = _product_scene(offsets)
    rule = Quadrature.build([Band(500.0, 550.0)], order=4)
    resolution = Resolution.build(rule, *system.spectral_tables())
    value = system.band_contrasts(rule, jnp.zeros(1), resolution=resolution)[0, 0, 0]
    photons = np.asarray(
        system.star.photon_flux_density(
            jnp.asarray(star_grid), jnp.zeros(1), dist_pc=DIST_PC
        )
    ).reshape(-1)
    edges = np.union1d(planet, star_grid)
    edges = np.union1d(edges[(edges > 500.0) & (edges < 550.0)], [500.0, 550.0])
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi)

    def read(x):
        return np.interp(x, planet, contrast), np.interp(x, star_grid, photons)

    def simpson(f):
        return np.sum((hi - lo) / 6 * (f(lo) + 4 * f(mid) + f(hi)))

    num = simpson(lambda x: np.prod(read(x), axis=0))
    den = simpson(lambda x: read(x)[1])
    np.testing.assert_allclose(value, num / den, rtol=1e-12, atol=0)


def test_knot_caches_hold_at_most_32_grids():
    """Fails if the grid caches are unbounded: after 40 distinct grids each
    holds at most 32 entries (exact count; a float64 grid at R = 1e6 is 14 MB)."""
    from hwoutils.leaves import HostGrid

    from skyscapes._tables import knots_of, wavenumber_knots

    knots_of.cache_clear()
    wavenumber_knots.cache_clear()
    for i in range(40):
        knots_of(HostGrid([400.0 + i, 500.0 + i]))
        wavenumber_knots(HostGrid([1.0e4 + i, 2.0e4 + i]))
    assert knots_of.cache_info().currsize <= 32
    assert wavenumber_knots.cache_info().currsize <= 32


def test_batch_knots_of_an_invalid_table_raise_naming_it():
    """Exact basis: the table's own error is kept, not read as 'no knots'.

    A one-wavelength time series and a stand-in table with invalid knots raise
    ``ValueError`` naming the class; a smooth source with no ``knots`` gives
    None. Fails if the error is swallowed into None.
    """
    from skyscapes.physical_model import TimeSeriesPhysicalModel
    from skyscapes.scene._system_batch import _knots

    series = TimeSeriesPhysicalModel(
        wavelengths_nm=jnp.array([500.0]),
        times_d=jnp.array([0.0, 1.0]),
        contrast_grid=jnp.ones((1, 1, 2)),
    )
    with pytest.raises(ValueError, match="TimeSeriesPhysicalModel"):
        _knots(series)

    class Broken:
        @property
        def knots(self):
            raise ValueError("knots out of order")

    with pytest.raises(ValueError, match="Broken: knots out of order"):
        _knots(Broken())
    assert _knots(object()) is None
