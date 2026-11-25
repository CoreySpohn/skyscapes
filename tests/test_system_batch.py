"""Fixed-capacity scene batches, including inert padding and traced readers."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import GM_SUN_AU3_PER_D2
from hwoutils.epoch import Epoch
from hwoutils.leaves import canonical, check_tree
from optixstuff.grids import PixelGrid, SkyOrientation
from optixstuff.spectral import Band, Quadrature, Resolution

from skyscapes import AYOZodi
from skyscapes.disk import AYOExozodi
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet, Scene


def _scene(count):
    return Scene.from_classical(
        FlatStar(1e9),
        (Planet(jnp.ones(count), LambertianPhysicalModel(jnp.full(count, 0.3))),),
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.full(count, 0.001),
        dist_pc=10.0,
        a_AU=jnp.arange(count, dtype=float) + 1.0,
        e=0.08,
        i_rad=0.4,
        W_rad=0.3,
        w_rad=0.2,
        M0_rad=0.7,
    )


def _equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def _bits(x):
    """The array's raw bytes, so -0.0 and NaN payloads count, for any dtype."""
    x = np.asarray(x)
    return x.dtype.str, x.shape, np.ascontiguousarray(x).view(np.uint8)


def _same_bits(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        (xd, xs, xb), (yd, ys, yb) = _bits(x), _bits(y)
        assert (xd, xs) == (yd, ys)
        np.testing.assert_array_equal(xb, yb)


def test_round_trips():
    """Fails if host packing loses leaves; exact storage has zero rounding budget."""
    scenes = [_scene(k) for k in (0, 1, 3, 6)]
    split = Scene.from_classical(
        FlatStar(1e9),
        (
            Planet(jnp.array([2.0]), LambertianPhysicalModel(jnp.array([0.1]))),
            Planet(
                jnp.array([3.0, 4.0]), LambertianPhysicalModel(jnp.array([0.2, 0.3]))
            ),
        ),
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.array([0.001, 0.002, 0.003]),
        dist_pc=10.0,
        a_AU=jnp.array([3.0, 1.0, 2.0]),
        e=0.08,
        i_rad=0.4,
        W_rad=0.3,
        w_rad=0.2,
        M0_rad=0.7,
    )
    empty = Scene(scenes[0].star, (), scenes[0].dynamics)
    scenes.extend((split, empty))
    batch = Scene.from_scenes(scenes)
    for actual, expected in zip(batch.to_scenes(), scenes, strict=True):
        _same_bits(actual, canonical(expected))
        assert actual.body_index == expected.body_index
        restored = Scene.from_table(expected.to_table(), template=expected)
        _equal(restored, expected)
        assert restored.body_index == expected.body_index
    single = Scene(
        split.star,
        (
            Planet(
                jnp.array([2.0, 3.0, 4.0]),
                LambertianPhysicalModel(jnp.array([0.1, 0.2, 0.3])),
            ),
        ),
        split.dynamics,
        body_index=split.body_index,
    )
    stacked = jax.tree.map(lambda x: np.stack([x]), single)
    restored = stacked.to_scenes()[0]
    _equal(restored, single)
    assert restored.body_index == single.body_index
    table = batch.to_table()
    _equal(Scene.from_table(table, template=scenes[3]), batch)
    assert all(isinstance(x, jax.Array) for x in jax.tree.leaves(batch))


def test_batch_leaves_are_canonical():
    """Fails if a batch result keeps host leaves or the layout loses its integer kind.

    Truth: an unpacked scene equals its input leaf for leaf (integers and
    booleans exactly, floats bitwise, dtypes identical, since the packed
    values are the inputs' own, compared as bit patterns); the leaf rule and
    the tree structure are properties, so the tolerance is zero. The inputs
    are made canonical first because live scenes still hold Python-number
    leaves (a separate change), so the structure and trace count compare like
    with like.
    """
    scenes = [canonical(_scene(k)) for k in (0, 1, 3, 6)]
    batch = Scene.from_scenes(scenes)
    assert check_tree(batch) == ()
    assert not any(x.committed for x in jax.tree.leaves(batch))
    assert jnp.issubdtype(batch._source_layout.dtype, jnp.integer)
    assert batch.valid.dtype == bool
    traces = []

    @eqx.filter_jit
    def total(scene):
        traces.append(1)
        return scene.dynamics.planets.mass_Msun.sum()

    for actual, expected in zip(batch.to_scenes(), scenes, strict=True):
        assert check_tree(actual) == ()
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        _same_bits(actual, expected)
        total(actual)
        total(expected)
    assert len(traces) == 4
    cpu = jax.devices("cpu")[0]
    placed = Scene.from_scenes(scenes, device=cpu)
    assert all(x.devices() == {cpu} for x in jax.tree.leaves(placed))
    assert all(
        x.devices() == {cpu}
        for scene in placed.to_scenes(device=cpu)
        for x in jax.tree.leaves(scene)
    )


def test_round_trip_and_old_float_layout_tables():
    """Fails if the round trip changes a bit or a layout loses its integer kind.

    Truth: packing and unpacking move the inputs' own values, so the scenes
    equal the canonical inputs as bit patterns (zero tolerance), with an
    int32 layout and no leaf-rule violation. A table written before the layout
    became an integer array holds it as floats; casting is exact for integral
    values; a non-integral, non-finite or out-of-range value (below the -1
    sentinel or beyond int32) is refused in both precision lanes.
    """
    scenes = [canonical(_scene(k)) for k in (1, 3)]
    batch = Scene.from_scenes(scenes)
    assert batch._source_layout.dtype == np.int32
    assert check_tree(batch) == ()
    for actual, expected in zip(batch.to_scenes(), scenes, strict=True):
        assert check_tree(actual) == ()
        _same_bits(actual, expected)
    table = batch.to_table()
    old = dict(table, _source_layout=table["_source_layout"].astype(float))
    restored = Scene.from_table(old, template=scenes[1])
    assert restored._source_layout.dtype == np.int32
    assert check_tree(restored) == ()
    _same_bits(restored, batch)
    layout = old["_source_layout"]
    for value, match in (
        (layout + 0.5, "integer"),
        (np.where(layout < 0, np.inf, layout), "integer"),
        (np.where(layout < 0, np.nan, layout), "integer"),
        (np.where(layout < 0, 2.0**31, layout), r"\[-1"),
        (np.where(layout < 0, -2, layout).astype(np.int64), r"\[-1"),
        (layout >= 0, "integer"),
    ):
        bad = dict(old, _source_layout=value)
        with pytest.raises(ValueError, match=match):
            Scene.from_table(bad, template=scenes[1])


def _read(system, times, ephemeris=None, resolution=None):
    options = dict(ephemeris=ephemeris)
    resolution = Resolution() if resolution is None else resolution
    return dict(
        positions=system.positions(times, **options),
        position_angles=system.position_angles(times, **options),
        alpha_dMag=jnp.stack(system.alpha_dMag(times, **options), axis=-3),
        contrasts=system.contrasts(jnp.asarray(600.0), times, **options),
        band_contrasts=system.band_contrasts(
            jax.tree.map(lambda x: x.astype(times.dtype), _RULE),
            times,
            resolution=resolution,
            **options,
        ),
        planet_flux_densities=system.planet_flux_densities(
            jnp.asarray(600.0), times, **options
        ),
        star_reflex=system.star_reflex(times, **options),
        star_radial_velocity=system.star_radial_velocity(times, **options),
    )


_RULE = Quadrature.build([Band(560.0, 640.0)], order=2)
_read_jit = jax.jit(_read)
_ephemeris_jit = jax.jit(Scene.ephemeris)


@pytest.mark.parametrize("count", [0, 1, 3, 6])
def test_parity_and_padding(count):
    """Fails if batch/padding changes any reader; identical algebra demands bits.

    All capacity comparisons are bitwise; scalar/batched results allow 2 ULP
    for vectorized transcendental operations.
    """
    scene = _scene(count)
    times = jnp.array([17.0, 230.0])
    small = Scene.from_scenes([scene], capacity=6)
    large = Scene.from_scenes([scene], capacity=11)
    truth = _read_jit(scene, times)
    a, b = _read_jit(small, times), _read_jit(large, times)
    for name, expected in truth.items():
        actual, extended = a[name][0], b[name][0]
        if not name.startswith("star_"):
            actual, extended = actual[..., :count, :], extended[..., :count, :]
        np.testing.assert_array_max_ulp(actual, expected, maxulp=2)
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint8),
            np.asarray(extended).view(np.uint8),
            err_msg=name,
        )
    solve = Scene.ephemeris if count == 3 else _ephemeris_jit
    eph = solve(small, times)
    unpadded = solve(scene, times)
    extended_eph = solve(large, times)
    for field in ("position_AU", "velocity_AU_per_d"):
        actual = np.asarray(getattr(eph, field))[0, : count + 1]
        expected = np.asarray(getattr(unpadded, field))
        np.testing.assert_array_max_ulp(actual, expected, 2)
        extra = np.asarray(getattr(extended_eph, field))[0, : count + 1]
        np.testing.assert_array_equal(actual.view(np.uint8), extra.view(np.uint8))
    reused = _read_jit(small, times, eph)
    for name in a:
        # A supplied eager solve avoids fused orbital arithmetic (2 ULP floor).
        np.testing.assert_allclose(a[name], reused[name], rtol=1e-14, atol=1e-14)
    for name in ("contrasts", "band_contrasts", "planet_flux_densities"):
        np.testing.assert_array_equal(a[name][..., count:, :], 0.0)
    for x in jax.tree.leaves(a):
        assert np.all(np.isfinite(x))
    np.testing.assert_array_equal(small.planet_mass_Msun[..., count:], 0.0)
    np.testing.assert_array_equal(small.planets[0].Rp_Rearth[..., count:], 0.0)


def test_masked_gradients():
    """Fails if padded mass/orbit/model leaves influence a masked sum.

    Zero derivatives follow exact mask algebra; finite real derivatives must
    survive at circular, face-on padding, including geometry and dMag.
    """
    batch = Scene.from_scenes([_scene(1)], capacity=3)
    times = jnp.array([17.0, 230.0])

    def loss(s):
        values = _read(s, times)
        return sum(
            jnp.sum(x * (s.valid[..., None] if not name.startswith("star_") else 1.0))
            for name, x in values.items()
        )

    derivative = eqx.filter_jit(eqx.filter_grad(loss))(batch)
    for path, value in jax.tree_util.tree_flatten_with_path(derivative)[0]:
        assert np.all(np.isfinite(value)), jax.tree_util.keystr(path)
        if np.shape(value)[:2] == (1, 3):
            np.testing.assert_array_equal(np.asarray(value)[:, 1:], 0.0)
    # Mutating padded values must not perturb real bodies or stellar observables.
    mutated = eqx.tree_at(
        lambda s: s.dynamics.planets.mass_Msun, batch, jnp.array([[0.001, 20.0, 30.0]])
    )
    _equal(_read_jit(mutated, times), _read_jit(batch, times))


def test_take_vmap_times_and_compile():
    """Fails if gather picks the first axis, times cross rows, or counts retrace.

    Gather and repeated shared times are exact algebra; vmapped solves permit
    2 ULP for CPU vector math at the float64 rounding floor.
    """
    scenes = [_scene(k) for k in (0, 1, 3, 6, 1, 3)]
    batch = Scene.from_scenes(scenes, batch_shape=(2, 3))
    times = jnp.array([17.0, 230.0])
    direct = Scene(FlatStar(1e9), batch.planets[0], batch.dynamics, valid=batch.valid)
    np.testing.assert_array_equal(
        jax.jit(Scene.positions)(direct, times), _read_jit(batch, times)["positions"]
    )
    taken = jax.jit(lambda s, i: s.take(i))(batch, jnp.asarray(1))
    _equal(taken, jax.tree.map(lambda x: x[:, 1], batch))
    np.testing.assert_array_max_ulp(
        jax.vmap(lambda s: s.positions(times))(batch),
        np.stack(
            [
                jax.tree.map(lambda x, i=i: x[i], batch).positions(times)
                for i in range(2)
            ]
        ),
        2,
    )


def test_shared_times_and_one_compile():
    """Fails if equal time rows differ or valid counts retrace; exact mask algebra."""
    scenes = [_scene(k) for k in (0, 1, 3, 6, 1, 3)]
    batch = Scene.from_scenes(scenes, batch_shape=(2, 3))
    times = jnp.array([17.0, 230.0])
    _equal(
        _read_jit(batch, times), _read_jit(batch, jnp.broadcast_to(times, (2, 3, 2)))
    )
    compiled = jax.jit(lambda s: s.positions(times))
    compiled(batch).block_until_ready()
    changed = Scene.from_scenes(list(reversed(scenes)), batch_shape=(2, 3))
    compiled(changed).block_until_ready()
    assert compiled._cache_size() == 1


@pytest.mark.float64_only(
    reason="float32 orbital trace compared with a float64 baseline"
)
def test_precision():
    """Fails if packing truncates float64 or float32 cannot trace over five years.

    Float32 uses 1e-5 relative tolerance for accumulated orbital-phase roundoff;
    float64 output dtype and packing preservation (a round trip under x64) are
    exact requirements.
    """
    batch = Scene.from_scenes([_scene(3)], capacity=6)
    times = jnp.array([123.0, 777.0, 1500.0, 1826.25])
    truth = _read_jit(batch, times)
    assert all(x.dtype == np.float64 for x in jax.tree.leaves(truth))
    with jax.enable_x64(False):
        single = jax.tree.map(
            lambda x: x.astype(np.float32) if x.dtype != bool else x, batch
        )
        actual = _read_jit(single, times.astype(jnp.float32))
    rebuilt = Scene.from_scenes(batch.to_scenes(), capacity=6)
    _equal(rebuilt, batch)
    # A sky coordinate crosses zero; compare relative vector error there.
    delta = np.linalg.norm(actual["positions"] - truth["positions"], axis=-3)
    scale = np.linalg.norm(truth["positions"], axis=-3)
    assert np.all(delta < 1e-5 * scale)
    np.testing.assert_allclose(
        actual["contrasts"], truth["contrasts"], rtol=1e-5, atol=0.0
    )


@pytest.mark.parametrize("kind", ["capacity", "epoch", "star", "model", "disk"])
def test_invalid_batches(kind):
    """Fails if incompatible scenes silently pack; errors are exact host checks."""
    from skyscapes.disk import ExovistaDisk
    from skyscapes.physical_model import BlackbodyPhysicalModel
    from skyscapes.scene import BlackbodyStar

    a, b = _scene(1), _scene(3)
    options = {}
    if kind == "capacity":
        options["capacity"] = 2
    elif kind == "epoch":
        b = eqx.tree_at(
            lambda s: s.dynamics.frame,
            b,
            type(b.dynamics.frame)(b.dist_pc, Epoch(2451545.0)),
        )
    elif kind == "star":
        b = eqx.tree_at(lambda s: s.star, b, BlackbodyStar(5772.0, 7e8))
    elif kind == "model":
        b = eqx.tree_at(
            lambda s: s.planets[0].physical_model,
            b,
            BlackbodyPhysicalModel(jnp.full(3, 300.0)),
        )
    else:
        b = eqx.tree_at(
            lambda s: s.disk,
            b,
            ExovistaDisk(
                wavelengths_nm=jnp.array([500.0, 700.0]),
                contrast_cube=jnp.ones((2, 2, 2)),
                native_grid=PixelGrid.regular(
                    (jnp.ones((2, 2, 2))).shape[-2:],
                    0.01,
                    "arcsec",
                    orientation=SkyOrientation(0.0),
                ),
            ),
            is_leaf=lambda x: x is None,
        )
    with pytest.raises(ValueError):
        Scene.from_scenes([a, b], **options)


@pytest.mark.parametrize("kind", ["grid", "cached", "thermal", "time"])
def test_other_model_padding(kind):
    """Fails if radius-independent tables escape valid or grids become planets.

    Grids deliberately have length K: axis roles cannot be inferred from size
    alone. Round trips and zero outputs are exact storage/masking algebra.
    """
    from skyscapes.physical_model import (
        BlackbodyPhysicalModel,
        GridPhysicalModel,
        PrecomputedPhysicalModel,
        TimeSeriesPhysicalModel,
    )

    source = _scene(3)
    waves = jnp.array([500.0, 600.0, 700.0])
    if kind == "grid":
        model = GridPhysicalModel(
            waves, jnp.array([0.0, 90.0, 180.0]), jnp.full((3, 3, 3), 1e-9)
        )
    elif kind == "cached":
        model = PrecomputedPhysicalModel(
            jnp.full((3, 3), 0.3), jnp.array([1e4, 2e4, 3e4]), 3
        )
    elif kind == "thermal":
        model = BlackbodyPhysicalModel(jnp.full(3, 300.0))
    else:
        model = TimeSeriesPhysicalModel(
            waves, jnp.array([0.0]), jnp.full((3, 3, 1), 1e-9)
        )
    source = eqx.tree_at(lambda s: s.planets[0].physical_model, source, model)
    batch = Scene.from_scenes([source], capacity=6)
    _equal(batch.to_scenes()[0], source)
    _equal(Scene.from_table(batch.to_table(), template=batch), batch)
    times = jnp.array([17.0, 230.0])
    # The batch shares the source's tables, so their operators are the source's.
    resolution = Resolution.build(_RULE, *source.spectral_tables())
    values = _read_jit(batch, times, resolution=resolution)
    for name in ("contrasts", "band_contrasts", "planet_flux_densities"):
        np.testing.assert_array_equal(values[name][..., 3:, :], 0.0)

    derivative = eqx.filter_jit(
        eqx.filter_grad(lambda s: jnp.sum(s.contrasts(jnp.asarray(600.0), times)))
    )(batch)
    for value in jax.tree.leaves(derivative):
        assert np.all(np.isfinite(value))
        if np.shape(value)[:2] == (1, 6):
            np.testing.assert_array_equal(np.asarray(value)[:, 3:], 0.0)


def test_atmosphere_shared_axes():
    """Fails if shared atmospheric spectra are mistaken for planet parameters.

    K equals both spectral and layer lengths. Storage round trips and shared
    grid lengths are exact; no floating-point tolerance is needed.
    """
    from skyscapes.physical_model.exojax.components import PrecomputedAbsorption

    from .test_physical_model_exojax import make_physical_model

    model = make_physical_model(K=3, n_nu=3, n_layers=3)
    absorption = PrecomputedAbsorption(
        tuple(jnp.full((3, 3), 1e-22) for _ in model.species)
    )
    model = eqx.tree_at(lambda m: m.absorption, model, absorption)
    source = eqx.tree_at(lambda s: s.planets[0].physical_model, _scene(3), model)
    batch = Scene.from_scenes([source], capacity=6)
    packed = batch.planets[0].physical_model
    assert packed.surface.spectrum.shape == (1, 3)
    assert packed.species[0].rayleigh_xs.shape == (1, 3)
    assert packed.absorption.xsmatrix_per_species[0].shape == (1, 3, 3)
    _equal(batch.to_scenes()[0], source)


def test_flux_scale_survives_packing_and_tables_without_it_read_as_ones():
    """Fails if packing drops or misorders a planet scale, or an old table fails.

    Basis: exact storage, no rounding budget. The packed scene is sorted by
    semi-major axis, so a scale must follow its planet through the sort and the
    padding; a table written before the leaf existed reads as ones.
    """
    scene = Scene.from_classical(
        FlatStar(1e9, flux_scale=2.0),
        (
            Planet(jnp.array([2.0]), LambertianPhysicalModel(jnp.array([0.1])), 5.0),
            Planet(
                jnp.array([3.0, 4.0]),
                LambertianPhysicalModel(jnp.array([0.2, 0.3])),
                jnp.array([7.0, 9.0]),
            ),
        ),
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.array([0.001, 0.002, 0.003]),
        dist_pc=10.0,
        a_AU=jnp.array([3.0, 1.0, 2.0]),
        e=0.08,
        i_rad=0.4,
        W_rad=0.3,
        w_rad=0.2,
        M0_rad=0.7,
    )
    other = _scene(1)
    batch = Scene.from_scenes([scene, other], capacity=4)
    np.testing.assert_array_equal(batch.star.flux_scale, [2.0, 1.0])
    # Chain order is by semi-major axis: scene planets 1, 2, 0.
    np.testing.assert_array_equal(batch.planets[0].flux_scale[0], [7.0, 9.0, 5.0, 1.0])
    np.testing.assert_array_equal(batch.planets[0].flux_scale[1], [1.0, 1.0, 1.0, 1.0])
    restored = batch.to_scenes()[0]
    _equal(restored, scene)

    table = scene.to_table()
    assert "planets[0].flux_scale" in table and "star.flux_scale" in table
    old = {k: v for k, v in table.items() if not k.endswith("flux_scale")}
    back = Scene.from_table(old, template=scene)
    assert float(back.star.flux_scale) == 1.0
    np.testing.assert_array_equal(back.planets[0].flux_scale, [1.0])
    np.testing.assert_array_equal(back.planets[1].flux_scale, [1.0, 1.0])


def _full_scene(valid=None):
    """Two planets with a disk, a local zodi and a tilted disk midplane."""
    return Scene.from_classical(
        FlatStar(1e9),
        (Planet(jnp.ones(2), LambertianPhysicalModel(jnp.array([0.3, 0.4]))),),
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.array([0.001, 0.002]),
        dist_pc=10.0,
        a_AU=jnp.array([1.0, 2.0]),
        e=0.08,
        i_rad=0.4,
        W_rad=0.3,
        w_rad=0.2,
        M0_rad=0.7,
        valid=valid,
        disk=AYOExozodi(level_zodis=3.0),
        midplane_inc_deg=35.0,
        midplane_position_angle_deg=110.0,
        local_zodi=AYOZodi(surface_brightness_mag_arcsec2=21.5),
    )


def _source_reads(scene, times):
    east, north = jnp.array([0.03, -0.05]), jnp.array([0.04, 0.02])
    return dict(
        disk=scene.photon_radiance_at(jnp.array([550.0, 700.0]), 0.0, east, north),
        zodi=scene.sources(_RULE, Resolution(), times).local_zodi,
    )


def _fields_equal(actual, expected, skip=()):
    """Every dataclass field of the scene matches in structure and bits."""
    for field in dataclasses.fields(expected):
        if field.name in skip:
            continue
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        assert jax.tree.structure(a) == jax.tree.structure(b), field.name
        _same_bits(a, b)


def test_rebuilds_keep_every_source_model():
    """Fails if a table restore or an unpacking drops a field of the scene.

    Truth: the original scene, compared leaf by leaf as bit patterns (the
    stored values are the inputs' own, so the tolerance is zero) over every
    dataclass field of ``Scene``, so a field added later and forgotten at one
    rebuild site fails here, plus the disk radiance and the local-zodi node
    radiance read from the original. The second case unpacks an unbatched
    scene with one inactive slot, which leaves the star, disk, local zodi and
    midplane angles untouched.
    """
    times = jnp.array([0.0, 40.0])
    masked = canonical(_full_scene(valid=jnp.array([True, False])))
    original = canonical(_full_scene(valid=jnp.array([False, True])))
    restored = Scene.from_table(original.to_table(), template=original)
    _fields_equal(restored, original)
    assert not np.all(np.asarray(restored.valid))
    truth = _source_reads(original, times)
    for name, value in _source_reads(restored, times).items():
        np.testing.assert_array_equal(value, truth[name])

    (unpacked,) = masked.to_scenes()
    assert unpacked.n_planets == 1
    _fields_equal(
        unpacked,
        masked,
        skip=("planets", "dynamics", "valid", "body_index", "_source_layout"),
    )
    reads = _source_reads(unpacked, times)
    np.testing.assert_array_equal(reads["disk"], truth["disk"])
    np.testing.assert_array_equal(reads["zodi"], truth["zodi"])


def test_rebuilt_scene_reads_the_same_planets():
    """Fails if a table restore or an unpacking changes a planet read or element.

    Truth: the original scene, bitwise (the stored values are the inputs' own,
    so the tolerance is zero): planet flux densities and the orbital elements
    and mean motions of the dynamics, for a table round trip with a non-trivial
    ``valid`` mask and for the unpacked scene, whose one valid planet is the
    original's slot 1 read through the original's mask.
    """
    times = jnp.array([0.0, 40.0])
    original = canonical(_full_scene(valid=jnp.array([False, True])))
    restored = Scene.from_table(original.to_table(), template=original)
    flux = original.planet_flux_densities(jnp.asarray(600.0), times)
    np.testing.assert_array_equal(
        restored.planet_flux_densities(jnp.asarray(600.0), times), flux
    )
    _same_bits(restored.dynamics.orbits, original.dynamics.orbits)
    (unpacked,) = original.to_scenes()
    np.testing.assert_array_equal(
        unpacked.planet_flux_densities(jnp.asarray(600.0), times), flux[1:]
    )
    np.testing.assert_array_equal(
        unpacked.dynamics.orbits.n_rad_per_d, original.dynamics.orbits.n_rad_per_d[1:]
    )


@pytest.mark.parametrize("inactive_mass", [0.0, 1.0])
@pytest.mark.float64_only(reason="float64 orbital reference values")
def test_inactive_slot_mass_does_not_set_orbit_rates(inactive_mass):
    """Fails if an inactive planet's mass joins the enclosed mass of an outer planet.

    Truth: Kepler's third law, n^2 a^3 = GM_sun (M_star + m) with the star and
    the outer planet alone, from the Gaussian gravitational constant squared
    typed here (parity-checked against hwoutils), at 1e-12 relative, the
    double-precision floor for a square root of products. The inactive inner
    slot holds zero or one solar mass; the outer mean motion is bitwise the
    same for both, and a mask applied after the enclosed mass gives 0.02432744
    rad/day instead of 0.0172021.
    """
    gauss_k2 = 0.01720209895**2
    assert GM_SUN_AU3_PER_D2 == pytest.approx(gauss_k2, rel=1e-9)

    def outer_rate(mass):
        scene = Scene.from_classical(
            FlatStar(1.0),
            Planet(jnp.ones(2), LambertianPhysicalModel(jnp.array([0.3, 0.3]))),
            star_mass_Msun=1.0,
            planet_mass_Msun=jnp.array([mass, 3e-6]),
            dist_pc=10.0,
            a_AU=jnp.array([0.5, 1.0]),
            e=0.0,
            i_rad=0.0,
            W_rad=0.0,
            w_rad=0.0,
            M0_rad=0.0,
            valid=jnp.array([False, True]),
        )
        return scene.dynamics.orbits.n_rad_per_d[1]

    rate = outer_rate(inactive_mass)
    np.testing.assert_array_equal(rate, outer_rate(0.0))
    np.testing.assert_allclose(
        rate, np.sqrt(GM_SUN_AU3_PER_D2 * (1.0 + 3e-6)), rtol=1e-12, atol=0
    )
