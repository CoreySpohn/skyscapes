"""Fixed-capacity scene batches, including inert padding and traced readers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.epoch import Epoch
from hwoutils.grids import PixelGrid, SkyOrientation
from hwoutils.spectral import Band, Quadrature

from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet, System


def _scene(count):
    return System.from_classical(
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


def test_round_trips():
    """Fails if host packing loses leaves; exact storage has zero rounding budget."""
    scenes = [_scene(k) for k in (0, 1, 3, 6)]
    split = System.from_classical(
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
    empty = System(scenes[0].star, (), scenes[0].dynamics)
    scenes.extend((split, empty))
    batch = System.from_systems(scenes)
    for actual, expected in zip(batch.to_systems(), scenes, strict=True):
        _equal(actual, expected)
        assert actual.body_index == expected.body_index
        restored = System.from_table(expected.to_table(), template=expected)
        _equal(restored, expected)
        assert restored.body_index == expected.body_index
    single = System(
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
    restored = stacked.to_systems()[0]
    _equal(restored, single)
    assert restored.body_index == single.body_index
    table = batch.to_table()
    _equal(System.from_table(table, template=scenes[3]), batch)
    assert all(isinstance(x, np.ndarray) for x in jax.tree.leaves(batch))


def _read(system, times, ephemeris=None):
    options = dict(ephemeris=ephemeris)
    return dict(
        positions=system.positions(times, **options),
        position_angles=system.position_angles(times, **options),
        alpha_dMag=jnp.stack(system.alpha_dMag(times, **options), axis=-3),
        contrasts=system.contrasts(jnp.asarray(600.0), times, **options),
        band_contrasts=system.band_contrasts(
            jax.tree.map(lambda x: x.astype(times.dtype), _RULE),
            times,
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
_ephemeris_jit = jax.jit(System.ephemeris)


@pytest.mark.parametrize("count", [0, 1, 3, 6])
def test_parity_and_padding(count):
    """Fails if batch/padding changes any reader; identical algebra demands bits.

    All capacity comparisons are bitwise; scalar/batched results allow 2 ULP
    for vectorized transcendental operations at the float64 rounding floor.
    """
    scene = _scene(count)
    times = jnp.array([17.0, 230.0])
    small = System.from_systems([scene], capacity=6)
    large = System.from_systems([scene], capacity=11)
    truth = _read_jit(scene, times)
    a, b = _read_jit(small, times), _read_jit(large, times)
    for name, expected in truth.items():
        actual, extended = a[name][0], b[name][0]
        if not name.startswith("star_"):
            actual, extended = actual[..., :count, :], extended[..., :count, :]
        np.testing.assert_array_max_ulp(actual, expected, maxulp=2)
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint64),
            np.asarray(extended).view(np.uint64),
            err_msg=name,
        )
    solve = System.ephemeris if count == 3 else _ephemeris_jit
    eph = solve(small, times)
    unpadded = solve(scene, times)
    extended_eph = solve(large, times)
    for field in ("position_AU", "velocity_AU_per_d"):
        actual = np.asarray(getattr(eph, field))[0, : count + 1]
        expected = np.asarray(getattr(unpadded, field))
        np.testing.assert_array_max_ulp(actual, expected, 2)
        extra = np.asarray(getattr(extended_eph, field))[0, : count + 1]
        np.testing.assert_array_equal(actual.view(np.uint64), extra.view(np.uint64))
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
    batch = System.from_systems([_scene(1)], capacity=3)
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
    batch = System.from_systems(scenes, batch_shape=(2, 3))
    times = jnp.array([17.0, 230.0])
    direct = System(FlatStar(1e9), batch.planets[0], batch.dynamics, valid=batch.valid)
    np.testing.assert_array_equal(
        jax.jit(System.positions)(direct, times), _read_jit(batch, times)["positions"]
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
    batch = System.from_systems(scenes, batch_shape=(2, 3))
    times = jnp.array([17.0, 230.0])
    _equal(
        _read_jit(batch, times), _read_jit(batch, jnp.broadcast_to(times, (2, 3, 2)))
    )
    compiled = jax.jit(lambda s: s.positions(times))
    compiled(batch).block_until_ready()
    changed = System.from_systems(list(reversed(scenes)), batch_shape=(2, 3))
    compiled(changed).block_until_ready()
    assert compiled._cache_size() == 1


def test_precision():
    """Fails if packing truncates float64 or float32 cannot trace over five years.

    Float32 uses 1e-5 relative tolerance for accumulated orbital-phase roundoff;
    float64 output dtype and host packing preservation are exact requirements.
    """
    batch = System.from_systems([_scene(3)], capacity=6)
    times = jnp.array([123.0, 777.0, 1500.0, 1826.25])
    truth = _read_jit(batch, times)
    assert all(x.dtype == np.float64 for x in jax.tree.leaves(truth))
    with jax.enable_x64(False):
        single = jax.tree.map(
            lambda x: x.astype(np.float32) if x.dtype != bool else x, batch
        )
        actual = _read_jit(single, times.astype(jnp.float32))
        rebuilt = System.from_systems(batch.to_systems(), capacity=6)
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
        System.from_systems([a, b], **options)


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
    batch = System.from_systems([source], capacity=6)
    _equal(batch.to_systems()[0], source)
    _equal(System.from_table(batch.to_table(), template=batch), batch)
    times = jnp.array([17.0, 230.0])
    values = _read_jit(batch, times)
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
    batch = System.from_systems([source], capacity=6)
    packed = batch.planets[0].physical_model
    assert packed.surface.spectrum.shape == (1, 3)
    assert packed.species[0].rayleigh_xs.shape == (1, 3)
    assert packed.absorption.xsmatrix_per_species[0].shape == (1, 3, 3)
    _equal(batch.to_systems()[0], source)
