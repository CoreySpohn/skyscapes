"""Tests for building scenes from common-origin Cartesian states."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg
from hwoutils.leaves import canonical
from hwoutils.numerics import roundoff
from hwoutils.precision import precision_of

from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet, Scene


def _planets(count):
    return Planet(jnp.ones(count), LambertianPhysicalModel(jnp.full(count, 0.3)))


@pytest.fixture
def snapshot():
    """Three planets at 1 AU, with two tied semimajor axes and a moving origin."""
    position = np.array([[0, 0, 20], [1, 0, 20], [0, 1, 20], [0, -1, 20]])
    circular_speed = np.sqrt(G * Msun2kg * (1.0 + 0.001))
    velocity = np.array([[0, 0, 0], [0, 1.1, 0], [-0.8, 0, 0], [0.8, 0, 0]])
    return position, velocity * circular_speed + [0, 0, 0.2]


def _build(position, velocity, order=None):
    return Scene.from_cartesian(
        FlatStar(1.0),
        _planets(len(position) - 1),
        star_mass_Msun=1.0,
        planet_mass_Msun=0.001,
        dist_pc=10.0,
        position_AU=position,
        velocity_AU_per_d=velocity,
        order=order,
    )


def test_stable_star_relative_semimajor_axis_order(snapshot):
    """Tied inner planets keep scene order and the fastest planet sorts last."""
    scene = _build(*snapshot)
    assert scene.body_index == (2, 0, 1)


def test_explicit_order_with_traced_states(snapshot):
    """An explicit hierarchy places the input rows into the specified chain.

    Sixty-four roundings at the 1 AU orbital scale cover the Cartesian
    to Jacobi conversion, two quaternion rotations and barycentric sums.
    """
    position, velocity = snapshot
    scene = eqx.filter_jit(lambda r, v: _build(r, v, (2, 0, 1)))(position, velocity)
    assert scene.body_index == (1, 2, 0)
    states = scene.ephemeris(jnp.array([0.0])).position_AU[:, 0]
    np.testing.assert_allclose(
        states[1:] - states[0],
        position[[3, 1, 2]] - position[0],
        atol=float(roundoff(1.0, n_ops=64, dtype=states.dtype)),
        rtol=0,
    )


def test_traced_states_require_order(snapshot):
    """Tracing an automatic hierarchy reports the required static order."""
    with pytest.raises(ValueError, match="order"):
        eqx.filter_jit(_build)(*snapshot)


def test_auto_order_rejects_unbound_state(snapshot):
    """A transverse speed above escape speed cannot define a bound hierarchy."""
    position, velocity = snapshot
    velocity[1, 1] = 0.1
    with pytest.raises(ValueError, match="bound"):
        _build(position, velocity)


def test_one_planet_classical_cartesian_round_trip():
    """A Cartesian snapshot at a nonzero epoch recovers the full orbit in AU.

    Allow 128 roundings at apoapsis for Cartesian inversion, propagation
    over one orbit and rotation back into the sky frame.
    """
    star, planets = FlatStar(1.0), _planets(1)
    masses_and_distance = dict(star_mass_Msun=0.83, planet_mass_Msun=0.003, dist_pc=9)
    classical = Scene.from_classical(
        star,
        planets,
        **masses_and_distance,
        a_AU=2.4,
        e=0.37,
        i_rad=0.83,
        W_rad=1.2,
        w_rad=0.63,
        M0_rad=2.1,
    )
    epoch_d = 31.25
    snapshot = classical.ephemeris(jnp.array([epoch_d]))
    rebuilt = Scene.from_cartesian(
        star,
        planets,
        **masses_and_distance,
        position_AU=snapshot.position_AU[:, 0],
        velocity_AU_per_d=snapshot.velocity_AU_per_d[:, 0],
        epoch_d=epoch_d,
    )
    dates = jnp.array([-200.0, epoch_d, 600.0])
    np.testing.assert_allclose(
        rebuilt.ephemeris(dates).position_AU,
        classical.ephemeris(dates).position_AU,
        atol=float(roundoff(2.4 * (1 + 0.37), n_ops=128, dtype=dates.dtype)),
        rtol=0,
    )


@pytest.mark.parametrize("order", [None, (0,)])
def test_common_origin_is_removed_before_narrowing(order):
    """A translated bound snapshot retains sub-float32-ULP offsets.

    The radius and tangential speed are binary fractions, so float64 host
    subtraction is exact. At epoch zero both translations must give the
    identical elements and ephemeris, even in a float32 process. The analytic
    vis-viva radius allows eight roundings for norms and the energy difference.
    """
    position = np.array([[0.0, 0.0, 0.0], [0.0625, 0.0, 0.0]])
    velocity = np.array([[0.0, 0.0, 0.0], [0.0, 0.0625, 0.0]])
    reference = _build(position, velocity, order)
    scene = _build(position + 2**21, velocity + 2**21, order)
    assert np.isfinite(scene.dynamics.orbits.quat).all()
    # Vis-viva evaluated independently on the original float64 snapshot.
    r, v = position[1, 0], velocity[1, 1]
    a = 1 / (2 / r - v**2 / (G * Msun2kg * 1.001))
    np.testing.assert_allclose(
        scene.dynamics.orbits.a_AU,
        a,
        rtol=float(roundoff(1.0, n_ops=8, dtype=scene.dynamics.orbits.a_AU.dtype)),
        atol=0,
    )

    np.testing.assert_array_equal(
        scene.dynamics.orbits.a_AU, reference.dynamics.orbits.a_AU
    )
    np.testing.assert_array_equal(
        scene.positions(jnp.zeros(1)), reference.positions(jnp.zeros(1))
    )


def test_typed_cartesian_snapshot_keeps_its_precision():
    """Fails if host centering widens a float32 orbit in an x64 process."""
    star, planets, args = canonical(
        (
            FlatStar(1.0),
            _planets(1),
            dict(
                star_mass_Msun=1.0,
                planet_mass_Msun=0.0,
                dist_pc=10.0,
                position_AU=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
                velocity_AU_per_d=np.array([[0, 0, 0], [0, np.sqrt(G * Msun2kg), 0]]),
                epoch_d=0.0,
                midplane_inc_deg=0.0,
                midplane_position_angle_deg=0.0,
            ),
        ),
        precision="float32",
    )

    def build(values):
        return Scene.from_cartesian(star, planets, **values, order=(0,))

    for scene in (build(args), eqx.filter_jit(build)(args)):
        assert precision_of(scene) == np.float32
