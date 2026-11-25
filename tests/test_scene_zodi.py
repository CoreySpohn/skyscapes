"""Scene holds the local zodi and sky position; an analytic exozodi is a disk."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyscapes import AYOZodi, Scene
from skyscapes.disk import AYOExozodi
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet


def _scene(**extra):
    planet = Planet(jnp.ones(1), LambertianPhysicalModel(Ag=jnp.array([0.3])))
    return Scene.from_classical(
        FlatStar(1e9),
        (planet,),
        star_mass_Msun=1.0,
        planet_mass_Msun=3e-6,
        dist_pc=10.0,
        a_AU=jnp.array([1.0]),
        e=0.0,
        i_rad=0.5,
        W_rad=0.0,
        w_rad=0.0,
        M0_rad=0.0,
        **extra,
    )


def test_zodi_defaults_to_none_and_is_stored():
    """from_classical and the constructor carry the zodi on the scene."""
    assert _scene().local_zodi is None
    zodi = AYOZodi()
    scene = _scene(local_zodi=zodi)
    assert scene.local_zodi is zodi
    direct = Scene(scene.star, scene.planets, scene.dynamics, local_zodi=zodi)
    assert direct.local_zodi is zodi
    assert "local_zodi: AYOZodi" in repr(scene)
    assert "local_zodi: None" in repr(_scene())


def test_batches_reject_zodi():
    """A zodi has no batch axes, so packing or batching a zodi scene fails."""
    with pytest.raises(ValueError, match="disk or local zodi"):
        Scene.from_scenes([_scene(local_zodi=AYOZodi())])
    batch = Scene.from_scenes([_scene()])
    with pytest.raises(ValueError, match="no local zodi"):
        Scene(
            batch.star,
            batch.planets,
            batch.dynamics,
            valid=batch.valid,
            local_zodi=AYOZodi(),
        )


@pytest.mark.float64_only(reason="float64 photometry reference")
def test_ayo_exozodi_units():
    """Check AB normalization and photon units from SI primitives.

    A 1e-12 relative tolerance allows float64 unit arithmetic. The separation
    is one AU, a distance of 10 pc seeing 0.1 arcsec.
    """
    disk = AYOExozodi()
    np.testing.assert_array_equal(disk.spectral_breaks_nm(), np.empty(0))
    waves = jnp.array([450.0, 550.0, 750.0])
    zero_jy = 10 ** (-0.4 * 48.60) * 1e4 / 1e7 / 1e-26
    expected = zero_jy * 1e-26 / (6.62607015e-34 * np.asarray(waves))
    expected *= 3 * 10 ** (-0.4 * 22) / (np.pi / (180 * 3600)) ** 2
    got = disk.photon_radiance_at(
        waves,
        0.0,
        0.1,
        0.0,
        star=None,
        dist_pc=10.0,
        inclination_deg=0.0,
        position_angle_deg=0.0,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
    got = AYOExozodi(6.0, 23.0).photon_radiance_at(
        waves,
        0.0,
        0.0,
        0.1,
        star=None,
        dist_pc=10.0,
        inclination_deg=0.0,
        position_angle_deg=0.0,
    )
    np.testing.assert_allclose(got, expected * 2 * 10 ** (-0.4), rtol=1e-12, atol=0)


def test_ayo_exozodi_projected_separation_law():
    """Check r^-2 with physical separation = distance_pc * angle_arcsec.

    Exact powers of two set the ratios; 1e-12 allows JIT rounding.
    """
    disk = AYOExozodi()
    distances = jnp.array([10.0, 20.0, 10.0, 20.0])
    separations = jnp.array([0.1, 0.1, 0.2, 0.05])

    def read(distance, separation):
        return disk.photon_radiance_at(
            550.0,
            0.0,
            separation,
            0.0,
            star=None,
            dist_pc=distance,
            inclination_deg=0.0,
            position_angle_deg=0.0,
        )

    base = (
        10 ** (-0.4 * 48.60)
        * 1e-3
        / (6.62607015e-34 * 550)
        * 3
        * 10 ** (-0.4 * 22)
        / (np.pi / (180 * 3600)) ** 2
    )
    expected = jnp.array([1.0, 0.25, 0.25, 1.0]) * base
    np.testing.assert_allclose(
        jax.vmap(read)(distances, separations), expected, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(read))(distances, separations), expected, rtol=1e-12, atol=0
    )


def test_ayo_exozodi_disk_on_a_grid_and_in_a_scene():
    """Wavelength axes lead the sky grid, and Scene forwards to the disk."""
    scene = _scene(disk=AYOExozodi())
    east = jnp.linspace(0.1, 0.5, 4)[:, None]
    north = jnp.linspace(0.1, 0.3, 3)[None, :]
    wave = jnp.array([500.0, 600.0, 700.0])
    got = scene.photon_radiance_at(wave, 0.0, east, north)
    assert got.shape == (3, 4, 3)
    assert np.all(np.asarray(got) > 0)


def test_scene_position_is_read_from_the_frame_and_defaults_to_unplaced():
    """from_classical places the scene on the frame; omitting it leaves it unplaced."""
    assert _scene().ra_deg is None
    assert _scene().dec_deg is None
    placed = _scene(ra_deg=100.0, dec_deg=-20.0)
    np.testing.assert_allclose(placed.ra_deg, 100.0)
    np.testing.assert_allclose(placed.dec_deg, -20.0)
    assert placed.dynamics.frame.ra_deg is placed.ra_deg
    assert "ra_deg" not in vars(placed.star)


def test_from_scenes_refuses_a_mix_of_placed_and_unplaced_scenes():
    """A batch is all placed or all unplaced, never a mix."""
    placed = _scene(ra_deg=100.0, dec_deg=-20.0)
    with pytest.raises(ValueError, match="must all be placed"):
        Scene.from_scenes([placed, _scene()])
    batch = Scene.from_scenes([placed, _scene(ra_deg=10.0, dec_deg=5.0)])
    np.testing.assert_allclose(batch.ra_deg, [100.0, 10.0])
    assert Scene.from_scenes([_scene(), _scene()]).ra_deg is None


def test_table_round_trip_keeps_the_sky_position():
    """to_table and from_table preserve the placed frame and its epoch."""
    batch = Scene.from_scenes([_scene(ra_deg=100.0, dec_deg=-20.0)] * 2)
    back = Scene.from_table(batch.to_table(), template=batch)
    np.testing.assert_allclose(back.ra_deg, [100.0, 100.0])
    np.testing.assert_allclose(back.dec_deg, [-20.0, -20.0])
    assert back.epoch == batch.epoch


def test_scene_sky_frame_is_the_dynamics_frame():
    """Scene.sky_frame is the frame the dynamics own, placed or not."""
    scene = _scene(ra_deg=100.0, dec_deg=-20.0)
    assert scene.sky_frame is scene.dynamics.frame
    assert scene.sky_frame.ra_deg is scene.ra_deg
    assert _scene().sky_frame.ra_deg is None
