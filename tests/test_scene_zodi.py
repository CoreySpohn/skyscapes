"""Scene holds the local zodi and sky position; an analytic exozodi is a disk."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from zodi.jax import AYOExozodi, AYOZodi

from skyscapes import Scene
from skyscapes.disk import AYOExozodiDisk
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


def test_ayo_exozodi_disk_matches_the_scalar_model():
    """The disk adapter evaluates AYOExozodi at the projected separation."""
    model = AYOExozodi(level_zodis=2.0)
    disk = AYOExozodiDisk(model)
    wave = jnp.array([500.0, 700.0])
    east = jnp.array([0.2, -0.3, 0.05])
    north = jnp.array([0.1, 0.4, 0.0])
    got = disk.photon_radiance_at(wave, 0.0, east, north, star=None, dist_pc=10.0)
    want = model.photon_radiance(
        wave[:, None], separation_au=10.0 * jnp.hypot(east, north)
    )
    assert got.shape == (2, 3)
    np.testing.assert_array_equal(got, want)


def test_ayo_exozodi_disk_on_a_grid_and_in_a_scene():
    """Wavelength axes lead the sky grid, and Scene forwards to the disk."""
    scene = _scene(disk=AYOExozodiDisk())
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
