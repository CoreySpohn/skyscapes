"""Scene holds the zodi, and an analytic exozodi is a disk."""

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
    assert _scene().zodi is None
    zodi = AYOZodi()
    scene = _scene(zodi=zodi)
    assert scene.zodi is zodi
    direct = Scene(scene.star, scene.planets, scene.dynamics, zodi=zodi)
    assert direct.zodi is zodi
    assert "zodi: AYOZodi" in repr(scene)
    assert "zodi: None" in repr(_scene())


def test_batches_reject_zodi():
    """A zodi has no batch axes, so packing or batching a zodi scene fails."""
    with pytest.raises(ValueError, match="disk or zodi"):
        Scene.from_scenes([_scene(zodi=AYOZodi())])
    batch = Scene.from_scenes([_scene()])
    with pytest.raises(ValueError, match="no zodi"):
        Scene(
            batch.star,
            batch.planets,
            batch.dynamics,
            valid=batch.valid,
            zodi=AYOZodi(),
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
