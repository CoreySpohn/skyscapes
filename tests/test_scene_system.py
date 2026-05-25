"""scene.System -- star + tuple of planets + optional disk."""

from __future__ import annotations

import jax.numpy as jnp
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.system.orbit import KeplerianOrbit

from skyscapes.disk import ExovistaDisk
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet, System

SOLVER = get_grid_solver(level="scalar", E=False, trig=True, jit=True)


def _make_planet(a_AU: float, Ag: float, Rp_Rearth: float) -> Planet:
    orbit = KeplerianOrbit(
        a_AU=jnp.array([a_AU]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([jnp.pi / 3]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.0]),
        t0_d=jnp.array([0.0]),
    )
    physical_model = LambertianPhysicalModel(Ag=jnp.array([Ag]))
    return Planet(
        Rp_Rearth=jnp.array([Rp_Rearth]),
        Mp_Mearth=jnp.array([1.0]),
        orbit=orbit,
        physical_model=physical_model,
    )


def test_system_basic_fields():
    """System stores star and planets tuple; disk defaults to None."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(1.0, 0.3, 1.0)
    p2 = _make_planet(5.0, 0.5, 2.0)
    sys_obj = System(star=star, planets=(p1, p2), disk=None, trig_solver=SOLVER)
    assert sys_obj.n_planets == 2
    assert sys_obj.disk is None


def test_system_with_disk_field():
    """System stores an AbstractDisk when provided."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p = _make_planet(1.0, 0.3, 1.0)
    wl = jnp.linspace(400.0, 1000.0, 5)
    cube = jnp.full((wl.size, 4, 4), 1e-6)
    disk = ExovistaDisk(pixel_scale_arcsec=0.01, wavelengths_nm=wl, contrast_cube=cube)
    sys_obj = System(star=star, planets=(p,), disk=disk, trig_solver=SOLVER)
    assert sys_obj.disk is disk


def test_system_positions_stacks_planets():
    """System.positions concatenates per-planet (2, K, T) into (2, K_total, T)."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(1.0, 0.3, 1.0)
    p2 = _make_planet(5.0, 0.5, 2.0)
    sys_obj = System(star=star, planets=(p1, p2), disk=None, trig_solver=SOLVER)
    pos = sys_obj.positions(jnp.array([0.0, 100.0]))
    assert pos.shape == (2, 2, 2)


def test_system_contrasts_stacks_planets():
    """System.contrasts returns (K_total, T) per-planet contrast."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(1.0, 0.3, 1.0)
    p2 = _make_planet(5.0, 0.5, 2.0)
    sys_obj = System(star=star, planets=(p1, p2), disk=None, trig_solver=SOLVER)
    c = sys_obj.contrasts(jnp.array([600.0]), jnp.array([0.0]))
    assert c.shape == (2, 1)
    assert jnp.all(c >= 0.0)


def test_system_spec_flux_density_per_planet():
    """planet_flux_densities = contrasts x star.spec_flux_density."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(1.0, 0.3, 1.0)
    p2 = _make_planet(5.0, 0.5, 2.0)
    sys_obj = System(star=star, planets=(p1, p2), disk=None, trig_solver=SOLVER)
    f = sys_obj.planet_flux_densities(jnp.array([600.0]), jnp.array([0.0]))
    assert f.shape == (2, 1)
    c = sys_obj.contrasts(jnp.array([600.0]), jnp.array([0.0]))
    assert jnp.allclose(f, c * 1e9, rtol=1e-6)
