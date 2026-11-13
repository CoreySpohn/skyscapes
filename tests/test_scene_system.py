"""scene.System -- star + tuple of planets + optional disk."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg
from hwoutils.conversions import photons_per_nm_per_m2_to_jy
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit

from skyscapes.disk import ExovistaDisk
from skyscapes.physical_model import GridPhysicalModel, LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet, Star, System

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
    c = sys_obj.contrasts(jnp.array(600.0), jnp.array([0.0]))
    assert c.shape == (2, 1)
    assert jnp.all(c >= 0.0)


def test_system_spec_flux_density_per_planet():
    """planet_flux_densities = contrasts x star.spec_flux_density."""
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(1.0, 0.3, 1.0)
    p2 = _make_planet(5.0, 0.5, 2.0)
    sys_obj = System(star=star, planets=(p1, p2), disk=None, trig_solver=SOLVER)
    f = sys_obj.planet_flux_densities(jnp.array(600.0), jnp.array([0.0]))
    assert f.shape == (2, 1)
    c = sys_obj.contrasts(jnp.array(600.0), jnp.array([0.0]))
    assert jnp.allclose(f, c * 1e9, rtol=1e-6)


@pytest.mark.parametrize("target", ["planet", "system"])
@pytest.mark.parametrize("wavelength", [[600.0], [500.0, 600.0], [[600.0]]])
@pytest.mark.parametrize("traced", [False, True])
def test_contrast_rejects_wavelength_arrays(target, wavelength, traced):
    """Reject nonscalar wavelengths eagerly and at JIT trace time."""
    star = FlatStar(Msun2kg, 10.0, 1e9)
    planet = _make_planet(1.0, 0.3, 1.0)
    system = System(star, (planet,), SOLVER)
    times = jnp.array([0.0, 100.0])

    def evaluate(wavelength_nm):
        if target == "planet":
            return planet.contrast(SOLVER, wavelength_nm, times, star=star)
        return system.contrasts(wavelength_nm, times)

    evaluate = jax.jit(evaluate) if traced else evaluate
    with pytest.raises(ValueError, match=r"scalar.*contrast_cube"):
        evaluate(jnp.asarray(wavelength))


@pytest.mark.parametrize("tabulated", [False, True])
def test_bin_contrast_weights_flux_and_has_point_limit(tabulated):
    """Fails if contrast is averaged without photon weighting or loses its point limit.

    For F(lambda)=lambda/500 and C(lambda)=lambda/500, the band ratio
    is (center + width**2/(12*center))/500, independent of epoch.
    """
    wave = jnp.array([400.0, 500.0, 600.0, 800.0])
    if tabulated:
        star = Star(
            Ms_kg=Msun2kg,
            dist_pc=10.0,
            wavelengths_nm=wave,
            times_jd=jnp.array([0.0, 200.0]),
            flux_density_jy=photons_per_nm_per_m2_to_jy(wave / 500, wave)[:, None]
            * jnp.array([[1.0, 2.0]]),
            interpolation_method="linear",
        )
    else:
        star = FlatStar(Msun2kg, 10.0, 1.0)
    p = _make_planet(1.0, 0.3, 1.0)
    model = GridPhysicalModel(
        wave, jnp.array([0.0, 180.0]), (wave / 500)[None, :, None] * jnp.ones((1, 1, 2))
    )
    p = eqx.tree_at(lambda p: p.physical_model, p, model)
    system = System(star, (p, p), SOLVER)
    centers = jnp.array([550.0, 650.0, 575.0])
    widths = jnp.array([200.0, 100.0, 0.0])
    times = jnp.array([0.0, 100.0])
    actual = eqx.filter_jit(system.bin_mean_contrast_cube)(centers, widths, times)
    expected = (centers + (widths**2 / (12 * centers) if tabulated else 0)) / 500
    np.testing.assert_allclose(
        actual, jnp.broadcast_to(expected[:, None, None], (3, 2, 2)), rtol=1e-12
    )
    for width in (1e-5, 1e-14):
        tiny = system.bin_mean_contrast_cube(centers, jnp.full(3, width), times)
        np.testing.assert_allclose(
            tiny, system.contrast_cube(centers, times), rtol=1e-8
        )

    @eqx.filter_jit
    def mean(p):
        return system.bin_mean_contrast_cube(
            jnp.array([550.0 * p[0]]), jnp.array([200.0 * p[1]]), times
        )[0, 0, 0]

    point, steps = jnp.ones(2), jnp.eye(2) * 1e-5
    fd = jax.vmap(lambda d: (mean(point + d) - mean(point - d)) / 2e-5)(steps)
    np.testing.assert_allclose(jax.grad(mean)(point), fd, rtol=1e-6, atol=1e-9)
