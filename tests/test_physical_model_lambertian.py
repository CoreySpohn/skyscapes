"""LambertianPhysicalModel -- analytic Lambert phase reproduces orbix dMag."""

from __future__ import annotations

import jax.numpy as jnp
from hwoutils.constants import Rearth2AU

from skyscapes.physical_model import (
    AbstractPhysicalModel,
    LambertianPhysicalModel,
)


def test_lambertian_is_abstract_physical_model():
    """LambertianPhysicalModel subclasses AbstractPhysicalModel."""
    lamb = LambertianPhysicalModel(Ag=jnp.array([0.3]))
    assert isinstance(lamb, AbstractPhysicalModel)


def test_lambertian_phase_zero_angle():
    """At beta=0 (full illumination), Lambert phase = 1."""
    lamb = LambertianPhysicalModel(Ag=jnp.array([0.3]))
    contrast = lamb.contrast(
        phase_angle_rad=jnp.array([[0.0]]),
        dist_AU=jnp.array([[1.0]]),
        wavelength_nm=jnp.array([500.0]),
        Rp_Rearth=jnp.array([1.0]),
    )
    # contrast = Ag * phase(0) * (Rp/r)^2 = 0.3 * 1 * (Rearth2AU)^2
    expected = 0.3 * 1.0 * Rearth2AU**2
    assert jnp.allclose(contrast, expected, rtol=1e-6)


def test_lambertian_phase_pi_is_dark():
    """At beta=pi (back-lit), Lambert phase = 0 -> contrast = 0."""
    lamb = LambertianPhysicalModel(Ag=jnp.array([0.3]))
    contrast = lamb.contrast(
        phase_angle_rad=jnp.array([[jnp.pi]]),
        dist_AU=jnp.array([[1.0]]),
        wavelength_nm=jnp.array([500.0]),
        Rp_Rearth=jnp.array([1.0]),
    )
    assert jnp.allclose(contrast, 0.0, atol=1e-10)


def test_lambertian_inverse_square():
    """At fixed phase, contrast proportional to 1/r**2."""
    lamb = LambertianPhysicalModel(Ag=jnp.array([0.3]))
    c1 = lamb.contrast(
        phase_angle_rad=jnp.array([[jnp.pi / 3]]),
        dist_AU=jnp.array([[1.0]]),
        wavelength_nm=jnp.array([500.0]),
        Rp_Rearth=jnp.array([1.0]),
    )
    c2 = lamb.contrast(
        phase_angle_rad=jnp.array([[jnp.pi / 3]]),
        dist_AU=jnp.array([[2.0]]),
        wavelength_nm=jnp.array([500.0]),
        Rp_Rearth=jnp.array([1.0]),
    )
    assert jnp.allclose(c2 / c1, 0.25, rtol=1e-6)


def test_lambertian_wavelength_independent():
    """Grey Lambertian -- contrast should not depend on wavelength."""
    lamb = LambertianPhysicalModel(Ag=jnp.array([0.3]))
    c_blue = lamb.contrast(
        phase_angle_rad=jnp.array([[jnp.pi / 4]]),
        dist_AU=jnp.array([[1.0]]),
        wavelength_nm=jnp.array([400.0]),
        Rp_Rearth=jnp.array([1.0]),
    )
    c_red = lamb.contrast(
        phase_angle_rad=jnp.array([[jnp.pi / 4]]),
        dist_AU=jnp.array([[1.0]]),
        wavelength_nm=jnp.array([800.0]),
        Rp_Rearth=jnp.array([1.0]),
    )
    assert jnp.allclose(c_blue, c_red, rtol=1e-6)


def test_lambertian_matches_orbix_alpha_dmag():
    """Hand-built LambertianPhysicalModel reproduces the exact Lambert dMag."""
    from orbix.equations.phase import lambert_phase_exact
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.orbit import KeplerianOrbit

    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)

    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([jnp.pi / 3]),  # non-trivial inclination
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([jnp.pi / 4]),
        t0_d=jnp.array([0.0]),
    )
    Ms_kg = jnp.array([1.989e30])
    Ag = jnp.array([0.3])
    Rp_AU = jnp.array([1.0]) * Rearth2AU
    t_jd = jnp.array([0.0])

    # Rebuild the same phase/distance via orbit.propagate then feed
    # LambertianPhysicalModel. The output contrast must match dMag exactly.
    _, phase_angle_rad, dist_AU = orbit.propagate(solver, t_jd, Ms_kg=Ms_kg)

    phase = lambert_phase_exact(jnp.cos(phase_angle_rad), jnp.sin(phase_angle_rad))
    log_arg = Ag[:, None] * (Rp_AU[:, None] / dist_AU) ** 2 * phase
    dMag_ref = -2.5 * jnp.log10(log_arg + jnp.finfo(log_arg.dtype).tiny)

    lamb = LambertianPhysicalModel(Ag=jnp.array([0.3]))
    contrast = lamb.contrast(
        phase_angle_rad=phase_angle_rad,
        dist_AU=dist_AU,
        wavelength_nm=jnp.array([500.0]),
        Rp_Rearth=jnp.array([1.0]),
    )
    dMag_new = -2.5 * jnp.log10(contrast + jnp.finfo(contrast.dtype).tiny)
    assert jnp.allclose(dMag_new, dMag_ref, rtol=1e-6)
