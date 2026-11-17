"""LambertianPhysicalModel -- analytic Lambert phase and contrast."""

from __future__ import annotations

import jax.numpy as jnp
from hwoutils.constants import Rearth2AU

from skyscapes.physical_model import (
    AbstractPhysicalModel,
    LambertianPhysicalModel,
    lambert_phase,
)

from .conftest import two_body_reference


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
    """Fails if endpoint roundoff creates negative reflected light.

    Exact-algebra basis: Phi(0)=1 and Phi(pi)=0. Float32 pi makes the sine
    slightly negative, so the public phase function must clip that endpoint.
    The float64 contrast permits the sine's absolute roundoff at pi.
    """
    endpoints = lambert_phase(jnp.array([0.0, jnp.pi], dtype=jnp.float32))
    assert bool(jnp.all(endpoints == jnp.array([1.0, 0.0])))
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
    """Gray Lambertian -- contrast should not depend on wavelength."""
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


def test_lambertian_matches_closed_form_dmag():
    """Hand-built LambertianPhysicalModel reproduces the exact Lambert dMag."""
    elements = dict(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([jnp.pi / 3]),  # non-trivial inclination
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([jnp.pi / 4]),
        epoch_d=jnp.array([0.0]),
    )
    star_mass_kg = jnp.array([1.989e30])
    Ag = jnp.array([0.3])
    Rp_AU = jnp.array([1.0]) * Rearth2AU
    time_d = jnp.array([0.0])

    # Independent two-body phase/distance feed LambertianPhysicalModel.
    # The output contrast must match the analytic Lambert dMag.
    reference = two_body_reference(time_d, **elements, star_mass_kg=star_mass_kg)
    phase_angle_rad, dist_AU = reference.phase_angle_rad, reference.distance_AU

    phase = (
        jnp.sin(phase_angle_rad) + (jnp.pi - phase_angle_rad) * jnp.cos(phase_angle_rad)
    ) / jnp.pi
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
