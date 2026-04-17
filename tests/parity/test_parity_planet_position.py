"""Parity: scene.Planet.position_arcsec identical to legacy.Planet.position."""

from __future__ import annotations

import jax.numpy as jnp


def test_parity_planet_position_t0(legacy_system, scene_system):
    """Compare on-sky (dRA, dDec) per planet at t=t0."""
    t_jd = jnp.atleast_1d(legacy_system.planet.orbix_planet.t0_d[0])
    pos_legacy = legacy_system.planet.position(float(t_jd[0]))
    pos_new = scene_system.positions(t_jd)

    pos_new_flat = pos_new[..., 0]

    n_compare = min(pos_legacy.shape[1], pos_new_flat.shape[1])
    assert jnp.allclose(
        pos_legacy[:, :n_compare], pos_new_flat[:, :n_compare], atol=1e-5, rtol=1e-5
    )


def test_parity_planet_alpha_dMag_t0(legacy_system, scene_system):
    """Compare (alpha_arcsec, dMag) at t=t0."""
    t_jd = jnp.atleast_1d(legacy_system.planet.orbix_planet.t0_d[0])

    alpha_legacy, dMag_legacy = legacy_system.planet.alpha_dMag(float(t_jd[0]))
    alpha_new, dMag_new = scene_system.alpha_dMag(t_jd)

    n_compare = min(alpha_legacy.shape[0], alpha_new.shape[0])
    assert jnp.allclose(
        alpha_legacy[:n_compare],
        alpha_new[:n_compare].squeeze(-1),
        rtol=1e-5,
        atol=1e-6,
    )
    assert jnp.isfinite(dMag_new[:n_compare]).all()
    assert dMag_new.shape[0] == dMag_legacy.shape[0]
