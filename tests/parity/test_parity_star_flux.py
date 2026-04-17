"""Parity: scene.SpectrumStar identical to legacy.Star for spec_flux_density."""

from __future__ import annotations

import jax.numpy as jnp


def test_parity_star_flux_sampled_grid(legacy_system, scene_system):
    """SpectrumStar.spec_flux_density matches legacy.Star at every (wl, t) node."""
    wavelengths_nm = jnp.linspace(500.0, 900.0, 21)
    t_jd = legacy_system.star._times_jd[0] + jnp.linspace(0.0, 100.0, 11)

    for wl in wavelengths_nm:
        for t in t_jd:
            f_legacy = legacy_system.star.spec_flux_density(wl, t)
            f_new = scene_system.star.spec_flux_density(wl, t)
            assert jnp.allclose(f_new, f_legacy, rtol=1e-6), (
                f"mismatch at wl={wl}, t={t}: legacy={f_legacy}, new={f_new}"
            )


def test_parity_star_mass_rename(legacy_system, scene_system):
    """mass_kg (legacy) == Ms_kg (scene)."""
    assert jnp.allclose(
        jnp.asarray(legacy_system.star.mass_kg),
        jnp.asarray(scene_system.star.Ms_kg),
        rtol=1e-9,
    )


def test_parity_star_dist_pc(legacy_system, scene_system):
    """dist_pc matches between legacy and scene."""
    assert jnp.allclose(
        jnp.asarray(legacy_system.star.dist_pc),
        jnp.asarray(scene_system.star.dist_pc),
        rtol=1e-9,
    )
