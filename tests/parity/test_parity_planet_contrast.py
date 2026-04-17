"""Parity: scene.Planet.contrast ≡ legacy.Planet.contrast (via GridAtmosphere)."""

from __future__ import annotations

import jax.numpy as jnp


def test_parity_planet_contrast_grid(legacy_system, scene_system):
    """Contrast at a grid of (wavelength, time) matches within tolerance."""
    wavelengths_nm = jnp.linspace(500.0, 900.0, 5)
    t0 = float(legacy_system.planet.orbix_planet.t0_d[0])
    t_jd = jnp.array([t0, t0 + 50.0, t0 + 150.0])

    for wl in wavelengths_nm:
        for t in t_jd:
            c_legacy = legacy_system.planet.contrast(float(wl), float(t))
            c_new = scene_system.contrasts(
                jnp.atleast_1d(wl), jnp.atleast_1d(t)
            )[:, 0]
            n_compare = min(c_legacy.shape[0], c_new.shape[0])
            assert jnp.allclose(
                c_legacy[:n_compare], c_new[:n_compare], rtol=5e-2, atol=1e-9
            ), f"mismatch at wl={wl}, t={t}: legacy={c_legacy}, new={c_new}"
