"""Parity: scene.disk.ExovistaDisk ≡ legacy.Disk for flux density."""

from __future__ import annotations

import jax.numpy as jnp


def test_parity_disk_contrast_map(legacy_system, scene_system):
    """Surface-brightness contrast map at a fixed wavelength matches."""
    wl = jnp.array(600.0)
    t = jnp.array(0.0)
    legacy_contrast = legacy_system.disk._contrast_interp(wl)
    new_contrast = scene_system.disk.surface_brightness(wl, t)
    assert jnp.allclose(legacy_contrast, new_contrast, rtol=1e-6)


def test_parity_disk_flux_density(legacy_system, scene_system):
    """Disk flux density (contrast × star) matches."""
    wl = jnp.array(600.0)
    t = jnp.array(legacy_system.star._times_jd[0])
    f_legacy = legacy_system.disk.spec_flux_density(wl, t)
    c_new = scene_system.disk.surface_brightness(wl, t)
    f_new = c_new * scene_system.star.spec_flux_density(wl, t)
    assert jnp.allclose(f_new, f_legacy, rtol=1e-5)


def test_parity_disk_pixel_scale(legacy_system, scene_system):
    """pixel_scale_arcsec matches between legacy and scene disks."""
    assert jnp.isclose(
        legacy_system.disk.pixel_scale_arcsec,
        scene_system.disk.pixel_scale_arcsec,
        rtol=1e-9,
    )


def test_parity_disk_spatial_extent(legacy_system, scene_system):
    """spatial_extent() returns the same (width, height) for both disks."""
    w_l, h_l = legacy_system.disk.spatial_extent()
    w_n, h_n = scene_system.disk.spatial_extent()
    assert jnp.isclose(w_l, w_n, rtol=1e-9)
    assert jnp.isclose(h_l, h_n, rtol=1e-9)
