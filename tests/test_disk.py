"""skyscapes.disk -- AbstractDisk and ExovistaDisk."""

from __future__ import annotations

import jax.numpy as jnp

from skyscapes.disk import AbstractDisk, ExovistaDisk


def test_exovista_disk_flux_shape():
    """ExovistaDisk surface brightness has the cube's spatial shape."""
    wl = jnp.linspace(400.0, 1000.0, 5)
    cube = jnp.ones((wl.size, 4, 6)) * 1e-7
    d = ExovistaDisk(
        pixel_scale_arcsec=0.01,
        wavelengths_nm=wl,
        contrast_cube=cube,
    )
    sb = d.surface_brightness(
        wavelength_nm=jnp.array(500.0),
        time_jd=jnp.array(0.0),
        incl_deg=jnp.array(60.0),
        pa_deg=jnp.array(0.0),
    )
    assert sb.shape == (4, 6)
    assert jnp.allclose(sb, 1e-7, rtol=1e-5)


def test_exovista_disk_is_abstract():
    """ExovistaDisk satisfies the AbstractDisk interface."""
    wl = jnp.linspace(400.0, 1000.0, 5)
    cube = jnp.zeros((wl.size, 2, 2))
    d = ExovistaDisk(
        pixel_scale_arcsec=0.01,
        wavelengths_nm=wl,
        contrast_cube=cube,
    )
    assert isinstance(d, AbstractDisk)


# Removed in the 2026-05-16 sim_disk redesign:
# - UniformDisk and ParametricDisk classes: speculative scaffolds with no
#   production consumers. Test fixtures that need a cheap disk should use
#   ExovistaDisk with a small synthetic cube (see test_scene_system.py).
# - Per-disk inclination_deg / position_angle_deg fields: canonical disk
#   midplane orientation now lives on System.midplane_inc_deg /
#   System.midplane_pa_deg. See brain/specs/2026-05-16-sim-disk-redesign-design.md.
