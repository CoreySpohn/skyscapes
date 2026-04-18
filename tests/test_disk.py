"""skyscapes.disk -- AbstractDisk, UniformDisk, ParametricDisk, ExovistaDisk."""

from __future__ import annotations

import jax.numpy as jnp

from skyscapes.disk import (
    AbstractDisk,
    ExovistaDisk,
    ParametricDisk,
    UniformDisk,
)


def test_uniform_disk_is_abstract():
    """UniformDisk satisfies the AbstractDisk interface."""
    d = UniformDisk(contrast=jnp.array(1e-6), pixel_scale_arcsec=0.01, shape=(4, 4))
    assert isinstance(d, AbstractDisk)


def test_uniform_disk_surface_brightness_constant():
    """UniformDisk returns the same contrast at every pixel."""
    d = UniformDisk(contrast=jnp.array(2e-6), pixel_scale_arcsec=0.01, shape=(3, 5))
    sb = d.surface_brightness(wavelength_nm=jnp.array(500.0), time_jd=jnp.array(0.0))
    assert sb.shape == (3, 5)
    assert jnp.allclose(sb, 2e-6)


def test_uniform_disk_spatial_extent():
    """UniformDisk.spatial_extent matches (nx, ny) * pixel_scale."""
    d = UniformDisk(contrast=jnp.array(1e-6), pixel_scale_arcsec=0.02, shape=(5, 10))
    width, height = d.spatial_extent()
    assert jnp.isclose(width, 0.02 * 10)
    assert jnp.isclose(height, 0.02 * 5)


def test_parametric_disk_is_abstract():
    """ParametricDisk satisfies the AbstractDisk interface."""
    d = ParametricDisk(
        contrast_0=jnp.array(1e-6),
        r0_arcsec=jnp.array(0.1),
        alpha=jnp.array(-2.0),
        pixel_scale_arcsec=0.01,
        shape=(5, 5),
    )
    assert isinstance(d, AbstractDisk)


def test_parametric_disk_power_law_radial():
    """Off-center pixels follow the analytical power law ``c0 * (r/r0)^alpha``."""
    contrast_0 = 1e-6
    r0 = 0.1
    alpha = -2.0
    pixel_scale = 0.01
    d = ParametricDisk(
        contrast_0=jnp.array(contrast_0),
        r0_arcsec=jnp.array(r0),
        alpha=jnp.array(alpha),
        pixel_scale_arcsec=pixel_scale,
        shape=(5, 5),
    )
    sb = d.surface_brightness(wavelength_nm=jnp.array(500.0), time_jd=jnp.array(0.0))
    assert sb.shape == (5, 5)
    # Pixel (2, 3) is one step right of center -> r = 1 * pixel_scale.
    expected_edge = contrast_0 * (pixel_scale / r0) ** alpha
    assert jnp.allclose(sb[2, 3], expected_edge, rtol=1e-5)
    # Pixel (2, 4) is two steps right of center -> r = 2 * pixel_scale.
    expected_edge2 = contrast_0 * (2 * pixel_scale / r0) ** alpha
    assert jnp.allclose(sb[2, 4], expected_edge2, rtol=1e-5)
    # Radial falloff: pixel at r = 2 * ps has lower sb than at r = ps (alpha < 0).
    assert sb[2, 4] < sb[2, 3]


def test_exovista_disk_flux_shape():
    """ExovistaDisk surface brightness has the cube's spatial shape."""
    wl = jnp.linspace(400.0, 1000.0, 5)
    cube = jnp.ones((wl.size, 4, 6)) * 1e-7
    d = ExovistaDisk(
        pixel_scale_arcsec=0.01,
        wavelengths_nm=wl,
        contrast_cube=cube,
    )
    sb = d.surface_brightness(wavelength_nm=jnp.array(500.0), time_jd=jnp.array(0.0))
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


def test_uniform_disk_midplane_defaults():
    """UniformDisk stores midplane fields; defaults to zero when unspecified."""
    d = UniformDisk(contrast=jnp.array(1e-6), pixel_scale_arcsec=0.01, shape=(3, 3))
    assert d.inclination_deg == 0.0
    assert d.position_angle_deg == 0.0
    d2 = UniformDisk(
        contrast=jnp.array(1e-6),
        pixel_scale_arcsec=0.01,
        shape=(3, 3),
        inclination_deg=30.0,
        position_angle_deg=45.0,
    )
    assert d2.inclination_deg == 30.0
    assert d2.position_angle_deg == 45.0


def test_parametric_disk_midplane_defaults():
    """ParametricDisk stores midplane fields; defaults to zero when unspecified."""
    d = ParametricDisk(
        contrast_0=jnp.array(1e-6),
        r0_arcsec=jnp.array(0.1),
        alpha=jnp.array(-2.0),
        pixel_scale_arcsec=0.01,
        shape=(3, 3),
        inclination_deg=60.0,
        position_angle_deg=120.0,
    )
    assert d.inclination_deg == 60.0
    assert d.position_angle_deg == 120.0


def test_exovista_disk_midplane_defaults():
    """ExovistaDisk stores midplane fields; defaults to zero when unspecified."""
    wl = jnp.linspace(400.0, 1000.0, 3)
    cube = jnp.zeros((wl.size, 2, 2))
    d = ExovistaDisk(
        pixel_scale_arcsec=0.01,
        wavelengths_nm=wl,
        contrast_cube=cube,
    )
    assert d.inclination_deg == 0.0
    assert d.position_angle_deg == 0.0
    d2 = ExovistaDisk(
        pixel_scale_arcsec=0.01,
        wavelengths_nm=wl,
        contrast_cube=cube,
        inclination_deg=15.0,
        position_angle_deg=200.0,
    )
    assert d2.inclination_deg == 15.0
    assert d2.position_angle_deg == 200.0
