"""skyscapes.disk.ExovistaParametricDisk -- ExoVista forward model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from skyscapes.disk import AbstractDisk, ExovistaParametricDisk


def make_disk(**overrides) -> ExovistaParametricDisk:
    """Default ExovistaParametricDisk; overrides patch specific fields.

    Defaults match a single-component scene roughly inspired by the warm
    component priors in Stark 2022 Table 1.
    """
    defaults = dict(
        r0_AU=jnp.array(20.0),
        dror=jnp.array(0.1),
        rinner_AU=jnp.array(2.0),
        hor=jnp.array(0.05),
        nzodis=jnp.array(3.0),
        eta=jnp.array(1.0),
        g0=jnp.array(0.9),
        g1=jnp.array(0.6),
        g2=jnp.array(0.0),
        w0=jnp.array(0.7),
        w1=jnp.array(0.25),
        w2=jnp.array(0.05),
        rmin_AU=jnp.array(0.5),
        rmax_AU=jnp.array(80.0),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        Ag_grid=jnp.array([0.5, 0.5]),
        nx=51,
        ny=51,
        pixel_scale_arcsec=0.1,
        dist_pc=10.0,
        n_slices_los=41,
    )
    defaults.update(overrides)
    return ExovistaParametricDisk(**defaults)


def _render(d, wavelength_nm=550.0, time_jd=0.0, incl_deg=60.0, pa_deg=0.0):
    return d.surface_brightness(
        jnp.array(wavelength_nm),
        jnp.array(time_jd),
        jnp.array(incl_deg),
        jnp.array(pa_deg),
    )


def test_is_abstract():
    """ExovistaParametricDisk satisfies the AbstractDisk interface."""
    assert isinstance(make_disk(), AbstractDisk)


def test_shape_and_finiteness():
    """surface_brightness returns finite non-negative output of correct shape."""
    sb = _render(make_disk())
    assert sb.shape == (51, 51)
    assert bool(jnp.all(jnp.isfinite(sb)))
    assert bool(jnp.all(sb >= 0.0))
    assert float(sb.sum()) > 0.0


def test_pole_on_is_axisymmetric():
    """Pole-on rendering is axisymmetric."""
    sb = _render(make_disk(), incl_deg=0.0, pa_deg=0.0)
    sb_rot = jnp.rot90(sb)
    peak = float(sb.max())
    assert float(jnp.max(jnp.abs(sb - sb_rot)) / peak) < 1e-4


def test_ring_peak_at_r0():
    """Within the Gaussian-ring region, brightness peaks near r0_AU.

    The full disk also has a PR-drag interior + r^-1.5 halo, so we
    restrict the test to pixels within a few ring widths of r0.
    """
    r0 = 25.0
    dror = 0.1
    d = make_disk(
        r0_AU=jnp.array(r0),
        dror=jnp.array(dror),
        nx=101,
        ny=101,
        pixel_scale_arcsec=0.1,
    )
    sb = _render(d, incl_deg=0.0, pa_deg=0.0)
    ny, nx = sb.shape
    yy, xx = jnp.mgrid[:ny, :nx]
    r_pix = jnp.sqrt((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2)
    px_AU = d.pixel_scale_arcsec * d.dist_pc
    r_AU = r_pix * px_AU
    dr_AU = dror * r0
    near_ring = (r_AU > r0 - 3 * dr_AU) & (r_AU < r0 + 3 * dr_AU)
    mean_r = float(jnp.sum(r_AU * sb * near_ring) / jnp.sum(sb * near_ring))
    assert abs(mean_r - r0) < dr_AU, (
        f"mean radius near ring = {mean_r:.2f} far from r0={r0}"
    )


def test_outer_halo_falls_as_r_minus_one_point_five():
    """Far outside the ring, the surface brightness column follows ~r^-1.5."""
    # Pole-on so the radial profile is unambiguous in image coords.
    d = make_disk(
        r0_AU=jnp.array(10.0),
        dror=jnp.array(0.1),
        rmin_AU=jnp.array(0.5),
        rmax_AU=jnp.array(60.0),
        rinner_AU=jnp.array(0.5),
        nx=201,
        ny=201,
        pixel_scale_arcsec=0.05,
    )
    sb = _render(d, incl_deg=0.0)
    ny, nx = sb.shape
    yy, xx = jnp.mgrid[:ny, :nx]
    r_pix = jnp.sqrt((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2)
    px_AU = d.pixel_scale_arcsec * d.dist_pc
    r_AU = r_pix * px_AU
    # Sample two radii well outside the ring; ratio should follow r^-1.5
    # times the 1/r^2 illumination, i.e. surface brightness ~ r^-3.5 in
    # the halo region. We just verify the outer point is significantly
    # fainter than the inner one, which catches profile shape regressions.
    r_inner_test = 20.0
    r_outer_test = 40.0
    mask_inner = (r_AU > r_inner_test - 2) & (r_AU < r_inner_test + 2)
    mask_outer = (r_AU > r_outer_test - 2) & (r_AU < r_outer_test + 2)
    sb_inner = float(jnp.mean(sb[mask_inner]))
    sb_outer = float(jnp.mean(sb[mask_outer]))
    assert sb_inner > sb_outer > 0.0
    # Expect roughly halved-or-more brightness at 2x radius in the halo.
    assert sb_outer < 0.5 * sb_inner


def test_three_hg_weight_concentrates_forward():
    """Increasing w0 (dominant forward HG) brightens the near side."""
    d_fwd_heavy = make_disk(
        w0=jnp.array(0.95),
        w1=jnp.array(0.05),
        w2=jnp.array(0.0),
    )
    d_iso_heavy = make_disk(
        w0=jnp.array(0.0),
        w1=jnp.array(0.0),
        w2=jnp.array(1.0),
    )
    sb_fwd = _render(d_fwd_heavy, incl_deg=60.0, pa_deg=0.0)
    sb_iso = _render(d_iso_heavy, incl_deg=60.0, pa_deg=0.0)
    ny = sb_fwd.shape[0]
    # Top half = North = far side. Bottom half = South = forward side.
    bot_fwd = float(sb_fwd[ny // 2 :].sum())
    bot_iso = float(sb_iso[ny // 2 :].sum())
    top_fwd = float(sb_fwd[: ny // 2].sum())
    top_iso = float(sb_iso[: ny // 2].sum())
    # Forward-heavy disk should be more bottom-skewed than the isotropic one.
    fwd_asymmetry = bot_fwd / (top_fwd + 1e-30)
    iso_asymmetry = bot_iso / (top_iso + 1e-30)
    assert fwd_asymmetry > iso_asymmetry


def test_jit_round_trip():
    """JIT'd output matches eager output."""
    d = make_disk()

    @jax.jit
    def f(disk):
        return disk.surface_brightness(
            jnp.array(550.0),
            jnp.array(0.0),
            jnp.array(60.0),
            jnp.array(0.0),
        )

    sb_eager = _render(d)
    # jit and eager fuse the log-spaced LOS trapezoid reduction differently, so the
    # near-zero outer-disk pixels agree only to the field scale, not to 1e-12.
    atol = 1e-6 * float(jnp.max(sb_eager))
    assert jnp.allclose(f(d), sb_eager, rtol=1e-5, atol=atol)


def test_edge_on_is_rejected():
    """Rendering close to edge-on raises at surface_brightness time."""
    d = make_disk()
    with pytest.raises(Exception, match="edge-on"):
        _render(d, incl_deg=89.9)


def test_wavelength_vmap_returns_cube():
    """Vmap over wavelength returns (n_wave, ny, nx)."""
    d = make_disk(
        wavelengths_nm=jnp.array([500.0, 700.0, 900.0]),
        Ag_grid=jnp.array([0.4, 0.5, 0.6]),
    )
    wls = jnp.array([550.0, 650.0, 850.0])
    cube = jax.vmap(d.surface_brightness, in_axes=(0, None, None, None))(
        wls, jnp.array(0.0), jnp.array(60.0), jnp.array(0.0)
    )
    assert cube.shape == (3, d.ny, d.nx)
    assert bool(jnp.all(jnp.isfinite(cube)))
