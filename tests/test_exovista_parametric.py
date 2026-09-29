"""skyscapes.disk.ExovistaParametricDisk -- ExoVista forward model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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
    # Row index increases along +y. At pa = 0 and incl < 90 the near,
    # forward-scattering half lies along +y (rows ny // 2 onward) and the far
    # half along -y (rows below ny // 2).
    near_fwd = float(sb_fwd[ny // 2 :].sum())
    near_iso = float(sb_iso[ny // 2 :].sum())
    far_fwd = float(sb_fwd[: ny // 2].sum())
    far_iso = float(sb_iso[: ny // 2].sum())
    # The forward-heavy disk should be more near-skewed than the isotropic one.
    fwd_asymmetry = near_fwd / (far_fwd + 1e-30)
    iso_asymmetry = near_iso / (far_iso + 1e-30)
    assert fwd_asymmetry > iso_asymmetry


@pytest.mark.parametrize("incl", [30.0, 60.0, 80.0])
def test_supplementary_inclination_is_the_mirrored_physical_map(incl):
    """``incl`` and ``180 - incl`` render one disk mirrored across its line of nodes.

    Reflecting the sky through the line of nodes (``y -> -y`` at ``pa = 0``)
    maps the disk normal at ``incl`` onto minus the normal at ``180 - incl``;
    the density is even in height and the phase function depends only on the
    depth toward the observer, so the maps are row-reversed copies. The
    forward-peaked three-component phase function brightens the near half,
    which lies along ``+y`` below 90 degrees and along ``-y`` above it.
    Tolerance basis: floating-point.
    """
    d = make_disk()
    low = np.asarray(_render(d, incl_deg=incl, pa_deg=0.0))
    high = np.asarray(_render(d, incl_deg=180.0 - incl, pa_deg=0.0))
    for sb in (low, high):
        assert np.all(np.isfinite(sb))
        assert np.all(sb >= 0.0), f"negative radiance, min {sb.min():.3g}"
        assert sb.sum() > 0.0
    np.testing.assert_allclose(high, low[::-1], rtol=1e-9, atol=1e-12 * low.max())
    c = low.shape[0] // 2
    assert low[c + 1 :].sum() > low[:c].sum(), "near (+y) half not brighter"
    assert high[:c].sum() > high[c + 1 :].sum(), "near (-y) half not brighter"


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


@pytest.mark.parametrize("incl", [89.9, 90.0, 90.3])
def test_edge_on_is_rejected(incl):
    """At and near edge-on (either side of 90) the render raises, also under jit."""
    d = make_disk()
    with pytest.raises(Exception, match="edge-on"):
        _render(d, incl_deg=incl)
    with pytest.raises(Exception, match="edge-on"):
        jax.jit(lambda disk: _render(disk, incl_deg=incl))(d)


def _apertures(img: np.ndarray, incl: float, px_AU: float) -> dict[str, float]:
    """Total flux, near-side minor-axis and ansa aperture fluxes of the ring.

    Apertures have a 3-pixel radius and sit on the 20 AU ring at ``pa = 0``;
    the near half is along +y below 90 deg and along -y above.
    """
    c = img.shape[0] // 2
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]] - c
    cos_i = np.cos(np.radians(incl))
    near_row = np.sign(cos_i) * round(20.0 * abs(cos_i) / px_AU)
    ansa_col = round(20.0 / px_AU)
    near = xx**2 + (yy - near_row) ** 2 <= 9
    ansa = (xx - ansa_col) ** 2 + yy**2 <= 9
    return {
        "total": float(img.sum()),
        "near": float(img[near].sum()),
        "ansa": float(img[ansa].sum()),
    }


@pytest.mark.parametrize("incl", [0.0, 60.0, 80.0, 120.0])
def test_los_node_ladder_converges_at_second_order(incl):
    """Three-level LOS node ladder (21, 41, 81 nodes) on the default ring.

    The node spacing halves in a mapped variable at each level and the
    trapezoid rule is second order in it: total and aperture fluxes must show
    an observed order in [1.5, 3] (above 2 is cancellation), with a
    Richardson error estimate at 41 nodes below 1.5 percent. No order is
    claimed for the pixel-wise image (truncation edges); its max-norm change
    must more than halve per level and stay below 1.5 percent of the peak at
    the finest step. Tolerance basis: spike (orders 2.0-2.8; 41-node errors
    0.7-1.0 percent, matching the 641-node reference).
    """
    imgs = {
        n: np.asarray(_render(make_disk(n_slices_los=n), incl_deg=incl))
        for n in (21, 41, 81)
    }
    for img in imgs.values():
        assert np.all(np.isfinite(img)) and np.all(img >= 0.0)
    resp = {n: _apertures(img, incl, px_AU=1.0) for n, img in imgs.items()}
    for key in resp[81]:
        d_coarse = abs(resp[21][key] - resp[41][key])
        d_fine = abs(resp[41][key] - resp[81][key])
        order = np.log2(d_coarse / d_fine)
        assert 1.5 <= order <= 3.0, f"{key}: observed order {order:.2f}"
        err_41 = d_fine * 2.0**order / (2.0**order - 1.0) / abs(resp[81][key])
        assert err_41 < 0.015, f"{key}: 41-node error {err_41:.2e}"
    peak = imgs[81].max()
    e_coarse = np.abs(imgs[21] - imgs[41]).max() / peak
    e_fine = np.abs(imgs[41] - imgs[81]).max() / peak
    assert e_fine < 0.5 * e_coarse
    assert e_fine < 0.015


def _halo_view(n_scale_heights: float) -> ExovistaParametricDisk:
    """The default ring on an image that reaches ``rmax_AU`` (2 AU/pixel)."""
    return make_disk(
        nx=81,
        ny=81,
        pixel_scale_arcsec=0.2,
        n_slices_los=161,
        n_scale_heights=n_scale_heights,
    )


def _radial_band(img: np.ndarray, lo_AU: float, hi_AU: float) -> float:
    """Pole-on flux between two disk radii on the 2 AU/pixel image."""
    c = img.shape[0] // 2
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]] - c
    r_AU = 2.0 * np.hypot(xx, yy)
    return float(img[(r_AU >= lo_AU) & (r_AU <= hi_AU)].sum())


def test_vertical_support_ladder_converges_on_the_ring():
    """Support ladder (1.5, 3, 6 ring scale heights) against 24, on the ring.

    ``ExovistaParametricDisk`` sizes the LOS window from the scale height at
    the ring center ``r0_AU``. On the ring (18-22 AU) the ladder must shrink
    at each level and the default (6) must agree with 24 to 1e-3, pole-on
    and at 60 deg. Tolerance basis: spike (default-to-24 changes of 8e-5
    pole-on and 2.6e-4 at 60 deg, at 161 nodes).
    """
    for incl in (0.0, 60.0):
        ring = {}
        for k in (1.5, 3.0, 6.0, 24.0):
            img = np.asarray(_render(_halo_view(k), incl_deg=incl))
            assert np.all(img >= 0.0)
            if incl == 0.0:
                ring[k] = _radial_band(img, 18.0, 22.0)
            else:
                ring[k] = _apertures(img, incl, px_AU=2.0)["ansa"]
        assert abs(ring[3.0] - ring[6.0]) < abs(ring[1.5] - ring[3.0])
        assert abs(ring[6.0] / ring[24.0] - 1.0) < 1e-3, f"incl={incl}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The LOS window is sized from the scale height at the ring center, so "
        "at radius r it spans only n_scale_heights * r0 / r local scale "
        "heights and keeps erf(n_scale_heights * r0 / (sqrt(2) r)) of the "
        "halo column (0.95 at 3 r0 and 0.87 at 4 r0 for the default 6)."
    ),
)
def test_vertical_support_converges_in_the_outer_halo():
    """The default support must capture the halo column (55-75 AU) to 1e-3.

    Pole-on, the halo band's flux at the default support is compared with a
    support four times larger. Known limitation: the band loses ~6 percent.
    """
    default = _radial_band(
        np.asarray(_render(_halo_view(6.0), incl_deg=0.0)), 55.0, 75.0
    )
    wide = _radial_band(np.asarray(_render(_halo_view(24.0), incl_deg=0.0)), 55.0, 75.0)
    assert abs(default / wide - 1.0) < 1e-3


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
