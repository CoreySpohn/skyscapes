"""skyscapes.disk.ExovistaParametricDisk -- ExoVista forward model."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erfcx
from zodi import DensityDust

from skyscapes.disk import AbstractDisk, ExovistaParametricDisk
from skyscapes.disk._los import los_integrate_scattered
from skyscapes.disk.exovista_parametric import exovista_density
from tests.conftest import radiance, sky_grid

_GRID = sky_grid(51, 51, 0.1)


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
        n_slices_los=41,
        amplitude=DensityDust(
            level=jnp.array(3.0),
            profile_id="exovista-parametric",
            calibration_id="unit-profile",
        ),
    )
    defaults.update(overrides)
    return ExovistaParametricDisk(**defaults)


def _render(
    d,
    wavelength_nm=550.0,
    time_d=0.0,
    inclination_deg=60.0,
    position_angle_deg=90.0,
    *,
    grid=_GRID,
):
    """Render on an explicit caller-owned grid."""
    return radiance(
        d, wavelength_nm, time_d, inclination_deg, position_angle_deg, grid=grid
    )


def test_is_abstract():
    """ExovistaParametricDisk satisfies the AbstractDisk interface."""
    assert isinstance(make_disk(), AbstractDisk)


def test_shape_and_finiteness():
    """photon_radiance returns finite non-negative output of correct shape."""
    sb = _render(make_disk())
    assert sb.shape == (51, 51)
    assert bool(jnp.all(jnp.isfinite(sb)))
    assert bool(jnp.all(sb >= 0.0))
    assert float(sb.sum()) > 0.0


def test_pole_on_is_axisymmetric():
    """Pole-on rendering is axisymmetric."""
    sb = _render(make_disk(), position_angle_deg=90.0 - (0.0), inclination_deg=0.0)
    sb_rot = jnp.rot90(sb)
    peak = float(sb.max())
    assert float(jnp.max(jnp.abs(sb - sb_rot)) / peak) < 1e-4


def test_ring_peak_at_r0():
    """Within the Gaussian-ring region, brightness peaks near r0_AU.

    The full disk also has a PR-drag interior + r^-1.5 halo, so we
    restrict the test to pixels within a few ring widths of r0.
    """
    grid = sky_grid(101, 101, 0.1)
    r0 = 25.0
    dror = 0.1
    d = make_disk(r0_AU=jnp.array(r0), dror=jnp.array(dror))
    sb = _render(d, position_angle_deg=90.0 - (0.0), inclination_deg=0.0, grid=grid)
    ny, nx = sb.shape
    yy, xx = jnp.mgrid[:ny, :nx]
    r_pix = jnp.sqrt((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2)
    px_AU = grid.step * 10.0
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
    grid = sky_grid(201, 201, 0.05)
    d = make_disk(
        r0_AU=jnp.array(10.0),
        dror=jnp.array(0.1),
        rmin_AU=jnp.array(0.5),
        rmax_AU=jnp.array(60.0),
        rinner_AU=jnp.array(0.5),
    )
    sb = _render(d, inclination_deg=0.0, grid=grid)
    ny, nx = sb.shape
    yy, xx = jnp.mgrid[:ny, :nx]
    r_pix = jnp.sqrt((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2)
    px_AU = grid.step * 10.0
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


def test_density_is_continuous_at_the_ring_edges():
    """Fails if a sample exactly on ``r0 -+ dr`` falls outside every branch.

    The ring, halo and inward-decay branches all equal ``exp(-0.5)`` at the
    ring edges. A line-of-sight node at the midplane crossing lands exactly
    on an edge for some pixels (here ``r0 + dr = 22`` AU), and a density of
    one there, the piecewise fallback, brightened that pixel by 19 percent.
    """
    edges = jnp.array([18.0, 22.0])
    just_inside = jnp.array([18.0 + 1e-9, 22.0 - 1e-9])
    parameters = dict(
        r0_AU=jnp.array(20.0),
        dror=jnp.array(0.1),
        rinner_AU=jnp.array(2.0),
        hor=jnp.array(0.05),
        density_multiplier=jnp.array(3.0),
        eta=jnp.array(1.0),
    )
    on_edge = exovista_density(edges, jnp.zeros(2), **parameters)
    inside = exovista_density(just_inside, jnp.zeros(2), **parameters)
    np.testing.assert_allclose(on_edge, inside, rtol=1e-8)


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
    sb_fwd = _render(d_fwd_heavy, position_angle_deg=90.0 - (0.0), inclination_deg=60.0)
    sb_iso = _render(d_iso_heavy, position_angle_deg=90.0 - (0.0), inclination_deg=60.0)
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
    low = np.asarray(_render(d, position_angle_deg=90.0 - (0.0), inclination_deg=incl))
    high = np.asarray(
        _render(d, position_angle_deg=90.0 - (0.0), inclination_deg=180.0 - incl)
    )
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
        return radiance(
            disk,
            jnp.array(550.0),
            jnp.array(0.0),
            jnp.array(60.0),
            90.0 - (jnp.array(0.0)),
            dist_pc=10.0,
            grid=_GRID,
        )

    sb_eager = _render(d)
    # jit and eager fuse the LOS quadrature reduction differently, so the
    # near-zero outer-disk pixels agree only to the field scale, not to 1e-12.
    atol = 1e-6 * float(jnp.max(sb_eager))
    assert jnp.allclose(f(d), sb_eager, rtol=1e-5, atol=atol)


@pytest.mark.parametrize("incl", [89.9, 90.0, 90.3])
def test_edge_on_has_supplementary_symmetry(incl):
    """Fails if the shared edge-on extension breaks ExovistaParametricDisk geometry."""
    d = make_disk()
    image = jax.jit(lambda i: _render(d, inclination_deg=i))(jnp.asarray(incl))
    mirrored = _render(d, inclination_deg=180.0 - incl)
    np.testing.assert_allclose(image, mirrored[::-1], rtol=1e-9, atol=1e-12)
    assert jnp.all(jnp.isfinite(image)) and image.max() > 0


def test_legendre_window_captures_the_outer_halo():
    """Fails if the local Gaussian column is truncated in the outer halo.

    Exact-algebra basis: integral exp(-z^2/(2 sigma^2))/(r^2+z^2) dz
    is pi/r * erfcx(r/(sqrt(2)*sigma)). Allow 3e-7 relative for omitted
    exp(-16) tails at four local sqrt(2)*sigma heights and quadrature error.
    """
    radii = jnp.array([[55.0, 65.0, 75.0]])
    hor = 0.05

    def density(r, z, valid):
        return jnp.where(
            valid, (r / 20.0) ** -1.5 * jnp.exp(-0.5 * (z / (hor * r)) ** 2), 0.0
        )

    image = los_integrate_scattered(
        density,
        lambda cosine: jnp.ones_like(cosine),
        incl_deg=0.0,
        position_angle_deg=0.0,
        rmin_AU=0.5,
        rmax_AU=80.0,
        east_AU=radii,
        north_AU=jnp.zeros_like(radii),
        n_slices_los=48,
        scale_height_fn=lambda r: jnp.sqrt(2.0) * hor * r,
    )
    expected = (np.asarray(radii) / 20.0) ** -1.5 * np.pi / np.asarray(radii)
    expected *= erfcx(1.0 / (np.sqrt(2.0) * hor))
    np.testing.assert_allclose(image, expected, rtol=3e-7, atol=0.0)


def test_wavelength_vmap_returns_cube():
    """Vmap over wavelength returns (n_wave, ny, nx)."""
    d = make_disk(
        wavelengths_nm=jnp.array([500.0, 700.0, 900.0]),
        Ag_grid=jnp.array([0.4, 0.5, 0.6]),
    )
    wls = jnp.array([550.0, 650.0, 850.0])
    cube = jax.vmap(
        partial(radiance, d, grid=_GRID, dist_pc=10.0), in_axes=(0, None, None, None)
    )(wls, jnp.array(0.0), jnp.array(60.0), jnp.array(0.0))
    assert cube.shape == (3, _GRID.shape[0], _GRID.shape[1])
    assert bool(jnp.all(jnp.isfinite(cube)))
