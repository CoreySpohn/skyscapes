"""skyscapes.disk.GraterDisk -- Augereau 1999 scattered-light disk."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyscapes.disk import AbstractDisk, GraterDisk


def eqx_replace(module, **updates):
    """Tiny helper: replace eqx.Module fields without importing eqx here."""
    return eqx.tree_at(
        lambda m: [getattr(m, k) for k in updates],
        module,
        list(updates.values()),
    )


def make_disk(g_HG: float = 0.3, Ag: float = 0.5, **overrides) -> GraterDisk:
    """Build a default-parameter GraterDisk; overrides patch specific fields.

    ``g_HG`` and ``Ag`` are convenience scalars expanded to constant grids
    over the default wavelength range.
    """
    defaults = dict(
        sma_AU=jnp.array(50.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-5.0),
        ksi0_AU=jnp.array(1.0),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(5.0),
        rmax_AU=jnp.array(200.0),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([g_HG, g_HG]),
        Ag_grid=jnp.array([Ag, Ag]),
        nx=51,
        ny=51,
        pixel_scale_arcsec=0.2,  # px_AU = 2 AU at 10 pc -> image covers ~+/-50 AU
        dist_pc=10.0,
        n_slices_los=31,
    )
    defaults.update(overrides)
    return GraterDisk(**defaults)


def _render(d: GraterDisk, wavelength_nm=500.0, time_jd=0.0, incl_deg=60.0, pa_deg=0.0):
    """Convenience: render a disk at default render-time geometry."""
    return d.surface_brightness(
        jnp.array(wavelength_nm),
        jnp.array(time_jd),
        jnp.array(incl_deg),
        jnp.array(pa_deg),
    )


def test_grater_disk_is_abstract():
    """GraterDisk satisfies the AbstractDisk interface."""
    assert isinstance(make_disk(), AbstractDisk)


def test_shape_and_finiteness():
    """surface_brightness returns the expected shape with finite, non-neg values."""
    sb = _render(make_disk())
    assert sb.shape == (51, 51)
    assert bool(jnp.all(jnp.isfinite(sb)))
    assert bool(jnp.all(sb >= 0.0))
    assert float(sb.sum()) > 0.0


def test_pole_on_is_axisymmetric():
    """incl=0 + pa=0 -> image equals its 90 deg rotation (within tol)."""
    sb = _render(make_disk(), incl_deg=0.0, pa_deg=0.0)
    sb_rot = jnp.rot90(sb)
    peak = float(sb.max())
    rel_err = float(jnp.max(jnp.abs(sb - sb_rot)) / peak)
    assert rel_err < 1e-4, f"pole-on disk not axisymmetric: rel_err={rel_err}"


def test_hg_sign_flip_swaps_asymmetry():
    """Flipping g_HG flips the forward/back asymmetry of an inclined disk."""
    sb_fwd = _render(make_disk(g_HG=0.3), incl_deg=60.0, pa_deg=0.0)
    sb_bwd = _render(make_disk(g_HG=-0.3), incl_deg=60.0, pa_deg=0.0)
    ny = sb_fwd.shape[0]
    top_fwd = float(sb_fwd[: ny // 2].sum())
    bot_fwd = float(sb_fwd[ny // 2 :].sum())
    top_bwd = float(sb_bwd[: ny // 2].sum())
    bot_bwd = float(sb_bwd[ny // 2 :].sum())
    assert (bot_fwd > top_fwd) != (bot_bwd > top_bwd)


@pytest.mark.parametrize("incl", [30.0, 60.0, 80.0])
def test_supplementary_inclination_is_the_mirrored_physical_map(incl):
    """``incl`` and ``180 - incl`` render one disk mirrored across its line of nodes.

    Reflecting the sky through the line of nodes (``y -> -y`` at ``pa = 0``)
    maps the disk normal at ``incl`` onto minus the normal at ``180 - incl``.
    The density is even in height and the scattering angle depends only on the
    depth toward the observer, so the two maps are row-reversed copies (the
    pixel grid is symmetric about its center). Forward scattering (g > 0) makes
    the near half brighter, and the near half lies along ``+y`` (rows above the
    center row) below 90 degrees and along ``-y`` above it; so the pair is
    neither identical nor negated, and a sign error in either the path measure
    or the geometry fails here. Tolerance basis: floating-point (the two
    renders differ only by the rounding of ``cos(180 - incl)`` against
    ``-cos(incl)``).
    """
    d = make_disk(g_HG=0.4)
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


def test_retrograde_orientation_is_the_same_midplane():
    """``(180 - incl, pa + 180)`` is the same midplane as ``(incl, pa)``.

    Turning the line of nodes by 180 degrees and supplementing the inclination
    flips only the side of the disk the observer calls its normal; the dust
    distribution is symmetric about the midplane, so the image is unchanged,
    pixel for pixel. Tolerance basis: floating-point.
    """
    d = make_disk(g_HG=0.4)
    ref = np.asarray(_render(d, incl_deg=55.0, pa_deg=30.0))
    same = np.asarray(_render(d, incl_deg=125.0, pa_deg=210.0))
    assert np.all(same >= 0.0)
    np.testing.assert_allclose(same, ref, rtol=1e-9, atol=1e-12 * ref.max())


def test_jit_round_trip():
    """JIT'd surface_brightness matches the eager output."""
    d = make_disk()

    @jax.jit
    def f(disk):
        return disk.surface_brightness(
            jnp.array(500.0), jnp.array(0.0), jnp.array(60.0), jnp.array(0.0)
        )

    sb_jit = f(d)
    sb_eager = _render(d, incl_deg=60.0)
    # jit and eager fuse the log-spaced LOS trapezoid reduction differently, so the
    # near-zero outer-disk pixels agree only to the field scale, not to 1e-12.
    atol = 1e-6 * float(jnp.max(sb_eager))
    assert jnp.allclose(sb_jit, sb_eager, rtol=1e-5, atol=atol)


def test_grad_through_sma():
    """jax.grad of mean brightness wrt sma_AU is finite."""
    d = make_disk()

    def loss(sma):
        new_d = eqx_replace(d, sma_AU=sma)
        return _render(new_d).mean()

    g = jax.grad(loss)(jnp.array(50.0))
    assert jnp.isfinite(g)


def test_wavelength_dependent_g_HG():
    """Linearly varying g_HG(lambda) reproduces scalar HG at each endpoint."""
    g_blue, g_red = 0.6, 0.1
    d_var = make_disk(
        wavelengths_nm=jnp.array([500.0, 900.0]),
        g_HG_grid=jnp.array([g_blue, g_red]),
    )
    d_blue = make_disk(g_HG=g_blue)
    d_red = make_disk(g_HG=g_red)
    assert jnp.allclose(
        _render(d_var, wavelength_nm=500.0),
        _render(d_blue, wavelength_nm=500.0),
        rtol=1e-5,
    )
    assert jnp.allclose(
        _render(d_var, wavelength_nm=900.0),
        _render(d_red, wavelength_nm=900.0),
        rtol=1e-5,
    )


@pytest.mark.parametrize("incl", [88.0, 92.0])
def test_high_inclination_is_finite(incl):
    """A highly inclined (but not edge-on) disk renders finite and nonnegative.

    The LOS quadrature is log-spaced and concentrated at the midplane crossing
    (matching the GRaTeR-JAX reference), so the model stays accurate well past
    the old arctan(rmax/zmax) guard (~83 deg for this disk). 88 and 92 deg
    (|cos| = 0.035) are highly inclined on either side of 90 but far from the
    cos_i -> 0 singularity.
    """
    sb = _render(make_disk(), incl_deg=incl)
    assert sb.shape == (51, 51)
    assert bool(jnp.all(jnp.isfinite(sb)))
    assert bool(jnp.all(sb >= 0.0))
    assert float(sb.sum()) > 0.0


@pytest.mark.parametrize("incl", [89.9, 90.0, 90.3])
def test_edge_on_render_is_rejected(incl):
    """At and near edge-on the render raises instead of returning a number.

    The LOS window about the midplane crossing has half-width
    ``zmax / |cos(incl)|``; at exactly 90 deg ``cos`` rounds to ~6e-17 and a
    map built on that denominator is not a result. Both sides of 90 are
    rejected, eagerly and under ``jax.jit``.
    """
    d = make_disk()
    with pytest.raises(Exception, match="edge-on"):
        _render(d, incl_deg=incl)
    with pytest.raises(Exception, match="edge-on"):
        jax.jit(lambda disk: _render(disk, incl_deg=incl))(d)


# --------------------------------------------------------------------------
# LOS node and vertical support convergence
# --------------------------------------------------------------------------
#
# Representative disk: a narrow belt at 3 AU (1.5-5 AU) around a star at
# 10 pc, forward scattering (g = 0.4), on a 121-pixel image at 0.0085
# arcsec/pixel (0.085 AU/pixel), rendered with 41 LOS nodes for display.
# The precision such an image claims is set by its display: a 300:1
# logarithmic color scale over 256 levels resolves log10(300) / 256 dex, 2.2
# percent in brightness. Every discretization error at the display setting
# must sit inside that step.
_DISPLAY_PRECISION = 10.0 ** (np.log10(300.0) / 256.0) - 1.0
_BELT_PX_AU = 0.085


def _belt(n_slices_los: int, n_scale_heights: float = 6.0) -> GraterDisk:
    return GraterDisk(
        sma_AU=jnp.array(3.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-3.0),
        ksi0_AU=jnp.array(0.1),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(1.5),
        rmax_AU=jnp.array(5.0),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([0.4, 0.4]),
        Ag_grid=jnp.array([0.3, 0.3]),
        nx=121,
        ny=121,
        pixel_scale_arcsec=0.0085,
        dist_pc=10.0,
        n_slices_los=n_slices_los,
        n_scale_heights=n_scale_heights,
    )


def _belt_responses(img: np.ndarray, incl: float) -> dict[str, float]:
    """Total flux and three aperture fluxes of a belt image at ``pa = 0``.

    Apertures: 4-pixel radius on the near-side minor axis of the 3 AU ring
    (the near half is along +y below 90 deg and along -y above), 4-pixel
    radius on the 3 AU ansa, and 2-pixel radius at 4.6 AU on the major axis,
    where the flared layer is thickest relative to the LOS support.
    """
    c = img.shape[0] // 2
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]] - c
    cos_i = np.cos(np.radians(incl))
    near_row = np.sign(cos_i) * round(3.0 * abs(cos_i) / _BELT_PX_AU)
    ansa_col = round(3.0 / _BELT_PX_AU)
    outer_col = round(4.6 / _BELT_PX_AU)
    near = xx**2 + (yy - near_row) ** 2 <= 16
    ansa = (xx - ansa_col) ** 2 + yy**2 <= 16
    outer = (xx - outer_col) ** 2 + yy**2 <= 4
    return {
        "total": float(img.sum()),
        "near": float(img[near].sum()),
        "ansa": float(img[ansa].sum()),
        "outer": float(img[outer].sum()),
    }


@pytest.mark.parametrize("incl", [0.0, 60.0, 80.0, 120.0])
def test_los_node_ladder_converges_at_second_order(incl):
    """Three-level LOS node ladder (21, 41, 81 nodes) on the belt disk.

    The nodes are uniform in a mapped variable whose spacing halves at each
    level, and the trapezoid rule is second order in it; the integrated
    responses (total and aperture fluxes) must show an observed order in
    [1.5, 3] (higher than 2 is cancellation, not failure), and the Richardson
    error estimate at 41 nodes must sit inside the display precision. The
    image is a pixel-wise response with truncation edges in radius, so no
    order is claimed for it: its max-norm change must shrink by more than
    half per level and stay inside the display precision. Tolerance basis:
    spike (measured orders 2.0-2.5 and 41-node errors 0.7-1.0 percent at 0,
    60, 80 and 120 deg).
    """
    imgs = {
        n: np.asarray(_render(_belt(n), wavelength_nm=550.0, incl_deg=incl))
        for n in (21, 41, 81)
    }
    for img in imgs.values():
        assert np.all(np.isfinite(img)) and np.all(img >= 0.0)
    resp = {n: _belt_responses(img, incl) for n, img in imgs.items()}
    for key in resp[81]:
        d_coarse = abs(resp[21][key] - resp[41][key])
        d_fine = abs(resp[41][key] - resp[81][key])
        order = np.log2(d_coarse / d_fine)
        assert 1.5 <= order <= 3.0, f"{key}: observed order {order:.2f}"
        err_41 = d_fine * 2.0**order / (2.0**order - 1.0) / abs(resp[81][key])
        assert err_41 < _DISPLAY_PRECISION, f"{key}: 41-node error {err_41:.2e}"
    peak = imgs[81].max()
    e_coarse = np.abs(imgs[21] - imgs[41]).max() / peak
    e_fine = np.abs(imgs[41] - imgs[81]).max() / peak
    assert e_fine < 0.5 * e_coarse
    assert e_fine < _DISPLAY_PRECISION


@pytest.mark.parametrize("incl", [0.0, 60.0, 120.0])
def test_vertical_support_ladder_converges_at_the_outer_radius(incl):
    """Three-level support ladder (1, 2, 4 scale heights) and the default 6.

    ``GraterDisk`` sizes the LOS window from the flared scale height at
    ``rmax_AU``, so the window covers at least ``n_scale_heights`` local
    heights at every radius. For the Gaussian layer the truncation error
    falls faster than any power of the support; the ladder must shrink at
    each level, and the default must agree with a doubled support (12) to a
    tenth of the display precision in every integrated response, including
    the outer aperture, where the support is tightest. At 161 nodes the node
    error of those responses is ~6e-4 and moves by less than 1e-4 across the
    ladder. Pixels on the truncation edges carry a node error that moves with
    the window (0.25 percent of the peak), so the image is held to the
    display precision itself. Tolerance basis: spike (default-to-doubled
    changes below 3e-4 in the integrated responses).
    """
    imgs = {
        k: np.asarray(_render(_belt(161, k), wavelength_nm=550.0, incl_deg=incl))
        for k in (1.0, 2.0, 4.0, 6.0, 12.0)
    }
    resp = {k: _belt_responses(img, incl) for k, img in imgs.items()}
    for key in resp[12.0]:
        ref = resp[12.0][key]
        d12 = abs(resp[1.0][key] - resp[2.0][key])
        d24 = abs(resp[2.0][key] - resp[4.0][key])
        assert d24 < d12, f"{key}: support ladder does not shrink"
        rel = abs(resp[6.0][key] - ref) / abs(ref)
        assert rel < 0.1 * _DISPLAY_PRECISION, f"{key}: support error {rel:.2e}"
    peak = imgs[12.0].max()
    assert np.abs(imgs[6.0] - imgs[12.0]).max() / peak < _DISPLAY_PRECISION


def test_support_window_is_sized_at_the_outer_radius():
    """One scale height of support captures ``erf(rmax / r)`` of the column.

    Pole-on, each pixel's LOS runs along the disk normal at fixed radius
    ``r``, so truncating it at ``k`` scale heights evaluated at ``rmax_AU``
    keeps the fraction ``erf(k * h(rmax) / h(r))`` of a Gaussian layer; with
    linear flaring (``beta = 1``) that is ``erf(k * rmax / r)``. A window
    sized at the reference radius would keep ``erf(k * sma / r)`` instead
    (0.64 rather than 0.88 at 4.6 AU), so this pins which radius sets the
    support. Tolerance basis: spike (measured departures of 5e-4, from the
    1/d^2 and phase factors across the window and the 161-node error).
    """
    from math import erf

    cut = np.asarray(_render(_belt(161, 1.0), wavelength_nm=550.0, incl_deg=0.0))
    full = np.asarray(_render(_belt(161, 12.0), wavelength_nm=550.0, incl_deg=0.0))
    c = full.shape[0] // 2
    for col in (c + 40, c + 47, c + 54):
        r_AU = (col - c) * _BELT_PX_AU
        kept = cut[c, col] / full[c, col]
        assert kept == pytest.approx(erf(5.0 / r_AU), abs=5e-3), f"r={r_AU:.2f} AU"


def test_wavelength_out_of_range_returns_nan():
    """Querying outside the wavelength grid returns NaN, not silent extrapolation."""
    d = make_disk(
        wavelengths_nm=jnp.array([500.0, 700.0]),
        g_HG_grid=jnp.array([0.3, 0.3]),
        Ag_grid=jnp.array([0.5, 0.5]),
    )
    sb = _render(d, wavelength_nm=1500.0)
    assert bool(jnp.all(jnp.isnan(sb)))


def test_wavelength_vmap_returns_cube():
    """Vmap over wavelength expands the output to (n_wave, ny, nx)."""
    d = make_disk(
        wavelengths_nm=jnp.array([500.0, 700.0, 900.0]),
        g_HG_grid=jnp.array([0.5, 0.3, 0.1]),
        Ag_grid=jnp.array([0.4, 0.5, 0.6]),
    )
    wls = jnp.array([550.0, 650.0, 850.0])
    cube = jax.vmap(d.surface_brightness, in_axes=(0, None, None, None))(
        wls, jnp.array(0.0), jnp.array(60.0), jnp.array(0.0)
    )
    assert cube.shape == (3, d.ny, d.nx)
    assert bool(jnp.all(jnp.isfinite(cube)))
    assert not jnp.allclose(cube[0], cube[1])
