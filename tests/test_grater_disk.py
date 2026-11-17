"""skyscapes.disk.GraterDisk -- Augereau 1999 scattered-light disk."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from zodi import DensityDust

from skyscapes.disk import AbstractDisk, GraterDisk
from tests.conftest import radiance, sky_grid

_GRID = sky_grid(51, 51, 0.2)


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
        amplitude=DensityDust(
            level=1.0, profile_id="grater-1999", calibration_id="unit-profile"
        ),
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
        n_slices_los=31,
    )
    defaults.update(overrides)
    return GraterDisk(**defaults)


def _render(
    d,
    wavelength_nm=500.0,
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


def test_grater_disk_is_abstract():
    """GraterDisk satisfies the AbstractDisk interface."""
    assert isinstance(make_disk(), AbstractDisk)


def test_shape_and_finiteness():
    """photon_radiance returns the expected shape with finite, non-neg values."""
    sb = _render(make_disk())
    assert sb.shape == (51, 51)
    assert bool(jnp.all(jnp.isfinite(sb)))
    assert bool(jnp.all(sb >= 0.0))
    assert float(sb.sum()) > 0.0


def test_pole_on_is_axisymmetric():
    """incl=0 + pa=0 -> image equals its 90 deg rotation (within tol)."""
    sb = _render(make_disk(), position_angle_deg=90.0 - (0.0), inclination_deg=0.0)
    sb_rot = jnp.rot90(sb)
    peak = float(sb.max())
    rel_err = float(jnp.max(jnp.abs(sb - sb_rot)) / peak)
    assert rel_err < 1e-4, f"pole-on disk not axisymmetric: rel_err={rel_err}"


def test_hg_sign_flip_swaps_asymmetry():
    """Flipping g_HG flips the forward/back asymmetry of an inclined disk."""
    sb_fwd = _render(
        make_disk(g_HG=0.3), position_angle_deg=90.0 - (0.0), inclination_deg=60.0
    )
    sb_bwd = _render(
        make_disk(g_HG=-0.3), position_angle_deg=90.0 - (0.0), inclination_deg=60.0
    )
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


def test_retrograde_orientation_is_the_same_midplane():
    """``(180 - incl, pa + 180)`` is the same midplane as ``(incl, pa)``.

    Turning the line of nodes by 180 degrees and supplementing the inclination
    flips only the side of the disk the observer calls its normal; the dust
    distribution is symmetric about the midplane, so the image is unchanged,
    pixel for pixel. Tolerance basis: floating-point.
    """
    d = make_disk(g_HG=0.4)
    ref = np.asarray(_render(d, position_angle_deg=90.0 - (30.0), inclination_deg=55.0))
    same = np.asarray(
        _render(d, position_angle_deg=90.0 - (210.0), inclination_deg=125.0)
    )
    assert np.all(same >= 0.0)
    np.testing.assert_allclose(same, ref, rtol=1e-9, atol=1e-12 * ref.max())


def test_jit_round_trip():
    """JIT'd photon_radiance matches the eager output."""
    d = make_disk()

    @jax.jit
    def f(disk):
        return radiance(
            disk,
            jnp.array(500.0),
            jnp.array(0.0),
            jnp.array(60.0),
            90.0 - (jnp.array(0.0)),
            dist_pc=10.0,
            grid=_GRID,
        )

    sb_jit = f(d)
    sb_eager = _render(d, inclination_deg=60.0)
    # jit and eager fuse the LOS quadrature reduction differently, so the
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

    The LOS quadrature covers the flared layer around the midplane crossing
    (matching the GRaTeR-JAX reference), so the model stays accurate well past
    the old arctan(rmax/zmax) guard (~83 deg for this disk). 88 and 92 deg
    (|cos| = 0.035) are highly inclined on either side of 90 but far from the
    cos_i -> 0 singularity.
    """
    sb = _render(make_disk(), inclination_deg=incl)
    assert sb.shape == (51, 51)
    assert bool(jnp.all(jnp.isfinite(sb)))
    assert bool(jnp.all(sb >= 0.0))
    assert float(sb.sum()) > 0.0


@pytest.mark.parametrize(
    "incl, cutoff_width", [(89.5, 0.0), (90.0, 0.0), (89.5, 0.2), (90.0, 0.2)]
)
def test_edge_on_render_gradient_and_convergence(incl, cutoff_width):
    """Fails if the edge-on window diverges or loses gradients and convergence.

    Smooth radial cutoffs remove node-crossing jumps. Tolerance basis:
    the 0.5 percent image-accuracy bound against a 513-node reference,
    normalized by the peak. Gauss-Legendre errors need not be monotonic
    with node count. Inclination derivatives use central differences.
    """
    grid = sky_grid(17, 9, 0.04)
    disk = replace(
        _belt(257),
        cutoff_width_AU=jnp.asarray(cutoff_width),
    )
    render = eqx.filter_jit(
        lambda d, i: radiance(d, 550.0, 0.0, i, 90.0 - (13.0), dist_pc=10.0, grid=grid)
    )
    images = [
        render(replace(disk, n_slices_los=n), jnp.asarray(incl))
        for n in (129, 257, 513)
    ]
    peak = images[-1].max()
    residuals = [
        float(jnp.max(jnp.abs(image - images[-1])) / peak) for image in images[:-1]
    ]
    print(f"LOS ladder {incl}, cutoff {cutoff_width}: {residuals}")
    assert max(residuals) < 0.005
    point = jnp.asarray(incl)
    derivative = jax.jacfwd(lambda i: render(disk, i))(point)
    step = 1e-4
    fd = (render(disk, point + step) - render(disk, point - step)) / (2 * step)
    error = float(jnp.max(jnp.abs(derivative - fd)) / peak)
    print(f"Inclination gradient error {incl}, cutoff {cutoff_width}: {error:.3g}")
    np.testing.assert_allclose(derivative / peak, fd / peak, rtol=2e-5, atol=1e-8)
    assert jnp.all(jnp.isfinite(derivative))
    assert jnp.max(jnp.abs(derivative)) > 1e-4 * peak


def test_edge_on_continuity_and_smooth_cutoff_gradients():
    """Fails if a rule has a seam near edge-on or taper parameters detach.

    Includes inclinations on both sides of edge-on. The hard-cutoff default
    retains sample-crossing jumps; the
    positive-width fit uses a C2 taper so both the geometry and density are
    continuous. The Gauss-Legendre window has no threshold in inclination and
    must pass the same probes.
    """
    grid = sky_grid(17, 9, 0.04)
    disk = replace(
        _belt(257),
        cutoff_width_AU=jnp.asarray(0.2),
    )
    render = eqx.filter_jit(
        lambda d, i: radiance(d, 550.0, 0.0, i, 90.0 - (0.0), dist_pc=10.0, grid=grid)
    )
    angles = jnp.array([89.3, 89.43, 89.5, 90.0, 90.5])
    images = jax.vmap(lambda i: render(disk, i))(angles)
    assert jnp.all(jnp.isfinite(images))
    np.testing.assert_allclose(images[-1], images[2, ::-1], atol=1e-12, rtol=1e-10)
    seam = jnp.rad2deg(jnp.arccos(0.01))
    probes = jnp.concatenate([angles, jnp.array([seam])])
    step = 1e-5
    left = jax.vmap(lambda i: render(disk, i - step))(probes)
    right = jax.vmap(lambda i: render(disk, i + step))(probes)
    change = float(jnp.max(jnp.abs(left - right)) / images.max())
    print(f"Continuity probe peak-normalized change: {change:.3g}")
    assert change < 1e-3

    def loss(p):
        updated = eqx_replace(
            disk, cutoff_width_AU=p[0] * 0.2, rmin_AU=p[1] * 1.5, rmax_AU=p[2] * 5.0
        )
        return render(updated, jnp.asarray(89.5)).sum() / images[2].sum()

    p = jnp.ones(3)
    directions = jnp.eye(3) * 1e-5
    fd = jax.vmap(lambda d: (loss(p + d) - loss(p - d)) / 2e-5)(directions)
    np.testing.assert_allclose(jax.grad(loss)(p), fd, rtol=1e-5, atol=1e-7)


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
    cube = jax.vmap(
        partial(radiance, d, grid=_GRID, dist_pc=10.0), in_axes=(0, None, None, None)
    )(wls, jnp.array(0.0), jnp.array(60.0), jnp.array(0.0))
    assert cube.shape == (3, _GRID.shape[0], _GRID.shape[1])
    assert bool(jnp.all(jnp.isfinite(cube)))
    assert not jnp.allclose(cube[0], cube[1])


def test_ring_major_axis_has_fits_sky_pa():
    """North-through-east PA is measured with east=-column, north=+row."""
    grid = sky_grid(101, 101, 0.1)
    disk = make_disk(
        g_HG=0.0,
        sma_AU=jnp.array(20.0),
        alpha_in=jnp.array(30.0),
        alpha_out=jnp.array(-30.0),
        ksi0_AU=jnp.array(0.4),
        rmin_AU=jnp.array(10.0),
        rmax_AU=jnp.array(30.0),
    )
    sky_pa = 60.0
    image = np.asarray(
        _render(
            disk,
            position_angle_deg=90.0 - (90.0 - sky_pa),
            inclination_deg=60.0,
            grid=grid,
        )
    )
    row, col = np.indices(image.shape, dtype=float)
    east, north = -(col - 50), row - 50
    ee = np.sum(image * east**2)
    nn = np.sum(image * north**2)
    en = np.sum(image * east * north)
    measured = np.degrees(0.5 * np.arctan2(2 * en, nn - ee)) % 180
    difference = abs((measured - sky_pa + 90) % 180 - 90)
    assert difference < 0.5, f"sky PA {measured:.3f} deg, expected {sky_pa} deg"


def _belt(n_slices_los: int) -> GraterDisk:
    """Narrow belt for edge-on convergence and gradient checks."""
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
        n_slices_los=n_slices_los,
        amplitude=DensityDust(
            level=1.0, profile_id="grater-1999", calibration_id="unit-profile"
        ),
    )
