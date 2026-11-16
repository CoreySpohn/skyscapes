"""skyscapes.disk.CompositeDisk -- sum-of-components."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.grids import PixelGrid, SkyOrientation
from zodi import DensityDust

from skyscapes.disk import AbstractDisk, CompositeDisk, ExovistaDisk, GraterDisk
from tests.conftest import radiance, sky_grid

_GRID = sky_grid(51, 51, 0.2)

_WL = jnp.array(500.0)
_T = jnp.array(0.0)
_INCL = jnp.array(60.0)
_PA = jnp.array(0.0)


def _make_grater(sma_AU: float, ksi0_AU: float = 1.0) -> GraterDisk:
    return GraterDisk(
        sma_AU=jnp.array(sma_AU),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-5.0),
        ksi0_AU=jnp.array(ksi0_AU),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(0.5 * sma_AU),
        rmax_AU=jnp.array(2.5 * sma_AU),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([0.3, 0.3]),
        Ag_grid=jnp.array([0.5, 0.5]),
        n_slices_los=31,
        amplitude=DensityDust(
            level=1.0, profile_id="grater-1999", calibration_id="unit-profile"
        ),
    )


def test_composite_is_abstract():
    """CompositeDisk satisfies the AbstractDisk interface."""
    c = CompositeDisk(components=(_make_grater(20.0),))
    assert isinstance(c, AbstractDisk)


def test_composite_sums_components():
    """Two-component composite equals the sum of the individual renderings."""
    d1 = _make_grater(20.0)
    d2 = _make_grater(40.0)
    composite = CompositeDisk(components=(d1, d2))
    sb_sum = radiance(composite, _WL, _T, _INCL, 90.0 - (_PA), dist_pc=10.0, grid=_GRID)
    sb_indiv = radiance(
        d1, _WL, _T, _INCL, 90.0 - (_PA), dist_pc=10.0, grid=_GRID
    ) + radiance(d2, _WL, _T, _INCL, 90.0 - (_PA), dist_pc=10.0, grid=_GRID)
    assert sb_sum.shape == (51, 51)
    assert jnp.allclose(sb_sum, sb_indiv, rtol=1e-5, atol=1e-12)


def test_composite_retrograde_render_is_nonnegative_and_mirrored():
    """Above 90 degrees the summed map is the physical, mirrored one.

    Each component at ``180 - incl`` is its ``incl`` map reflected across the
    line of nodes (row reversal at ``pa = 0``), so the sum is too; a component
    returning a negated map would make the sum negative. Tolerance basis:
    floating-point.
    """
    composite = CompositeDisk(components=(_make_grater(20.0), _make_grater(40.0)))
    low = np.asarray(
        radiance(
            composite, _WL, _T, jnp.array(60.0), 90.0 - (_PA), dist_pc=10.0, grid=_GRID
        )
    )
    high = np.asarray(
        radiance(
            composite, _WL, _T, jnp.array(120.0), 90.0 - (_PA), dist_pc=10.0, grid=_GRID
        )
    )
    assert np.all(high >= 0.0), f"negative radiance, min {high.min():.3g}"
    np.testing.assert_allclose(high, low[::-1], rtol=1e-9, atol=1e-12 * low.max())


def test_composite_jit_round_trip():
    """JIT'd composite matches the eager output."""
    composite = CompositeDisk(components=(_make_grater(20.0), _make_grater(40.0)))

    @jax.jit
    def f(c):
        return radiance(c, _WL, _T, _INCL, 90.0 - (_PA), dist_pc=10.0, grid=_GRID)

    sb_jit = f(composite)
    sb_eager = radiance(
        composite, _WL, _T, _INCL, 90.0 - (_PA), dist_pc=10.0, grid=_GRID
    )
    # jit and eager fuse the log-spaced LOS trapezoid reduction differently, so the
    # near-zero outer-disk pixels agree only to the field scale, not to 1e-12.
    atol = 1e-6 * float(jnp.max(sb_eager))
    assert jnp.allclose(sb_jit, sb_eager, rtol=1e-5, atol=atol)


def test_composite_rejects_empty():
    """Construction with no components raises."""
    with pytest.raises(ValueError, match="at least one component"):
        CompositeDisk(components=())


def test_composite_resamples_mismatched_native_grids():
    """Fails if native pixel contrasts are summed before density conversion.

    Constant densities on overlapping grids must sum exactly (roundoff).
    """
    from hwoutils.constants import arcsec2rad

    from skyscapes.scene import FlatStar

    grid = sky_grid(3, 3, 0.01)
    coarse = sky_grid(5, 5, 0.02)
    a = ExovistaDisk(
        grid, jnp.array([400.0, 700.0]), jnp.ones((2, 3, 3)) * grid.pixel_area
    )
    b = ExovistaDisk(
        coarse, jnp.array([400.0, 700.0]), jnp.ones((2, 5, 5)) * coarse.pixel_area
    )
    image = CompositeDisk((a, b)).photon_radiance(
        550.0, 0.0, star=FlatStar(1.0), dist_pc=10.0, grid=grid
    )
    np.testing.assert_allclose(image.data * arcsec2rad**2, 2.0, rtol=2e-15)


def test_composite_heterogeneous_components():
    """A GraterDisk + ExovistaDisk pair composes if extents match."""
    grater = _make_grater(20.0)
    cube = jnp.zeros((2, _GRID.shape[0], _GRID.shape[1])) + 1e-9
    exo = ExovistaDisk(
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        contrast_cube=cube,
        native_grid=PixelGrid.regular(
            (cube).shape[-2:],
            _GRID.step,
            "arcsec",
            orientation=SkyOrientation(0.0),
        ),
    )
    composite = CompositeDisk(components=(grater, exo))
    sb = radiance(composite, _WL, _T, _INCL, 90.0 - (_PA), dist_pc=10.0, grid=_GRID)
    assert sb.shape == (_GRID.shape[0], _GRID.shape[1])
    assert bool(jnp.all(jnp.isfinite(sb)))
