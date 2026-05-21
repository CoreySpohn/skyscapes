"""skyscapes.disk.CompositeDisk -- sum-of-components."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from skyscapes.disk import AbstractDisk, CompositeDisk, ExovistaDisk, GraterDisk

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
        nx=51,
        ny=51,
        pixel_scale_arcsec=0.2,
        dist_pc=10.0,
        n_slices_los=31,
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
    sb_sum = composite.surface_brightness(_WL, _T, _INCL, _PA)
    sb_indiv = d1.surface_brightness(_WL, _T, _INCL, _PA) + d2.surface_brightness(
        _WL, _T, _INCL, _PA
    )
    assert sb_sum.shape == (51, 51)
    assert jnp.allclose(sb_sum, sb_indiv, rtol=1e-5, atol=1e-12)


def test_composite_jit_round_trip():
    """JIT'd composite matches the eager output."""
    composite = CompositeDisk(components=(_make_grater(20.0), _make_grater(40.0)))

    @jax.jit
    def f(c):
        return c.surface_brightness(_WL, _T, _INCL, _PA)

    sb_jit = f(composite)
    sb_eager = composite.surface_brightness(_WL, _T, _INCL, _PA)
    assert jnp.allclose(sb_jit, sb_eager, rtol=1e-5, atol=1e-12)


def test_composite_rejects_empty():
    """Construction with no components raises."""
    with pytest.raises(ValueError, match="at least one component"):
        CompositeDisk(components=())


def test_composite_rejects_mismatched_extents():
    """Components with different spatial extents are rejected."""
    d1 = _make_grater(20.0)
    d2 = GraterDisk(
        sma_AU=jnp.array(20.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-5.0),
        ksi0_AU=jnp.array(1.0),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(10.0),
        rmax_AU=jnp.array(50.0),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([0.3, 0.3]),
        Ag_grid=jnp.array([0.5, 0.5]),
        nx=51,
        ny=51,
        pixel_scale_arcsec=0.4,  # different from d1
        dist_pc=10.0,
        n_slices_los=31,
    )
    with pytest.raises(ValueError, match="spatial_extent"):
        CompositeDisk(components=(d1, d2))


def test_composite_heterogeneous_components():
    """A GraterDisk + ExovistaDisk pair composes if extents match."""
    grater = _make_grater(20.0)
    cube = jnp.zeros((2, grater.ny, grater.nx)) + 1e-9
    exo = ExovistaDisk(
        pixel_scale_arcsec=grater.pixel_scale_arcsec,
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        contrast_cube=cube,
    )
    composite = CompositeDisk(components=(grater, exo))
    sb = composite.surface_brightness(_WL, _T, _INCL, _PA)
    assert sb.shape == (grater.ny, grater.nx)
    assert bool(jnp.all(jnp.isfinite(sb)))
