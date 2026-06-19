"""skyscapes.disk.GraterDisk -- Augereau 1999 scattered-light disk."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
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
    assert jnp.allclose(sb_jit, sb_eager, rtol=1e-5, atol=1e-12)


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


def test_high_inclination_is_finite():
    """A highly inclined (but not edge-on) disk renders finite and positive.

    The LOS quadrature is log-spaced and concentrated at the midplane crossing
    (matching the GRaTeR-JAX reference), so the model stays accurate well past
    the old arctan(rmax/zmax) guard (~83 deg for this disk). incl=88 (cos=0.035)
    is highly inclined but far from the true cos_i -> 0 singularity.
    """
    sb = _render(make_disk(), incl_deg=88.0)
    assert sb.shape == (51, 51)
    assert bool(jnp.all(jnp.isfinite(sb)))
    assert bool(jnp.all(sb >= 0.0))
    assert float(sb.sum()) > 0.0


def test_los_quadrature_converges_at_high_inclination():
    """Log-spaced LOS quadrature is converged: 31 slices ~= 121 slices."""
    coarse = _render(make_disk(n_slices_los=31), incl_deg=80.0)
    fine = _render(make_disk(n_slices_los=121), incl_deg=80.0)
    rel = float(jnp.abs(coarse.sum() - fine.sum()) / fine.sum())
    assert rel < 0.03, f"LOS quadrature not converged at incl=80: rel={rel}"


def test_edge_on_render_is_rejected():
    """Rendering at the true cos_i -> 0 singularity raises at render time."""
    d = make_disk()
    with pytest.raises(Exception, match="edge-on"):
        _render(d, incl_deg=89.9)


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
