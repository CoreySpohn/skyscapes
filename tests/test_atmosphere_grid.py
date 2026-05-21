"""GridAtmosphere -- 2D interpolation over (wavelength, phase-angle)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from skyscapes.atmosphere import (
    AbstractAtmosphere,
    GridAtmosphere,
)


def _flat_grid():
    """Constant-contrast grid: 0.1 everywhere."""
    wl = jnp.linspace(400.0, 1000.0, 10)
    phase_deg = jnp.linspace(0.0, 180.0, 19)
    contrast = jnp.full((wl.size, phase_deg.size), 0.1)
    return wl, phase_deg, contrast


def test_grid_is_abstract_atmosphere():
    """GridAtmosphere subclasses AbstractAtmosphere."""
    wl, phase_deg, contrast = _flat_grid()
    atm = GridAtmosphere(
        Rp_Rearth=jnp.array([1.0]),
        wavelengths_nm=wl,
        phase_angle_deg=phase_deg,
        contrast_grid=contrast[None, ...],  # (K=1, n_wl, n_phase)
    )
    assert isinstance(atm, AbstractAtmosphere)


def test_grid_returns_grid_value_on_nodes():
    """Querying inside a constant-0.1 grid returns 0.1 to interp precision."""
    wl, phase_deg, contrast = _flat_grid()
    atm = GridAtmosphere(
        Rp_Rearth=jnp.array([1.0]),
        wavelengths_nm=wl,
        phase_angle_deg=phase_deg,
        contrast_grid=contrast[None, ...],
    )
    # Interior phase in radians: 90 deg = pi/2
    result = atm.reflected_spectrum(
        phase_angle_rad=jnp.array([[jnp.pi / 2]]),
        dist_AU=jnp.array([[1.0]]),  # ignored by GridAtmosphere
        wavelength_nm=jnp.array([600.0]),
    )
    assert jnp.allclose(result, 0.1, rtol=1e-6)


def test_grid_multiple_planets_independent():
    """Two planets with different grids -- interp is per-planet."""
    wl = jnp.linspace(400.0, 1000.0, 10)
    phase_deg = jnp.linspace(0.0, 180.0, 19)
    # Planet 0: constant 0.1; Planet 1: constant 0.2
    grids = jnp.stack(
        [
            jnp.full((wl.size, phase_deg.size), 0.1),
            jnp.full((wl.size, phase_deg.size), 0.2),
        ],
        axis=0,
    )
    atm = GridAtmosphere(
        Rp_Rearth=jnp.array([1.0, 2.0]),
        wavelengths_nm=wl,
        phase_angle_deg=phase_deg,
        contrast_grid=grids,
    )
    result = atm.reflected_spectrum(
        phase_angle_rad=jnp.array([[jnp.pi / 3], [jnp.pi / 3]]),
        dist_AU=jnp.array([[1.0], [1.0]]),
        wavelength_nm=jnp.array([600.0]),
    )
    assert jnp.allclose(result[0], 0.1, rtol=1e-6)
    assert jnp.allclose(result[1], 0.2, rtol=1e-6)


def test_grid_handles_x64_query_against_float32_cube():
    """Regression: GridAtmosphere tolerates float64 query under x64.

    The loaded ExoVista contrast cube is float32. Under
    ``with jax.enable_x64():`` the orbit propagator returns float64
    phase angles, and ``interpax.interp2d`` previously raised
    ``TypeError: switch branches must have equal output types`` on the
    mismatch. ``reflected_spectrum`` now promotes all interp inputs to
    a common dtype before calling interpax.
    """
    wl, phase_deg, contrast = _flat_grid()
    # Force the cube and grids to float32 (matches what the ExoVista
    # loader produces on disk).
    atm = GridAtmosphere(
        Rp_Rearth=jnp.asarray([1.0], dtype=jnp.float32),
        wavelengths_nm=wl.astype(jnp.float32),
        phase_angle_deg=phase_deg.astype(jnp.float32),
        contrast_grid=contrast[None, ...].astype(jnp.float32),
    )

    with jax.enable_x64():
        result = atm.reflected_spectrum(
            phase_angle_rad=jnp.array([[jnp.pi / 2]], dtype=jnp.float64),
            dist_AU=jnp.array([[1.0]], dtype=jnp.float64),
            wavelength_nm=jnp.array(600.0, dtype=jnp.float64),
        )
        # Result follows the promoted dtype (float64).
        assert result.dtype == jnp.float64
        assert jnp.allclose(result, 0.1, rtol=1e-5)
