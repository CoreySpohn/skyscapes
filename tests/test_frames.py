"""Unit tests for skyscapes.io._frames -- rotation to sky coordinates."""

from __future__ import annotations

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from exoverses.util.misc import gen_rotate_to_sky_coords as exoverses_rotate

from skyscapes.io._frames import rotate_to_sky_coords


def test_zero_inc_zero_pa_is_z_flip_only():
    """With inc=pa=0, the only effect is flipping the z component."""
    v = jnp.array([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]])
    out = rotate_to_sky_coords(v, inc_deg=0.0, pa_deg=0.0)
    np.testing.assert_allclose(out, jnp.array([[1.0, 2.0, -3.0], [4.0, -1.0, -0.5]]))


def test_matches_exoverses_general_case():
    """Bit-for-bit match with exoverses' reference implementation."""
    rng = np.random.default_rng(0)
    v_np = rng.normal(size=(7, 3))
    inc_deg = 30.0
    pa_deg = 47.5

    expected = exoverses_rotate(
        v_np.copy(), inc_deg * u.deg, pa_deg * u.deg, convention="exovista"
    )
    with jax.enable_x64():
        got = rotate_to_sky_coords(jnp.asarray(v_np), inc_deg=inc_deg, pa_deg=pa_deg)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=0)


def test_matches_exoverses_negative_pa():
    """Exoverses' star.py applies a negative PA convention; verify our port matches."""
    rng = np.random.default_rng(1)
    v_np = rng.normal(size=(5, 3))
    inc_deg = -12.3
    pa_deg = 88.0

    expected = exoverses_rotate(
        v_np.copy(), inc_deg * u.deg, pa_deg * u.deg, convention="exovista"
    )
    with jax.enable_x64():
        got = rotate_to_sky_coords(jnp.asarray(v_np), inc_deg=inc_deg, pa_deg=pa_deg)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=0)
