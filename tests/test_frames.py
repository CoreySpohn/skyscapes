"""Unit tests for skyscapes.io._frames -- rotation to sky coordinates."""

from __future__ import annotations

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from exoverses.util.misc import gen_rotate_to_sky_coords as exoverses_rotate

from skyscapes.io._frames import rotate_to_sky_coords


def test_zero_inc_zero_pa_is_identity():
    """With inc=pa=0, the proper rotation is the identity (z is away)."""
    v = jnp.array([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]])
    out = rotate_to_sky_coords(v, inc_deg=0.0, pa_deg=0.0)
    np.testing.assert_allclose(out, v)


def test_matches_exoverses_general_case():
    """Exoverses ends in toward-z; negating it gives our away-z frame."""
    rng = np.random.default_rng(0)
    v_np = rng.normal(size=(7, 3))
    inc_deg = 30.0
    pa_deg = 47.5

    expected = exoverses_rotate(
        v_np.copy(), inc_deg * u.deg, pa_deg * u.deg, convention="exovista"
    )
    expected[:, 2] *= -1  # Change chart from toward to away.
    with jax.enable_x64():
        got = rotate_to_sky_coords(jnp.asarray(v_np), inc_deg=inc_deg, pa_deg=pa_deg)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=0)


def test_matches_exoverses_negative_pa():
    """The same away-z conversion applies at negative inclination."""
    rng = np.random.default_rng(1)
    v_np = rng.normal(size=(5, 3))
    inc_deg = -12.3
    pa_deg = 88.0

    expected = exoverses_rotate(
        v_np.copy(), inc_deg * u.deg, pa_deg * u.deg, convention="exovista"
    )
    expected[:, 2] *= -1  # Change chart from toward to away.
    with jax.enable_x64():
        got = rotate_to_sky_coords(jnp.asarray(v_np), inc_deg=inc_deg, pa_deg=pa_deg)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize(
    ("inc", "pa"), [(0.0, 0.0), (35.0, 20.0), (60.0, 130.0), (125.0, -40.0)]
)
def test_sky_rotation_is_proper(inc, pa):
    """A right-handed frame preserves lengths, angles, and orientation."""
    rotation = np.asarray(rotate_to_sky_coords(jnp.eye(3), inc, pa)).T
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize(("inc", "pa"), [(35.0, 20.0), (60.0, 130.0), (125.0, -40.0)])
def test_disk_axes_preserve_physical_plane_and_near_half(inc, pa):
    """Independent Rz(pa) Rx(-inc) basis, with the near half at negative z."""
    from skyscapes.viz._geometry import disk_axes_sky

    i, p = np.radians([inc, pa])
    nodes = np.array([np.cos(p), np.sin(p), 0.0])
    near = np.array([-np.sin(p) * np.cos(i), np.cos(p) * np.cos(i), -np.sin(i)])
    normal = np.array([-np.sin(p) * np.sin(i), np.cos(p) * np.sin(i), np.cos(i)])
    np.testing.assert_allclose(
        disk_axes_sky(inc, pa), [nodes, near, normal], atol=1e-14
    )
    assert near[2] < 0
