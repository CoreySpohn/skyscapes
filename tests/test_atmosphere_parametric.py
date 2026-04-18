"""ParametricAtmosphere stub -- raises NotImplementedError until RT lands."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from skyscapes.atmosphere import AbstractAtmosphere, ParametricAtmosphere


def test_parametric_is_abstract_atmosphere():
    """ParametricAtmosphere satisfies the AbstractAtmosphere interface."""
    atm = ParametricAtmosphere(Rp_Rearth=jnp.array([1.0]))
    assert isinstance(atm, AbstractAtmosphere)


def test_parametric_raises_on_call():
    """Stub raises NotImplementedError until the RT adapter lands."""
    atm = ParametricAtmosphere(Rp_Rearth=jnp.array([1.0]))
    with pytest.raises(NotImplementedError):
        atm.reflected_spectrum(
            phase_angle_rad=jnp.array([[0.0]]),
            dist_AU=jnp.array([[1.0]]),
            wavelength_nm=jnp.array([500.0]),
        )
