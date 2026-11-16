"""Absolute radiance and pointing contracts for every local-zodi implementation."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import zodi
from hwoutils.constants import arcsec2rad
from hwoutils.conversions import AB_ZERO_POINT_JY

from skyscapes.background import (
    AbstractLocalZodi,
    AYOZodi,
    LeinertZodi,
    PrecomputedZodi,
)

POINTING = dict(ecliptic_latitude_deg=30.0, solar_longitude_deg=120.0)


@pytest.mark.parametrize("kind", AbstractLocalZodi.__subclasses__())
def test_radiance_contract(kind):
    """Fails on a per-arcsec^2 or per-micron output or dropped wavelength axes.

    Absolute expectations use SI h, the exact AB zero point and the solid
    angle of one arcsec squared, typed independently. 3e-14 relative allows
    float64 algebra and log interpolation roundoff; atol=0 protects scale.
    Constant parity uses 2e-15 relative for float64 rounding.
    """
    assert AB_ZERO_POINT_JY == pytest.approx(3630.7805477010028, rel=2e-15)
    np.testing.assert_allclose(arcsec2rad**2, 2.3504430539097885e-11, rtol=2e-15)
    wave = jnp.array([[500.0, 550.0], [600.0, 700.0]])
    fixed_v = (
        3630.7805477010028
        * 10 ** (-0.4 * 22)
        * 1e-26
        / (6.62607015e-34 * 550e-9)
        * 1e-9
        / 2.3504430539097885e-11
    )
    if kind is AYOZodi:
        model = kind()
        expected = fixed_v * zodi.color_correction(wave, 550.0, photon_units=True)
        np.testing.assert_allclose(
            model.photon_radiance(jnp.array(550.0), **POINTING),
            fixed_v,
            rtol=3e-14,
            atol=0,
        )
    elif kind is LeinertZodi:
        model = kind()
        power = zodi.specific_intensity(120.0, 30.0, np.asarray(wave), anchor="table19")
        expected = (
            power / (6.62607015e-34 * 299792458.0 / (np.asarray(wave) * 1e-9)) * 1e-3
        )
    else:
        assert kind is PrecomputedZodi
        model = kind(jnp.array([500.0, 550.0, 600.0, 700.0]), jnp.full(4, fixed_v))
        expected = np.full(wave.shape, fixed_v)
    actual = model.photon_radiance(wave, **POINTING)
    assert actual.shape == wave.shape
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    if kind is not LeinertZodi:
        np.testing.assert_array_equal(
            actual,
            model.photon_radiance(
                wave, ecliptic_latitude_deg=75.0, solar_longitude_deg=90.0
            ),
        )


def test_pointing_order_and_radiance_units():
    """Fails on latitude/longitude swaps, wrong sr scale, or wrong color.

    The deliberately asymmetric (120,30) position differs from the swapped
    lookup. AYO and Leinert share Table 19 color, but their default V scales
    only approximately coincide. Setting AYO to the table's AB magnitude at
    (90,0) makes them coincide exactly; 3e-14 bounds float64 log roundoff.
    """
    wave = jnp.array([500.0, 550.0, 700.0])
    model = LeinertZodi()
    expected = (
        zodi.specific_intensity(120.0, 30.0, wave, photon_units=True, anchor="table19")
        / 1000
    )
    swapped = (
        zodi.specific_intensity(30.0, 120.0, wave, photon_units=True, anchor="table19")
        / 1000
    )
    np.testing.assert_allclose(
        model.photon_radiance(wave, **POINTING), expected, rtol=3e-14, atol=0
    )
    assert np.all(np.abs(np.asarray(expected / swapped) - 1) > 0.1)
    pointing = dict(ecliptic_latitude_deg=0.0, solar_longitude_deg=90.0)
    mag = zodi.surface_brightness_ab_mag(
        jnp.array(90.0), jnp.array(0.0), jnp.array(550.0), anchor="table19"
    )
    fixed = AYOZodi(surface_brightness_mag_arcsec2=mag)
    radiance = model.photon_radiance(wave, **pointing)
    np.testing.assert_allclose(
        fixed.photon_radiance(wave, **pointing), radiance, rtol=3e-14, atol=0
    )
    cached = PrecomputedZodi(wave, radiance)
    np.testing.assert_allclose(
        cached.photon_radiance(wave, **pointing), radiance, rtol=3e-14, atol=0
    )


def test_radiance_jax():
    """Fails if JIT or wavelength vmap changes zodi; 3e-14 is float64 roundoff."""
    model = LeinertZodi()
    wave = jnp.array([500.0, 550.0, 700.0])

    def evaluate(w):
        return model.photon_radiance(w, **POINTING)

    loop = jnp.stack([evaluate(w) for w in wave])
    np.testing.assert_allclose(eqx.filter_jit(evaluate)(wave), loop, rtol=3e-14, atol=0)
    np.testing.assert_allclose(jax.vmap(evaluate)(wave), loop, rtol=3e-14, atol=0)
