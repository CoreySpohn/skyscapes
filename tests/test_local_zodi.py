"""Absolute radiance and sky-frame contracts for every local-zodi model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import zodi
from hwoutils import constants as const
from orbix.observatory import ecliptic_pointing
from orbix.system import SkyFrame
from zodi.tables import TABLE19_WAVELENGTH_UM

from skyscapes import AbstractLocalZodi, AYOZodi, LeinertZodi, PrecomputedZodi

FRAME = SkyFrame(10.0, ra_deg=100.0, dec_deg=-20.0)
UNPLACED = SkyFrame(10.0)
TIME_D = 40.0


def _pointing():
    return ecliptic_pointing(FRAME.ra_deg, FRAME.dec_deg, FRAME.epoch, TIME_D)


@pytest.mark.parametrize("kind", AbstractLocalZodi.__subclasses__())
def test_radiance_contract(kind):
    """Fails on a per-arcsec^2 or per-micron output or dropped wavelength axes.

    Absolute expectations use SI h, the exact AB zero point and the solid
    angle of one arcsec squared, computed from radians. 3e-14 relative allows
    float64 algebra and log interpolation roundoff; atol=0 protects scale.
    """
    arcsec_sr = (np.pi / (180 * 3600)) ** 2
    wave = jnp.array([[500.0, 550.0], [600.0, 700.0]])
    fixed_v = (
        const.AB_ZERO_POINT_JY
        * 10 ** (-0.4 * 22)
        * 1e-26
        / (const.h * 550e-9)
        * 1e-9
        / arcsec_sr
    )
    if kind is AYOZodi:
        model = kind()
        expected = fixed_v * zodi.color_correction(wave, 550.0, photon_units=True)
        np.testing.assert_allclose(
            model.photon_radiance(jnp.array(550.0), FRAME, TIME_D),
            fixed_v,
            rtol=3e-14,
            atol=0,
        )
    elif kind is LeinertZodi:
        model = kind()
        latitude_deg, longitude_deg = _pointing()
        power = zodi.specific_intensity(
            longitude_deg, latitude_deg, np.asarray(wave), anchor="table19"
        )
        expected = power / (const.h * const.c / (np.asarray(wave) * 1e-9)) * 1e-3
    else:
        assert kind is PrecomputedZodi
        model = kind(jnp.array([500.0, 550.0, 600.0, 700.0]), jnp.full(4, fixed_v))
        expected = np.full(wave.shape, fixed_v)
    actual = model.photon_radiance(wave, FRAME, TIME_D)
    breaks = model.spectral_breaks_nm()
    assert isinstance(breaks, np.ndarray)
    expected_breaks = (
        model.wavelengths_nm
        if kind is PrecomputedZodi
        else TABLE19_WAVELENGTH_UM * 1000
    )
    np.testing.assert_array_equal(breaks, expected_breaks)
    assert actual.shape == wave.shape
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    if kind is not LeinertZodi:
        other = SkyFrame(10.0, ra_deg=10.0, dec_deg=75.0)
        np.testing.assert_array_equal(actual, model.photon_radiance(wave, other, 0.0))
        np.testing.assert_array_equal(
            actual, model.photon_radiance(wave, UNPLACED, TIME_D)
        )


def test_leinert_follows_the_frame_and_the_time():
    """Fails if the sightline ignores the position or the Sun's motion.

    The two frames and the two times give four different sightlines, so the
    four radiances must be distinct. The asymmetric pointing also differs from
    the swapped (latitude, longitude) lookup by more than ten percent.
    """
    wave = jnp.array([500.0, 550.0, 700.0])
    model = LeinertZodi()
    latitude_deg, longitude_deg = _pointing()
    expected = (
        zodi.specific_intensity(
            longitude_deg, latitude_deg, wave, photon_units=True, anchor="table19"
        )
        / 1000
    )
    swapped = (
        zodi.specific_intensity(
            latitude_deg, longitude_deg, wave, photon_units=True, anchor="table19"
        )
        / 1000
    )
    np.testing.assert_allclose(
        model.photon_radiance(wave, FRAME, TIME_D), expected, rtol=3e-14, atol=0
    )
    assert np.all(np.abs(np.asarray(expected / swapped) - 1) > 0.1)
    later = model.photon_radiance(wave, FRAME, TIME_D + 90.0)
    elsewhere = model.photon_radiance(
        wave, SkyFrame(10.0, ra_deg=250.0, dec_deg=30.0), TIME_D
    )
    assert np.all(np.asarray(later) != np.asarray(expected))
    assert np.all(np.asarray(elsewhere) != np.asarray(expected))


def test_models_share_table19_colour_at_the_reference_sightline():
    """Fails on a wrong colour or sr scale.

    Setting AYO to the table's AB magnitude at (longitude 90, latitude 0)
    makes it coincide exactly with Leinert there; the cached table of that
    radiance reads back unchanged. 3e-14 bounds float64 log roundoff.
    """
    wave = jnp.array([500.0, 550.0, 700.0])
    mag = zodi.surface_brightness_ab_mag(
        jnp.array(90.0), jnp.array(0.0), jnp.array(550.0), anchor="table19"
    )
    radiance = (
        zodi.specific_intensity(
            jnp.array(90.0), jnp.array(0.0), wave, photon_units=True, anchor="table19"
        )
        / 1000
    )
    fixed = AYOZodi(surface_brightness_mag_arcsec2=mag)
    np.testing.assert_allclose(
        fixed.photon_radiance(wave, FRAME, TIME_D), radiance, rtol=3e-14, atol=0
    )
    cached = PrecomputedZodi(wave, radiance)
    np.testing.assert_allclose(
        cached.photon_radiance(wave, FRAME, TIME_D), radiance, rtol=3e-14, atol=0
    )


def test_leinert_refuses_an_unplaced_frame_and_the_others_do_not():
    """A sightline-dependent model names the missing position; the rest ignore it."""
    wave = jnp.array([550.0])
    with pytest.raises(ValueError, match="no sky position"):
        LeinertZodi().photon_radiance(wave, UNPLACED, TIME_D)
    with pytest.raises(ValueError, match="no sky position"):
        eqx.filter_jit(LeinertZodi().photon_radiance)(wave, UNPLACED, TIME_D)
    assert np.all(np.isfinite(AYOZodi().photon_radiance(wave, UNPLACED, TIME_D)))
    table = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.array([1.0, 2.0]))
    assert np.all(np.isfinite(table.photon_radiance(wave, UNPLACED, TIME_D)))


@pytest.mark.parametrize("kind", [AYOZodi, LeinertZodi, PrecomputedZodi])
def test_radiance_jax(kind):
    """Fails if JIT or wavelength vmap changes the radiance.

    3e-14 is float64 roundoff. Leinert also reduces the Sun's longitude modulo
    360 degrees, which compiled and eager code round differently by about
    3e-12 relative; 1e-9 bounds it.
    """
    rtol = 1e-9 if kind is LeinertZodi else 3e-14
    model = (
        kind(jnp.array([500.0, 700.0]), jnp.array([2e10, 5e10]))
        if kind is PrecomputedZodi
        else kind()
    )
    wave = jnp.array([500.0, 550.0, 700.0])

    def evaluate(w):
        return model.photon_radiance(w, FRAME, TIME_D)

    loop = jnp.stack([evaluate(w) for w in wave])
    np.testing.assert_allclose(eqx.filter_jit(evaluate)(wave), loop, rtol=rtol, atol=0)
    np.testing.assert_allclose(jax.vmap(evaluate)(wave), loop, rtol=rtol, atol=0)


def test_time_is_traced_and_batched():
    """Fails if the observation time cannot be a traced or vmapped argument.

    The Sun's longitude is a large angle reduced modulo 360 degrees, so a
    vmapped and a looped evaluation agree to the float64 roundoff of that
    reduction (about 1e-11 relative); 1e-9 bounds it.
    """
    wave = jnp.array([550.0])
    model = LeinertZodi()
    times = jnp.array([0.0, 100.0, 200.0])
    batched = jax.vmap(lambda t: model.photon_radiance(wave, FRAME, t))(times)
    loop = jnp.stack([model.photon_radiance(wave, FRAME, t) for t in times])
    np.testing.assert_allclose(batched, loop, rtol=1e-9, atol=0)
    assert len(set(np.asarray(batched).ravel().tolist())) == 3


def test_precomputed_interpolation_and_support():
    """Fails if interpolation is constant, clamped, or reshaped incorrectly.

    Tolerance basis: exact-algebra for a linear spectrum, y = 2*x - 100.
    Unsupported queries must be NaN, including under JIT.
    """
    model = PrecomputedZodi(jnp.array([400.0, 600.0]), jnp.array([700.0, 1100.0]))
    wave = jnp.array([[399.0, 400.0, 450.0], [600.0, 601.0, jnp.nan]])
    expected = np.array([[np.nan, 700.0, 800.0], [1100.0, np.nan, np.nan]])
    actual = eqx.filter_jit(model.photon_radiance)(wave, FRAME, TIME_D)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert model.photon_radiance(jnp.array(450.0), FRAME, TIME_D) == 800.0
