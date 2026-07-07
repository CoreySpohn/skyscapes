"""Tests for skyscapes.background.leinert (ported from orbix's zodiacal tests)."""

import jax
import jax.numpy as jnp
from hwoutils.conversions import flux_jy_to_mag, mag_to_flux_jy

from skyscapes.background.leinert import (
    ayo_default_zodi_flux_jy,
    ayo_default_zodi_mag,
    create_zodi_spectrum_jax,
    leinert_zodi_factor,
    leinert_zodi_mag,
    zodi_color_correction,
)


class TestZodiacalLight:
    """Test zodiacal light functions."""

    def test_ayo_default_v_band(self):
        """AYO default at V-band should be exactly 22 mag."""
        mag = ayo_default_zodi_mag(550.0)
        assert jnp.isclose(mag, 22.0, atol=0.01)

    def test_leinert_factor_ecliptic_pole(self):
        """Factor at ecliptic pole should be 77/259 approx 0.297."""
        factor = leinert_zodi_factor(90.0, 90.0)
        assert jnp.isclose(factor, 77.0 / 259.0, atol=0.01)

    def test_leinert_factor_symmetry(self):
        """Factor should be symmetric in ecliptic latitude."""
        f_pos = leinert_zodi_factor(30.0, 135.0)
        f_neg = leinert_zodi_factor(-30.0, 135.0)
        assert jnp.isclose(f_pos, f_neg, atol=1e-6)

    def test_leinert_mag_brighter_at_lower_lat(self):
        """Lower ecliptic latitude should be brighter (lower mag)."""
        mag_low = leinert_zodi_mag(550.0, 10.0, 135.0)
        mag_high = leinert_zodi_mag(550.0, 60.0, 135.0)
        assert float(mag_low) < float(mag_high)

    def test_mag_flux_roundtrip(self):
        """Mag -> flux -> mag should roundtrip (zero-point-agnostic)."""
        mag = 22.0
        flux = mag_to_flux_jy(mag)
        mag_back = flux_jy_to_mag(flux)
        assert jnp.isclose(mag, mag_back, atol=1e-10)

    def test_color_correction_identity(self):
        """Color correction at reference wavelength should be 1.0."""
        cc = zodi_color_correction(550.0, 550.0, photon_units=False)
        assert jnp.isclose(cc, 1.0, atol=1e-6)

    def test_color_correction_photon_vs_power_units(self):
        """Photon units should differ from power units by a lambda factor."""
        corr_photon = zodi_color_correction(700.0, photon_units=True)
        corr_power = zodi_color_correction(700.0, photon_units=False)
        assert corr_photon != corr_power

    def test_ayo_flux_positive(self):
        """AYO-default zodiacal flux (Jy/arcsec^2) should be positive."""
        flux = ayo_default_zodi_flux_jy(550.0)
        assert flux > 0

    def test_zodi_jit(self):
        """Zodiacal light functions should JIT-compile."""
        f = jax.jit(leinert_zodi_mag)
        mag = f(550.0, 30.0, 135.0)
        assert jnp.isfinite(mag)

    def test_zodi_grad(self):
        """Zodiacal light should be differentiable w.r.t. wavelength."""
        grad_fn = jax.grad(lambda wl: leinert_zodi_mag(wl, 30.0, 135.0))
        g = grad_fn(550.0)
        assert jnp.isfinite(g)


class TestCreateZodiSpectrumJax:
    """Tests for zodiacal light spectrum generation."""

    def test_output_shape(self):
        """Output should match input wavelength array shape."""
        wavelengths = jnp.array([400.0, 500.0, 600.0, 700.0, 800.0])
        spectrum = create_zodi_spectrum_jax(wavelengths)
        assert spectrum.shape == wavelengths.shape

    def test_all_positive(self):
        """All spectral values should be positive."""
        wavelengths = jnp.linspace(400.0, 900.0, 10)
        spectrum = create_zodi_spectrum_jax(wavelengths)
        assert jnp.all(spectrum > 0)


def test_near_sun_and_out_of_range_queries_are_finite():
    """Near-Sun and out-of-range zodi queries clamp to finite values, not NaN."""
    from skyscapes.background.leinert import leinert_zodi_factor, zodi_color_correction

    assert jnp.isfinite(
        leinert_zodi_factor(7.0, 12.0)
    )  # near-Sun exclusion cell; pre-fix NaN
    assert jnp.isfinite(zodi_color_correction(100.0))  # out-of-range wavelength
