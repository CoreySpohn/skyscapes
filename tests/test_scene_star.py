"""scene.Star — AbstractStar, SimpleStar, SpectrumStar."""

from __future__ import annotations

import jax.numpy as jnp

from skyscapes.scene import AbstractStar, SimpleStar, SpectrumStar


def test_simple_star_fields():
    """SimpleStar stores Ms_kg / dist_pc and is an AbstractStar."""
    s = SimpleStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    assert s.Ms_kg == 1.989e30
    assert s.dist_pc == 10.0
    assert isinstance(s, AbstractStar)


def test_simple_star_flux_is_flat():
    """SimpleStar returns constant flux regardless of wavelength or time."""
    s = SimpleStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    f1 = s.spec_flux_density(500.0, 0.0)
    f2 = s.spec_flux_density(800.0, 100.0)
    assert jnp.allclose(f1, 1e9)
    assert jnp.allclose(f2, 1e9)


def test_spectrum_star_interpolates():
    """SpectrumStar Jy→photon conversion reproduces analytic value."""
    wl = jnp.linspace(400.0, 1000.0, 7)
    t = jnp.linspace(0.0, 10.0, 5)
    # 1 Jy everywhere → same phot number at each (wl, t) point
    flux_jy = jnp.ones((wl.size, t.size))
    s = SpectrumStar(
        Ms_kg=1.989e30,
        dist_pc=10.0,
        wavelengths_nm=wl,
        times_jd=t,
        flux_density_jy=flux_jy,
    )
    f = s.spec_flux_density(600.0, 5.0)
    # Jy→phot: 1 Jy * 1e-26 / (600 nm * h) ≈ 2.514e7
    expected = 1.0 * 1e-26 / (600.0 * 6.62607015e-34)
    assert jnp.allclose(f, expected, rtol=1e-4)


def test_spectrum_star_isinstance():
    """SpectrumStar is an AbstractStar."""
    wl = jnp.linspace(400.0, 1000.0, 3)
    t = jnp.linspace(0.0, 1.0, 3)
    s = SpectrumStar(
        Ms_kg=1.989e30,
        dist_pc=10.0,
        wavelengths_nm=wl,
        times_jd=t,
        flux_density_jy=jnp.zeros((wl.size, t.size)),
    )
    assert isinstance(s, AbstractStar)
