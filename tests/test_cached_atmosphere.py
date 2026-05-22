"""Tests for PrecomputedReflectivity and from_default_setup_cached.

Uses the fake engines from test_exojax_atmosphere so the spectrum is
deterministic without triggering real ExoJAX downloads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from skyscapes.atmosphere import (
    AbstractAtmosphere,
    PrecomputedReflectivity,
)

from .test_exojax_atmosphere import make_atmosphere


def test_precomputed_subclasses_abstract_atmosphere():
    """``PrecomputedReflectivity`` satisfies the ``AbstractAtmosphere`` contract."""
    atm = make_atmosphere(K=1)
    cached = PrecomputedReflectivity.from_atmosphere(atm)
    assert isinstance(cached, AbstractAtmosphere)


def test_precomputed_spectrum_matches_source_at_grid_wavelengths():
    """At a wavelength on the cached grid the spectrum matches source closely.

    Cubic interpolation between grid points loses some precision; we
    test on wavelengths sampled from the grid itself to keep the
    interpolation lossless.
    """
    K = 1
    atm = make_atmosphere(K=K)
    cached = PrecomputedReflectivity.from_atmosphere(atm)

    # Sample a wavelength on the grid (no interp error).
    nu_mid = atm.nu_grid[atm.n_nu // 2]
    wl_nm = jnp.array(1.0e7 / float(nu_mid))

    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    out_atm = atm.reflected_spectrum(phase, dist, wl_nm)
    out_cached = cached.reflected_spectrum(phase, dist, wl_nm)
    assert jnp.allclose(out_atm, out_cached, rtol=1.0e-5)


def test_precomputed_spectrum_cube_shape():
    """``reflected_spectrum_cube`` returns ``(W, K, T)`` and is finite."""
    K, T, W = 2, 3, 5
    atm = make_atmosphere(K=K)
    cached = PrecomputedReflectivity.from_atmosphere(atm)
    phase = jnp.full((K, T), 0.5)
    dist = jnp.full((K, T), 1.0)
    wls = jnp.linspace(450.0, 900.0, W)
    out = cached.reflected_spectrum_cube(phase, dist, wls)
    assert out.shape == (W, K, T)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_save_load_roundtrip_preserves_values(tmp_path):
    """``save`` then ``load`` yields the same spectrum."""
    K = 2
    atm = make_atmosphere(K=K)
    cached = PrecomputedReflectivity.from_atmosphere(atm)
    path = tmp_path / "cached.npz"
    cached.save(path)
    loaded = PrecomputedReflectivity.load(path)

    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)
    out_orig = cached.reflected_spectrum(phase, dist, wl)
    out_loaded = loaded.reflected_spectrum(phase, dist, wl)
    assert jnp.allclose(out_orig, out_loaded, rtol=1.0e-7)


def test_load_rejects_mismatched_cache_format_version(tmp_path):
    """Loading a cache with a different format version raises clearly."""
    import numpy as np

    path = tmp_path / "bad.npz"
    np.savez(
        path,
        cache_format_version=np.asarray(9999),  # bogus future version
        Rp_Rearth=np.ones(1),
        reflectivity=np.ones((1, 50)),
        nu_grid=np.linspace(1.0e4, 2.5e4, 50),
    )
    with pytest.raises(ValueError, match="format version"):
        PrecomputedReflectivity.load(path)


def test_precomputed_jit_stable():
    """JIT-compiling ``reflected_spectrum`` matches eager output."""
    K = 1
    atm = make_atmosphere(K=K)
    cached = PrecomputedReflectivity.from_atmosphere(atm)
    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)

    f = jax.jit(lambda a: a.reflected_spectrum(phase, dist, wl))
    eager = cached.reflected_spectrum(phase, dist, wl)
    jitted = f(cached)
    assert jnp.allclose(eager, jitted, rtol=1.0e-6)


def test_repr_mentions_precomputed_nature():
    """The repr identifies it as a precomputed (no-RT) atmosphere."""
    atm = make_atmosphere(K=2)
    cached = PrecomputedReflectivity.from_atmosphere(atm)
    s = repr(cached)
    assert "PrecomputedReflectivity" in s
    assert "K=2" in s
    assert "Wavelength:" in s


def test_cache_key_is_stable_and_input_sensitive():
    """The same kwargs hash to the same key; changing any kwarg flips it."""
    from skyscapes.atmosphere.exojax.atmosphere import _cache_key

    base_kwargs = {
        "Rp_Rearth": jnp.ones(1),
        "log_mmrs": {"H2O": jnp.full((1,), -2.5)},
        "T_eq_K": jnp.full((1,), 288.0),
        "wavelength_min_nm": 400.0,
        "n_layers": 100,
    }
    key1 = _cache_key(**base_kwargs)
    key2 = _cache_key(**base_kwargs)
    assert key1 == key2, "same kwargs should hash identically"

    # Flip a per-planet array value -- key must change.
    modified = {**base_kwargs, "Rp_Rearth": jnp.full((1,), 1.5)}
    key3 = _cache_key(**modified)
    assert key3 != key1

    # Flip a scalar config -- key must change.
    modified2 = {**base_kwargs, "n_layers": 50}
    key4 = _cache_key(**modified2)
    assert key4 != key1

    # Flip a dict-of-arrays value -- key must change.
    modified3 = {
        **base_kwargs,
        "log_mmrs": {"H2O": jnp.full((1,), -3.0)},
    }
    key5 = _cache_key(**modified3)
    assert key5 != key1


def test_cache_key_handles_eqx_modules():
    """Passing an AbstractMmrProfile (eqx.Module) is hashable."""
    from skyscapes.atmosphere.exojax.atmosphere import _cache_key
    from skyscapes.atmosphere.exojax.components import (
        ConstantMmr,
        StratosphericPeakMmr,
    )

    profile_const = ConstantMmr(log_mmr=jnp.full((1,), -2.5))
    profile_peak = StratosphericPeakMmr(
        log_peak_mmr=jnp.full((1,), -5.0),
        log_peak_pressure_bar=jnp.full((1,), -2.0),
        log_sigma_decades=jnp.full((1,), 0.5),
    )
    k_const = _cache_key(log_mmrs={"O3": profile_const})
    k_peak = _cache_key(log_mmrs={"O3": profile_peak})
    assert k_const != k_peak, "different profile types should hash differently"


def test_cached_setup_roundtrip_using_fake_engines(tmp_path, monkeypatch):
    """``from_default_setup_cached`` round-trips: build -> save -> load.

    Uses the fake engines from test_exojax_atmosphere by monkey-patching
    ``build_exojax_engines`` to return them, so we don't actually
    invoke ExoJAX. Verifies that the second call (cache hit) returns
    the same PrecomputedReflectivity as the first (cache miss).
    """
    import skyscapes.atmosphere.exojax.atmosphere as exojax_atmosphere_mod
    from skyscapes.atmosphere import ExoJaxAtmosphere

    K = 1
    atm = make_atmosphere(K=K)

    # Monkey-patch build_exojax_engines to skip ExoJAX entirely.
    def fake_build(*, molecules, **_):
        # The function signature isn't load-bearing for the test --
        # what matters is that downstream callers get something
        # consumable by _assemble_species + _assemble_bulk.
        species_prebuilt = {
            s.name: {
                "molmass": s.molmass,
                "opa": s.opa,
                "rayleigh_xs": s.rayleigh_xs,
            }
            for s in atm.species
        }
        bulk_prebuilt = {
            "name": atm.bulk.name,
            "molmass": atm.bulk.molmass,
            "rayleigh_xs": atm.bulk.rayleigh_xs,
        }
        return {
            "rt_engine": atm.rt_engine,
            "nu_grid": atm.nu_grid,
            "n_nu": atm.n_nu,
            "species_prebuilt": species_prebuilt,
            "bulk_prebuilt": bulk_prebuilt,
        }

    monkeypatch.setattr(exojax_atmosphere_mod, "build_exojax_engines", fake_build)

    kwargs = dict(
        Rp_Rearth=jnp.ones(K),
        log_mmrs={"H2O": jnp.full((K,), -3.0)},
        T_eq_K=jnp.full((K,), 288.0),
        T_alpha=jnp.full((K,), 0.07),
        log_surface_albedo=jnp.full((K,), jnp.log10(0.3)),
        log_gravity_cgs=jnp.full((K,), jnp.log10(981.0)),
        wavelength_min_nm=400.0,
        wavelength_max_nm=1000.0,
        n_wavenumbers=100,
        n_layers=20,
        cache_dir=tmp_path,
    )

    # First call: cache miss, builds + saves.
    cached_a = ExoJaxAtmosphere.from_default_setup_cached(**kwargs)
    assert isinstance(cached_a, PrecomputedReflectivity)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1, "cache file should be written on miss"

    # Second call: cache hit, loads same file.
    cached_b = ExoJaxAtmosphere.from_default_setup_cached(**kwargs)
    assert isinstance(cached_b, PrecomputedReflectivity)
    files_after = list(tmp_path.glob("*.npz"))
    assert len(files_after) == 1, "cache hit should not write a new file"

    # The two should produce identical spectra at the same wavelength.
    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)
    assert jnp.allclose(
        cached_a.reflected_spectrum(phase, dist, wl),
        cached_b.reflected_spectrum(phase, dist, wl),
    )
