"""Tests for PrecomputedPhysicalModel and from_default_setup_cached.

Uses the fake engines from test_physical_model_exojax so the spectrum
is deterministic without triggering real ExoJAX downloads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from skyscapes.physical_model import (
    AbstractPhysicalModel,
    PrecomputedPhysicalModel,
)

from .test_physical_model_exojax import make_physical_model


def test_precomputed_subclasses_abstract_physical_model():
    """``PrecomputedPhysicalModel`` satisfies the ``AbstractPhysicalModel`` contract."""
    model = make_physical_model(K=1)
    cached = PrecomputedPhysicalModel.from_physical_model(model)
    assert isinstance(cached, AbstractPhysicalModel)


def test_precomputed_spectrum_matches_source_at_grid_wavelengths():
    """At a wavelength on the cached grid the spectrum matches source closely.

    Cubic interpolation between grid points loses some precision; we
    test on wavelengths sampled from the grid itself to keep the
    interpolation lossless.
    """
    K = 1
    model = make_physical_model(K=K)
    cached = PrecomputedPhysicalModel.from_physical_model(model)

    # Sample a wavelength on the grid (no interp error).
    nu_mid = model.nu_grid[model.n_nu // 2]
    wl_nm = jnp.array(1.0e7 / float(nu_mid))

    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    Rp = jnp.ones(K)
    out_model = model.contrast(phase, dist, wl_nm, Rp)
    out_cached = cached.contrast(phase, dist, wl_nm, Rp)
    assert jnp.allclose(out_model, out_cached, rtol=1.0e-5)


def test_precomputed_spectrum_cube_shape():
    """``contrast_cube`` returns ``(W, K, T)`` and is finite."""
    K, T, W = 2, 3, 5
    model = make_physical_model(K=K)
    cached = PrecomputedPhysicalModel.from_physical_model(model)
    phase = jnp.full((K, T), 0.5)
    dist = jnp.full((K, T), 1.0)
    wls = jnp.linspace(450.0, 900.0, W)
    Rp = jnp.ones(K)
    out = cached.contrast_cube(phase, dist, wls, Rp)
    assert out.shape == (W, K, T)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_save_load_roundtrip_preserves_values(tmp_path):
    """``save`` then ``load`` yields the same spectrum."""
    K = 2
    model = make_physical_model(K=K)
    cached = PrecomputedPhysicalModel.from_physical_model(model)
    path = tmp_path / "cached.npz"
    cached.save(path)
    loaded = PrecomputedPhysicalModel.load(path)

    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)
    Rp = jnp.ones(K)
    out_orig = cached.contrast(phase, dist, wl, Rp)
    out_loaded = loaded.contrast(phase, dist, wl, Rp)
    assert jnp.allclose(out_orig, out_loaded, rtol=1.0e-7)


def test_load_rejects_mismatched_cache_format_version(tmp_path):
    """Loading a cache with a different format version raises clearly."""
    import numpy as np

    path = tmp_path / "bad.npz"
    np.savez(
        path,
        cache_format_version=np.asarray(9999),  # bogus future version
        reflectivity=np.ones((1, 50)),
        nu_grid=np.linspace(1.0e4, 2.5e4, 50),
    )
    with pytest.raises(ValueError, match="format version"):
        PrecomputedPhysicalModel.load(path)


def test_precomputed_jit_stable():
    """JIT-compiling ``contrast`` matches eager output."""
    K = 1
    model = make_physical_model(K=K)
    cached = PrecomputedPhysicalModel.from_physical_model(model)
    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)
    Rp = jnp.ones(K)

    f = jax.jit(lambda a: a.contrast(phase, dist, wl, Rp))
    eager = cached.contrast(phase, dist, wl, Rp)
    jitted = f(cached)
    assert jnp.allclose(eager, jitted, rtol=1.0e-6)


def test_repr_mentions_precomputed_nature():
    """The repr identifies it as a precomputed (no-RT) physical model."""
    model = make_physical_model(K=2)
    cached = PrecomputedPhysicalModel.from_physical_model(model)
    s = repr(cached)
    assert "PrecomputedPhysicalModel" in s
    assert "K=2" in s
    assert "Wavelength:" in s


def test_cache_key_is_stable_and_input_sensitive():
    """The same kwargs hash to the same key; changing any kwarg flips it."""
    from skyscapes.physical_model.exojax.physical_model import _cache_key

    base_kwargs = {
        "log_mmrs": {"H2O": jnp.full((1,), -2.5)},
        "T_eq_K": jnp.full((1,), 288.0),
        "wavelength_min_nm": 400.0,
        "n_layers": 100,
    }
    key1 = _cache_key(**base_kwargs)
    key2 = _cache_key(**base_kwargs)
    assert key1 == key2, "same kwargs should hash identically"

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
    from skyscapes.physical_model.exojax.components import (
        ConstantMmr,
        StratosphericPeakMmr,
    )
    from skyscapes.physical_model.exojax.physical_model import _cache_key

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

    Uses the fake engines from test_physical_model_exojax by monkey-patching
    ``build_exojax_engines`` to return them, so we don't actually
    invoke ExoJAX. Verifies that the second call (cache hit) returns
    the same PrecomputedPhysicalModel as the first (cache miss).
    """
    import skyscapes.physical_model.exojax.physical_model as exojax_pm_mod
    from skyscapes.physical_model import ExoJaxPhysicalModel

    K = 1
    model = make_physical_model(K=K)

    # Monkey-patch build_exojax_engines to skip ExoJAX entirely.
    def fake_build(*, molecules, **_):
        species_prebuilt = {
            s.name: {
                "molmass": s.molmass,
                "opa": s.opa,
                "rayleigh_xs": s.rayleigh_xs,
            }
            for s in model.species
        }
        bulk_prebuilt = {
            "name": model.bulk.name,
            "molmass": model.bulk.molmass,
            "rayleigh_xs": model.bulk.rayleigh_xs,
        }
        return {
            "rt_engine": model.rt_engine,
            "nu_grid": model.nu_grid,
            "n_nu": model.n_nu,
            "species_prebuilt": species_prebuilt,
            "bulk_prebuilt": bulk_prebuilt,
        }

    monkeypatch.setattr(exojax_pm_mod, "build_exojax_engines", fake_build)

    kwargs = dict(
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
    cached_a = ExoJaxPhysicalModel.from_default_setup_cached(**kwargs)
    assert isinstance(cached_a, PrecomputedPhysicalModel)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1, "cache file should be written on miss"

    # Second call: cache hit, loads same file.
    cached_b = ExoJaxPhysicalModel.from_default_setup_cached(**kwargs)
    assert isinstance(cached_b, PrecomputedPhysicalModel)
    files_after = list(tmp_path.glob("*.npz"))
    assert len(files_after) == 1, "cache hit should not write a new file"

    # The two should produce identical spectra at the same wavelength.
    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)
    Rp = jnp.ones(K)
    assert jnp.allclose(
        cached_a.contrast(phase, dist, wl, Rp),
        cached_b.contrast(phase, dist, wl, Rp),
    )
