"""skyscapes.atmosphere.ExoJaxAtmosphere -- adapter for ExoJAX's 2-stream RT.

The real ExoJAX engines require database downloads (POKAZATEL, UCL-4000,
etc.) on first use, which is too heavy for CI. These tests instead
construct ExoJaxAtmosphere with fake RT/opa engines that mimic the
ExoJAX API shapes, so we exercise the adapter's vmap/interp/Lambert
math without touching ExoJAX's internals.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from skyscapes.atmosphere import (
    AbstractAtmosphere,
    ExoJaxAtmosphere,
    LambertianAtmosphere,
)
from skyscapes.atmosphere.exojax.components import (
    Absorption,
    BulkGasResidual,
    ConstantMmr,
    GrayCloud,
    MieCloud,
    MolecularSpecies,
    PowerLawTPProfile,
    RayleighScattering,
    WavelengthDependentSurface,
)

# ---------------------------------------------------------------------------
# Fake ExoJAX engines with the same call signature shape contracts.
# ---------------------------------------------------------------------------


class _FakeRT:
    """Mimics exojax.rt.ArtReflectPure's interface used by ExoJaxAtmosphere.

    All quantities are JAX-compatible and return arrays of the expected
    shape so the adapter math can run end-to-end inside JIT.
    """

    def __init__(self, nu_grid, n_layers):
        self.nu_grid = nu_grid
        self.n_layers = n_layers
        # Isobaric grid spanning 1e-5 -- 1 bar (log-spaced).
        self.pressure = jnp.logspace(-5.0, 0.0, n_layers)

    def powerlaw_temperature(self, T_eq, alpha):
        return T_eq * self.pressure**alpha

    def constant_mmr_profile(self, value):
        return value * jnp.ones(self.n_layers)

    def opacity_profile_xs(self, xsmatrix, mmr_profile, molmass, gravity):
        # Real ExoJAX returns layer-wise tau; for the fake, just scale
        # xsmatrix by mmr * molmass / gravity (dimensionally similar).
        return xsmatrix * mmr_profile[:, None] * (molmass / gravity)

    def run(self, dtau, ssa, g, refl_surface, incoming_flux):
        # Fake "reflectivity": surface albedo attenuated by total tau.
        # Mirrors the structure of the real 2-stream output -- shape
        # (n_nu,) -- without doing the actual flux-adding math.
        total_tau = jnp.sum(dtau, axis=0)
        attenuation = jnp.exp(-total_tau)
        return refl_surface * attenuation * jnp.mean(incoming_flux)


class _FakeOpa:
    """Mimics exojax.opacity.OpaPremodit's xsmatrix method."""

    def __init__(self, nu_grid, n_layers, peak_xs: float = 1e-22):
        self.nu_grid = nu_grid
        self.n_layers = n_layers
        self.peak_xs = peak_xs

    def xsmatrix(self, Tarr, pressure):
        # Returns (n_layers, n_nu) cross-section matrix. We just emit
        # a flat baseline so the adapter math is exercised; the
        # absolute value doesn't matter for shape tests.
        return self.peak_xs * jnp.ones((self.n_layers, len(self.nu_grid)))


_FAKE_MOL_NAMES = ("H2O", "CO2", "CH4", "O2", "O3")
_FAKE_MOL_MASSES = (18.0, 44.0, 16.0, 32.0, 48.0)
_FAKE_LOG_MMRS = (-3.0, -4.0, -6.0, -1.0, -7.0)


def make_atmosphere(K: int = 2, n_nu: int = 100, n_layers: int = 20):
    """Build an ExoJaxAtmosphere with fake engines + default components.

    Constructs a five-molecule species tuple (H2O, CO2, CH4, O2, O3)
    + an N2 bulk residual to exercise the new composition-based
    structure without requiring real ExoJAX engines.
    """
    nu_grid = jnp.linspace(1.0e4, 2.5e4, n_nu)  # 400 -- 1000 nm range
    rt = _FakeRT(nu_grid, n_layers)
    # Tiny Rayleigh cross-sections keep scattering small relative to
    # absorption -- exercises the contract without re-running ExoJAX's
    # polarizability code path.
    ray_xs_per_mol = jnp.full((n_nu,), 1.0e-27)
    surface_albedo_spectrum = jnp.ones((n_nu,))

    species = tuple(
        MolecularSpecies(
            profile=ConstantMmr(log_mmr=jnp.full((K,), log_mmr)),
            name=name,
            molmass=mass,
            opa=_FakeOpa(nu_grid, n_layers),
            rayleigh_xs=ray_xs_per_mol,
        )
        for name, mass, log_mmr in zip(
            _FAKE_MOL_NAMES, _FAKE_MOL_MASSES, _FAKE_LOG_MMRS, strict=True
        )
    )
    bulk = BulkGasResidual(
        name="N2",
        molmass=28.0134,
        rayleigh_xs=jnp.full((n_nu,), 1.0e-27),
    )

    return ExoJaxAtmosphere(
        Rp_Rearth=jnp.ones(K),
        log_gravity_cgs=jnp.full((K,), jnp.log10(981.0)),
        species=species,
        bulk=bulk,
        tp_profile=PowerLawTPProfile(
            T_eq_K=jnp.full((K,), 288.0),
            T_alpha=jnp.full((K,), 0.07),
        ),
        absorption=Absorption(),
        scattering=RayleighScattering(),
        clouds=GrayCloud(
            log_pressure_bar=jnp.full((K,), -10.0),  # cloud essentially off
            log_opt_depth=jnp.full((K,), -10.0),  # tau ~ 0
        ),
        surface=WavelengthDependentSurface(
            log_albedo=jnp.full((K,), jnp.log10(0.3)),
            spectrum=surface_albedo_spectrum,
        ),
        rt_engine=rt,
        nu_grid=nu_grid,
        n_nu=n_nu,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_is_abstract_atmosphere():
    """ExoJaxAtmosphere satisfies the AbstractAtmosphere interface."""
    atm = make_atmosphere()
    assert isinstance(atm, AbstractAtmosphere)


def test_reflected_spectrum_shape():
    """Output has shape (K, T) matching the AbstractAtmosphere contract."""
    K = 3
    T = 5
    atm = make_atmosphere(K=K)
    phase = jnp.full((K, T), 0.5)
    dist = jnp.full((K, T), 1.0)
    wl = jnp.array(550.0)
    out = atm.reflected_spectrum(phase, dist, wl)
    assert out.shape == (K, T)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(jnp.all(out >= 0.0))


def test_reflected_spectrum_jit():
    """Output is JIT-stable; subsequent calls reuse the cached trace."""
    K, T = 2, 3
    atm = make_atmosphere(K=K)
    phase = jnp.full((K, T), 0.7)
    dist = jnp.full((K, T), 1.2)
    wl = jnp.array(600.0)

    f = jax.jit(lambda a: a.reflected_spectrum(phase, dist, wl))
    out_jit = f(atm)
    out_eager = atm.reflected_spectrum(phase, dist, wl)
    assert jnp.allclose(out_jit, out_eager, rtol=1e-5)


def test_lambert_geometry_dominates_when_surface_albedo_drives_signal():
    """High surface albedo + low absorption => Ag * Lambert_phase * (Rp/d)^2.

    Verifies that the per-planet contrast collapses to the closed-form
    Lambertian result in the limit where the atmosphere is transparent.
    The fake RT returns plane-parallel (spherical) reflectivity equal
    to surface_albedo=0.3; reflected_spectrum then applies the 2/3
    Lambertian-sphere conversion (Seager 2010 eq 3.36), so the
    equivalent geometric-albedo Lambertian comparator has
    Ag = (2/3) * 0.3 = 0.2.
    """
    K = 2
    T = 4
    surface_albedo = 0.3
    # Use very low MMRs so the fake exp(-tau) attenuation is ~1.
    atm = make_atmosphere(K=K)
    # Override every species' profile log_mmr to be tiny so tau ~ 0
    # and reflectivity ~ albedo. Assumes ConstantMmr profiles (the
    # default in make_atmosphere).
    atm = eqx.tree_at(
        lambda m: tuple(s.profile.log_mmr for s in m.species),
        atm,
        replace=tuple(jnp.full((K,), -30.0) for _ in atm.species),
    )

    phase = jnp.full((K, T), 0.5)
    dist = jnp.full((K, T), 1.0)
    wl = jnp.array(550.0)
    out = atm.reflected_spectrum(phase, dist, wl)

    # Compare to the LambertianAtmosphere closed-form contrast, with
    # Ag set to the geometric-albedo equivalent of the plane-parallel
    # spherical albedo coming out of the fake RT.
    lam = LambertianAtmosphere(
        Rp_Rearth=jnp.ones(K),
        Ag=jnp.full((K,), (2.0 / 3.0) * surface_albedo),
    )
    out_lam = lam.reflected_spectrum(phase, dist, wl)
    # Use a tight relative tolerance with a small atol that does not
    # swamp the ~1e-10 contrasts (default jnp.allclose atol=1e-8 would).
    assert jnp.allclose(out, out_lam, rtol=0.01, atol=1.0e-15)


def test_vmap_over_wavelength_returns_cube():
    """Vmapping over wavelength expands the output to (n_wave, K, T)."""
    K, T = 2, 3
    atm = make_atmosphere(K=K)
    phase = jnp.full((K, T), 0.5)
    dist = jnp.full((K, T), 1.0)
    wls = jnp.array([450.0, 550.0, 700.0, 900.0])

    cube = jax.vmap(atm.reflected_spectrum, in_axes=(None, None, 0))(phase, dist, wls)
    assert cube.shape == (4, K, T)
    assert bool(jnp.all(jnp.isfinite(cube)))


def test_reflected_spectrum_cube_matches_vmap():
    """``reflected_spectrum_cube`` agrees with vmapped scalar calls.

    The cube method exists to avoid re-running the 2-stream RT solver
    once per wavelength; result-wise it should equal the slow
    vmap-based path to numerical precision.
    """
    K, T = 2, 3
    atm = make_atmosphere(K=K)
    phase = jnp.full((K, T), 0.5)
    dist = jnp.full((K, T), 1.0)
    wls = jnp.array([450.0, 550.0, 700.0, 900.0])

    cube_via_method = atm.reflected_spectrum_cube(phase, dist, wls)
    cube_via_vmap = jax.vmap(atm.reflected_spectrum, in_axes=(None, None, 0))(
        phase, dist, wls
    )

    assert cube_via_method.shape == cube_via_vmap.shape == (4, K, T)
    assert jnp.allclose(cube_via_method, cube_via_vmap, rtol=1e-5)


def test_rayleigh_xs_is_wired_through_to_output():
    """A large bump in Rayleigh cross-section visibly changes the spectrum.

    Through the fake RT (pure ``exp(-tau)`` attenuation, no scattered-
    flux contribution), increasing Rayleigh opacity decreases the
    output. Real-physics behavior -- Rayleigh actually *brightening*
    the blue end via scattered flux, with the canonical ``1/lambda^4``
    wavelength dependence -- is exercised against ``ArtReflectPure``
    in the dev scripts that drive a full ExoJAX setup. Here we only
    verify that the Rayleigh fields are plumbed through to
    ``opacity_profile_xs``.
    """
    K = 1
    atm_low = make_atmosphere(K=K)
    # Note: this xs value (1e-2) is wildly unphysical -- real Rayleigh
    # is ~1e-26. We need a value this large to push the fake RT's
    # ``exp(-total_tau)`` out of the float32 noise floor, since the
    # fake doesn't use ``opacity_profile_xs`` in a physically
    # meaningful way (it just multiplies the inputs).
    atm_high = eqx.tree_at(
        lambda m: m.bulk.rayleigh_xs,
        atm_low,
        replace=jnp.full(atm_low.bulk.rayleigh_xs.shape, 1.0e-2),
    )

    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)

    out_low = float(atm_low.reflected_spectrum(phase, dist, wl)[0, 0])
    out_high = float(atm_high.reflected_spectrum(phase, dist, wl)[0, 0])
    assert out_low != out_high
    assert jnp.isfinite(jnp.asarray(out_high))


def test_clouds_change_the_spectrum():
    """Switching the cloud on changes the spectrum (vs cloud-free).

    The fake RT only models absorption, so it cannot test that clouds
    *brighten* a Lambertian atmosphere -- that's a scattering effect
    only ``ArtReflectPure`` captures. Here we just assert that the
    cloud machinery is wired in: the spectrum visibly changes when the
    cloud opacity is non-negligible.
    """
    K = 1
    atm_no_cloud = make_atmosphere(K=K)  # cloud tau effectively zero
    atm_with_cloud = eqx.tree_at(
        lambda m: (m.clouds.log_pressure_bar, m.clouds.log_opt_depth),
        atm_no_cloud,
        replace=(jnp.full((K,), -0.3), jnp.full((K,), 1.0)),  # tau=10 at 0.5 bar
    )

    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)

    out_no_cloud = float(atm_no_cloud.reflected_spectrum(phase, dist, wl)[0, 0])
    out_with_cloud = float(atm_with_cloud.reflected_spectrum(phase, dist, wl)[0, 0])
    assert out_no_cloud != out_with_cloud
    assert jnp.isfinite(jnp.asarray(out_with_cloud))


def test_mie_cloud_swaps_in_for_gray_cloud():
    """``MieCloud`` is a drop-in replacement for ``GrayCloud``.

    Constructs a MieCloud with synthetic ssa(lambda) and g(lambda)
    (skipping the slow real Mie-grid build) and verifies it produces
    a finite, sensible spectrum when swapped into a base atmosphere.
    """
    K = 1
    atm = make_atmosphere(K=K)
    n_nu = atm.n_nu
    # Synthetic wavelength-dependent ssa/g (real Mie water clouds peak
    # near ssa~1, g~0.85; we use slightly varied values here just to
    # exercise the (n_nu,) broadcast path).
    ssa_grid = jnp.linspace(0.95, 1.0, n_nu)
    g_grid = jnp.linspace(0.7, 0.9, n_nu)
    mie_cloud = MieCloud(
        log_pressure_bar=jnp.full((K,), jnp.log10(0.5)),
        log_opt_depth=jnp.full((K,), jnp.log10(2.0)),
        ssa_grid=ssa_grid,
        g_grid=g_grid,
    )
    atm_mie = eqx.tree_at(lambda m: m.clouds, atm, replace=mie_cloud)

    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)
    out = atm_mie.reflected_spectrum(phase, dist, wl)
    assert out.shape == (K, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_surface_albedo_spectrum_modulates_signal():
    """A wavelength-dependent surface profile maps into the output.

    Two surfaces with the same scalar albedo but opposite spectral
    profiles should produce different reflected-spectrum values at
    the same wavelength.
    """
    K = 1
    n_nu = 100
    atm_flat = make_atmosphere(K=K, n_nu=n_nu)
    # Build a "red surface" spectrum that's bright in the low-wavenumber
    # (long-wavelength) end and dim in the high-wavenumber (short-wl) end.
    red_spectrum = jnp.linspace(2.0, 0.1, n_nu)  # bright at low nu (red)
    blue_spectrum = jnp.linspace(0.1, 2.0, n_nu)  # bright at high nu (blue)

    atm_red = eqx.tree_at(lambda m: m.surface.spectrum, atm_flat, replace=red_spectrum)
    atm_blue = eqx.tree_at(
        lambda m: m.surface.spectrum, atm_flat, replace=blue_spectrum
    )

    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl_red = jnp.array(950.0)  # low wavenumber = long wavelength
    wl_blue = jnp.array(420.0)  # high wavenumber = short wavelength

    red_at_red = float(atm_red.reflected_spectrum(phase, dist, wl_red)[0, 0])
    blue_at_red = float(atm_blue.reflected_spectrum(phase, dist, wl_red)[0, 0])
    red_at_blue = float(atm_red.reflected_spectrum(phase, dist, wl_blue)[0, 0])
    blue_at_blue = float(atm_blue.reflected_spectrum(phase, dist, wl_blue)[0, 0])

    # "Red surface" should be brighter than "blue surface" at red wavelengths,
    # and vice versa.
    assert red_at_red > blue_at_red
    assert blue_at_blue > red_at_blue


def test_n2_mmr_is_clamped_when_others_oversaturate():
    """When tracked mmrs sum to > 1, N2 fraction clamps to zero (no NaN)."""
    K = 1
    atm = make_atmosphere(K=K)
    # Set every tracked species log_mmr to log10(0.5) -- their sum is
    # 2.5, well over 1, so the bulk-gas residual would be negative
    # without clamping. Assumes ConstantMmr profiles.
    atm_saturated = eqx.tree_at(
        lambda m: tuple(s.profile.log_mmr for s in m.species),
        atm,
        replace=tuple(jnp.full((K,), jnp.log10(0.5)) for _ in atm.species),
    )
    phase = jnp.full((K, 1), 0.5)
    dist = jnp.full((K, 1), 1.0)
    wl = jnp.array(550.0)
    out = atm_saturated.reflected_spectrum(phase, dist, wl)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_custom_molecule_set_works_end_to_end():
    """Construct an atmosphere with a non-default molecule mix.

    Builds an ExoJaxAtmosphere with three molecules (H2O, CO, SO2)
    plus an H2 bulk gas instead of the default 5-molecule + N2 setup.
    Verifies that the species tuple is honored throughout (absorption
    iterates the right opa engines; Rayleigh iterates the right
    cross-sections; repr shows the right composition).
    """
    K, n_nu, n_layers = 1, 50, 20
    nu_grid = jnp.linspace(1.0e4, 2.5e4, n_nu)
    rt = _FakeRT(nu_grid, n_layers)
    ray_xs = jnp.full((n_nu,), 1.0e-27)

    species = (
        MolecularSpecies(
            profile=ConstantMmr(log_mmr=jnp.full((K,), -3.0)),
            name="H2O",
            molmass=18.0,
            opa=_FakeOpa(nu_grid, n_layers),
            rayleigh_xs=ray_xs,
        ),
        MolecularSpecies(
            profile=ConstantMmr(log_mmr=jnp.full((K,), -5.0)),
            name="CO",
            molmass=28.0,
            opa=_FakeOpa(nu_grid, n_layers),
            rayleigh_xs=ray_xs,
        ),
        MolecularSpecies(
            profile=ConstantMmr(log_mmr=jnp.full((K,), -7.0)),
            name="SO2",
            molmass=64.0,
            opa=_FakeOpa(nu_grid, n_layers),
            rayleigh_xs=ray_xs,
        ),
    )
    bulk = BulkGasResidual(name="H2", molmass=2.016, rayleigh_xs=ray_xs)

    atm = ExoJaxAtmosphere(
        Rp_Rearth=jnp.ones(K),
        log_gravity_cgs=jnp.full((K,), jnp.log10(981.0)),
        species=species,
        bulk=bulk,
        tp_profile=PowerLawTPProfile(
            T_eq_K=jnp.full((K,), 288.0),
            T_alpha=jnp.full((K,), 0.07),
        ),
        absorption=Absorption(),
        scattering=RayleighScattering(),
        clouds=GrayCloud(
            log_pressure_bar=jnp.full((K,), -10.0),
            log_opt_depth=jnp.full((K,), -10.0),
        ),
        surface=WavelengthDependentSurface(
            log_albedo=jnp.full((K,), jnp.log10(0.3)),
            spectrum=jnp.ones((n_nu,)),
        ),
        rt_engine=rt,
        nu_grid=nu_grid,
        n_nu=n_nu,
    )

    # Shape + finiteness end-to-end.
    out = atm.reflected_spectrum(
        jnp.full((K, 1), 0.5),
        jnp.full((K, 1), 1.0),
        jnp.array(550.0),
    )
    assert out.shape == (K, 1)
    assert bool(jnp.all(jnp.isfinite(out)))

    # Repr reflects the composition.
    s = repr(atm)
    assert "[H2O, CO, SO2]" in s
    assert "bulk=H2" in s


def test_repr_summarizes_atmosphere_state():
    """``__repr__`` is human-readable and includes MMR + VMR side-by-side.

    Avoids the default Equinox PyTree dump (which would print every
    array element) and gives a concise per-planet summary including
    the implicit N2 residual.
    """
    atm = make_atmosphere(K=2)
    s = repr(atm)
    # Header lines.
    assert "ExoJaxAtmosphere(K=2)" in s
    assert "Wavelength:" in s
    assert "Planet 0:" in s
    assert "Planet 1:" in s
    # Per-planet thermodynamic + cloud + surface entries.
    assert "T(P=1bar) = 288.0 K" in s
    assert "Cloud:" in s
    assert "Surface albedo (scalar)" in s
    # Mixing-ratio table with both columns.
    assert "MMR / VMR" in s
    for mol in ("H2O", "CO2", "CH4", "O2", "O3", "N2"):
        assert mol in s


def test_repr_truncates_for_many_planets():
    """For K > 3, only the first 3 planets are shown in full."""
    atm = make_atmosphere(K=10)
    s = repr(atm)
    assert "Planet 0:" in s
    assert "Planet 2:" in s
    assert "Planet 3:" not in s
    assert "... and 7 more planets" in s


def test_atmosphere_init_signature_uses_default_setup():
    """``from_default_setup`` is the documented entry point.

    Verifies it exists and accepts the documented kwargs (without
    actually running it -- the real ExoJAX setup would trigger
    database downloads).
    """
    assert hasattr(ExoJaxAtmosphere, "from_default_setup")
    import inspect

    sig = inspect.signature(ExoJaxAtmosphere.from_default_setup)
    required_kwargs = {
        "Rp_Rearth",
        "log_mmrs",
        "T_eq_K",
        "T_alpha",
        "log_surface_albedo",
        "log_gravity_cgs",
    }
    for name in required_kwargs:
        assert name in sig.parameters, f"missing {name} in from_default_setup"
