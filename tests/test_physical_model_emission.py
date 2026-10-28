"""skyscapes.physical_model.ExoJaxEmissionModel -- thermal-emission contracts.

The analytic-limit tests build the real ExoJAX ``ArtEmisPure`` engine on a
mid-infrared grid but need no line lists: zero opacity, an isothermal
atmosphere, and fake cross-section engines pin the radiative-transfer
contract exactly. The final tier needs cached HITRAN line lists for an
Earth twin and skips with a stated reason when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from exojax.database.multimol import database_path_hitran12
from exojax.opacity import OpaPremodit
from hwoutils import constants as const
from scipy.special import expn

from skyscapes.physical_model import ExoJaxEmissionModel
from skyscapes.physical_model.exojax import (
    ConstantMmr,
    EmittingSurface,
    MolecularSpecies,
    PolynomialTPProfile,
    brightness_temperature,
    build_exojax_emission_engines,
    log_gravity_cgs_from_mass_radius,
)

# Constants typed once, independently of hwoutils (parity asserted below).
H_J_S = 6.62607015e-34
C_M_S = 299792458.0
K_B_J_K = 1.380649e-23
REARTH_M = 6.3781e6
PC_M = 3.085677581491367e16
JY_SI = 1.0e-26
ATOMIC_MASS_G = 1.66053906660e-24

# Earth twin: T(P) = sum a_i log10(P[bar])^i with log10(P0) = 0.006.
EARTH_TWIN_COEFFS = (285.218, 146.626, 99.703, 23.120, 1.674)
EARTH_TWIN_LOG_P0 = 0.006
EARTH_TWIN_LOG_MMRS = {
    "H2O": -3.0,
    "CO2": -3.387,
    "O3": -6.523,
    "CH4": -5.770,
    "N2O": -6.495,
}

_DATABASES_DIR = Path.home() / ".cache" / "skyscapes" / "exojax"
_LINE_LIST_MOLECULES = tuple(EARTH_TWIN_LOG_MMRS)


def _line_lists_cached() -> bool:
    """True if every molecule has its parsed (hdf5) or raw (par) HITRAN file."""
    return all(
        (_DATABASES_DIR / database_path_hitran12(m)).exists()
        or (_DATABASES_DIR / m / f"{m}.hdf5").exists()
        for m in _LINE_LIST_MOLECULES
    )


def _blackbody_flux_cgs(nu_cm, temperature_K):
    """``pi B_nu`` [erg/s/cm^2/cm^-1] from the Planck law in cgs primitives."""
    h, c, k = H_J_S * 1e7, C_M_S * 1e2, K_B_J_K * 1e7
    return (
        np.pi
        * 2.0
        * h
        * c**2
        * nu_cm**3
        / np.expm1(h * c * nu_cm / (k * temperature_K))
    )


class _ConstantOpa:
    """Cross-section engine with a given constant ``xsmatrix`` value."""

    def __init__(self, xs):
        self.xs = xs

    def xsmatrix(self, Tarr, pressure):
        return self.xs * jnp.ones((pressure.shape[0], N_NU))


N_NU = 400


# This grid tests even sizes without line opacities, not spectral convergence.
@pytest.mark.filterwarnings("ignore:Resolution may be too small:UserWarning")
def test_emission_odd_grid_size_rounds_up():
    """Premodit grids round odd requests up to the next even point count."""
    engines = build_exojax_emission_engines(molecules=(), n_wavenumbers=101, n_layers=2)
    assert engines["n_nu"] == 102
    assert len(engines["nu_grid"]) == 102


@pytest.fixture(scope="module")
def engines():
    """Real ArtEmisPure engine on a mid-infrared grid; no opacity attached."""
    return build_exojax_emission_engines(
        molecules=(),
        n_wavenumbers=N_NU,
        n_layers=60,
        surface_pressure_bar=10**EARTH_TWIN_LOG_P0,
    )


def _model(engines, coefficients, species=(), **kwargs):
    return ExoJaxEmissionModel.from_engines(
        engines=engines,
        species=species,
        temperature_coefficients=jnp.asarray([coefficients]),
        log_gravity_cgs=jnp.array([2.99]),
        **kwargs,
    )


def _species(xs):
    return (
        MolecularSpecies(
            profile=ConstantMmr(log_mmr=jnp.array([-2.0])),
            name="FAKE",
            molmass=44.0,
            opa=_ConstantOpa(xs),
            rayleigh_xs=jnp.zeros(N_NU),
        ),
    )


def test_constants_parity_with_hwoutils():
    """The independently typed constants agree with hwoutils."""
    assert const.h == H_J_S
    assert const.c == C_M_S
    assert const.k_B == K_B_J_K
    assert const.Rearth2m == REARTH_M
    assert const.pc2m == PC_M
    assert const.Jy == JY_SI


@vv.case("planet-thermal-emission", "code-verification")
def test_polynomial_tp_evaluates_the_given_coefficients():
    """T(P) equals the hand-summed polynomial in log10 P.

    Tolerance basis: floating-point.
    """
    profile = PolynomialTPProfile(coefficients=jnp.asarray([EARTH_TWIN_COEFFS]))
    pressure = np.array([1.0e-4, 1.0e-2, 1.0, 10**EARTH_TWIN_LOG_P0])
    expected = np.array(
        [
            sum(a * np.log10(p) ** i for i, a in enumerate(EARTH_TWIN_COEFFS))
            for p in pressure
        ]
    )
    np.testing.assert_allclose(
        profile.temperature(jnp.asarray(pressure)), expected, rtol=1e-12
    )
    # 1 bar has log10 P = 0, so only a_0 survives.
    np.testing.assert_allclose(
        profile.temperature(jnp.asarray(1.0)), EARTH_TWIN_COEFFS[0], rtol=1e-12
    )


@vv.case("planet-thermal-emission", "code-verification")
def test_earth_twin_profile_spans_216_to_286_kelvin(engines):
    """The Earth-twin coefficients give 216-286 K over the model column.

    Tolerance basis: literature (the quoted range is rounded to 1 K).
    """
    model = _model(engines, EARTH_TWIN_COEFFS)
    layers = np.asarray(model.layer_temperature())
    assert 215.0 < layers.min() < 217.0
    ground = float(model.surface.temperature_K[0])
    assert 285.5 < ground < 286.5
    assert layers.max() < ground


@pytest.mark.parametrize("surface_bar", [1.0, 10**EARTH_TWIN_LOG_P0, 4.0])
def test_pressure_column_ends_at_the_declared_surface(surface_bar):
    """Layer widths cover exactly the declared top-to-surface column."""
    top_bar = 1.0e-4
    built = build_exojax_emission_engines(
        molecules=(), n_wavenumbers=N_NU, surface_pressure_bar=surface_bar
    )
    rt = built["rt_engine"]
    np.testing.assert_allclose(np.sum(rt.dParr), surface_bar - top_bar, rtol=1e-10)
    np.testing.assert_allclose(rt.pressure_boundary[[0, -1]], [top_bar, surface_bar])
    np.testing.assert_allclose(rt.dParr, np.diff(rt.pressure_boundary), rtol=1e-10)
    np.testing.assert_allclose(
        rt.pressure,
        np.sqrt(rt.pressure_boundary[:-1] * rt.pressure_boundary[1:]),
        rtol=1e-10,
    )
    model = _model(built, (280.0, 40.0))
    np.testing.assert_allclose(
        model.surface.temperature_K, [280.0 + 40.0 * np.log10(surface_bar)], rtol=1e-12
    )
    assert rt.pressure[-1] < surface_bar


@vv.case("planet-thermal-emission", "code-verification")
def test_compute_tarr_matches_temperature(engines):
    """The layer temperatures are the profile evaluated at layer pressures."""
    model = _model(engines, EARTH_TWIN_COEFFS)
    np.testing.assert_allclose(
        model.layer_temperature(),
        model.tp_profile.temperature(model.rt_engine.pressure),
        rtol=1e-12,
    )


@vv.case("planet-thermal-emission", "code-verification")
@pytest.mark.parametrize("ground_K", [180.0, 300.0])
def test_zero_opacity_emits_the_ground_blackbody(engines, ground_K):
    """With no gas opacity the planet is a blackbody at the ground temperature.

    The atmosphere is set to 200 K so a solver that leaked atmospheric
    emission or ignored the ground would miss. Tolerance basis:
    exact-algebra (the quadrature integrates 2 mu exactly).
    """
    model = _model(
        engines,
        (200.0, 0.0, 0.0),
        surface_temperature_K=jnp.array([ground_K]),
    )
    nu = np.asarray(model.nu_grid)
    np.testing.assert_allclose(
        model.emergent_flux()[0], _blackbody_flux_cgs(nu, ground_K), rtol=1e-10
    )
    np.testing.assert_allclose(model.brightness_temperature()[0], ground_K, rtol=1e-9)


@vv.case("planet-thermal-emission", "code-verification")
def test_emissivity_scales_the_ground_emission(engines):
    """A gray ground of emissivity e emits e times the blackbody flux."""
    model = _model(
        engines,
        (200.0,),
        surface_temperature_K=jnp.array([290.0]),
        surface_emissivity=jnp.array([0.7]),
    )
    nu = np.asarray(model.nu_grid)
    np.testing.assert_allclose(
        model.emergent_flux()[0], 0.7 * _blackbody_flux_cgs(nu, 290.0), rtol=1e-10
    )


@vv.case("planet-thermal-emission", "code-verification")
@pytest.mark.parametrize("xs", [0.0, 1.0e-30, 1.0e-24, 1.0e-20])
def test_isothermal_atmosphere_is_a_blackbody_at_any_opacity(engines, xs):
    """An isothermal column over a ground at the same temperature is Planck.

    Holds for every optical depth from transparent to thick, so it checks
    the layer weighting rather than any one limit. Tolerance basis:
    exact-algebra (the layer shares sum to one).
    """
    temperature = 250.0
    model = _model(
        engines,
        (temperature,),
        species=_species(xs),
        surface_temperature_K=jnp.array([temperature]),
    )
    nu = np.asarray(model.nu_grid)
    np.testing.assert_allclose(
        model.emergent_flux()[0], _blackbody_flux_cgs(nu, temperature), rtol=1e-9
    )


@vv.case("planet-thermal-emission", "code-verification")
def test_thick_atmosphere_hides_a_hotter_ground(engines):
    """An optically thick isothermal column emits at its own temperature."""
    model = _model(
        engines,
        (250.0,),
        species=_species(1.0e-14),
        surface_temperature_K=jnp.array([320.0]),
    )
    nu = np.asarray(model.nu_grid)
    np.testing.assert_allclose(
        model.emergent_flux()[0], _blackbody_flux_cgs(nu, 250.0), rtol=1e-9
    )


@vv.case("planet-thermal-emission", "code-verification")
@pytest.mark.parametrize("tau", [0.1, 1.0, 3.0])
def test_finite_gray_slab_matches_analytic_angular_integral(engines, tau):
    """A cool slab over hot ground obeys F = Sa + 2 E3(tau) (Sg - Sa).

    Reference: Rybicki & Lightman (1979), Radiative Processes in
    Astrophysics, Sec. 1.4; E3 is defined in NIST DLMF 8.19.2.
    Tolerance basis: four-point Gauss-Legendre hemispheric transmission
    has at most 0.12 percent relative error at these depths; allow 0.2
    percent in the resulting flux. S denotes pi B.
    """
    column_bar = 10**EARTH_TWIN_LOG_P0 - 1.0e-4
    xs = tau * 44.0 * ATOMIC_MASS_G * 10**2.99 / (0.01 * column_bar * 1.0e6)
    model = _model(
        engines,
        (220.0,),
        species=_species(xs),
        surface_temperature_K=jnp.array([320.0]),
    )
    nu = np.asarray(model.nu_grid)
    atmosphere = _blackbody_flux_cgs(nu, 220.0)
    ground = _blackbody_flux_cgs(nu, 320.0)
    expected = atmosphere + 2.0 * expn(3, tau) * (ground - atmosphere)
    np.testing.assert_allclose(model.emergent_flux()[0], expected, rtol=2e-3)


@vv.case("planet-thermal-emission", "code-verification")
@pytest.mark.parametrize("upper_K,lower_K", [(220.0, 300.0), (300.0, 220.0)])
def test_two_temperature_layers_keep_top_to_bottom_order(upper_K, lower_K):
    """Two slabs retain their order in the formal solution (Rybicki & Lightman).

    Tolerance basis: 0.3 percent allows four-point angular quadrature
    error for the E3 transmissions at optical depths 0.3 and 3.3.
    """
    built = build_exojax_emission_engines(
        molecules=(), n_wavenumbers=N_NU, n_layers=2, pressure_top_bar=0.01
    )
    slope = lower_K - upper_K
    xs = 0.3 * 44.0 * ATOMIC_MASS_G * 10**2.99 / (0.01 * 0.09 * 1.0e6)
    model = _model(
        built,
        (upper_K + 1.5 * slope, slope),
        species=_species(xs),
        surface_temperature_K=jnp.array([320.0]),
    )
    nu = np.asarray(model.nu_grid)
    upper = _blackbody_flux_cgs(nu, upper_K)
    lower = _blackbody_flux_cgs(nu, lower_K)
    ground = _blackbody_flux_cgs(nu, 320.0)
    expected = (
        upper
        + 2.0 * expn(3, 0.3) * (lower - upper)
        + 2.0 * expn(3, 3.3) * (ground - lower)
    )
    np.testing.assert_allclose(model.emergent_flux()[0], expected, rtol=3e-3)


@vv.case("planet-thermal-emission", "code-verification")
def test_gray_species_column_opacity_has_independent_cgs_normalization():
    """Hydrostatic tau = sigma q P / (m_mol m_u g), with negligible top P.

    Reference: Pierrehumbert (2010), Principles of Planetary Climate,
    Ch. 4 (hydrostatic pressure-coordinate optical depth).
    Tolerance basis: 3e-8 allows ExoJAX's older atomic-mass constant;
    the omitted 1e-12 bar top contributes only one part in 1e12.
    """
    built = build_exojax_emission_engines(
        molecules=(), n_wavenumbers=N_NU, pressure_top_bar=1.0e-12
    )
    model = _model(built, (250.0,), species=_species(1.0e-24))
    dtau = model.absorption.compute(
        model.species,
        model.layer_temperature(),
        model.rt_engine.pressure,
        10**2.99,
        model.rt_engine,
    ).dtau_total
    expected = 1.0e-24 * 0.01 * 1.0e6 / (44.0 * ATOMIC_MASS_G * 10**2.99)
    np.testing.assert_allclose(np.sum(dtau, axis=0), expected, rtol=3e-8)


@vv.case("planet-thermal-emission", "code-verification")
def test_brightness_temperature_inverts_planck():
    """The brightness temperature of a Planck flux is its temperature."""
    nu = np.linspace(540.0, 2500.0, 50)
    for temperature in (100.0, 288.0, 1500.0):
        flux = _blackbody_flux_cgs(nu, temperature)
        np.testing.assert_allclose(
            brightness_temperature(nu, flux), temperature, rtol=1e-10
        )


@vv.case("planet-thermal-emission", "code-verification")
def test_flux_density_matches_si_planck_at_the_observer(engines):
    """Jy at 10 pc equals the SI Planck law times the solid angle.

    The expected value is built in SI from lambda and temperature; the
    model works in cgs per wavenumber, so this anchors the unit chain
    (per cm^-1 to per Hz, erg to W, Rp/d). Tolerance basis: linear
    interpolation of a smooth Planck curve on the 400-point grid (1e-4).
    """
    ground_K = 288.0
    model = _model(engines, (200.0,), surface_temperature_K=jnp.array([ground_K]))
    wavelength_m = np.array([6.0e-6, 10.0e-6, 15.0e-6])
    flux = model.flux_density_jy(
        wavelength_m * 1e9, jnp.array([1.0]), distance_pc=10.0
    )[:, 0]
    nu_hz = C_M_S / wavelength_m
    b_nu = (
        2.0
        * H_J_S
        * nu_hz**3
        / C_M_S**2
        / np.expm1(H_J_S * nu_hz / (K_B_J_K * ground_K))
    )
    expected = np.pi * b_nu * (REARTH_M / (10.0 * PC_M)) ** 2 / JY_SI
    np.testing.assert_allclose(flux, expected, rtol=1e-4)


def test_flux_interpolation_preserves_nonnegativity_at_an_emissivity_step(engines):
    """Resampling a nonnegative spectrum cannot undershoot an emissivity step."""
    spectrum = jnp.where(jnp.arange(N_NU) < N_NU // 2, 0.0, 1.0)
    model = _model(engines, (288.0,), surface_emissivity_spectrum=spectrum)
    nu = np.asarray(model.nu_grid)
    target_nu = np.linspace(nu[N_NU // 2 - 3], nu[N_NU // 2 + 3], 101)
    flux = np.asarray(model.flux_density_jy(1.0e7 / target_nu, jnp.array([1.0]), 10.0))
    assert np.all(np.isfinite(flux))
    assert np.all(flux >= 0.0)
    assert np.any(flux > 0.0)


def test_flux_interpolation_returns_nan_outside_the_spectral_grid(engines):
    """Requests beyond either grid edge return NaN, including inside JIT."""
    model = _model(engines, (288.0,))
    nu = np.asarray(model.nu_grid)
    wavelength_nm = jnp.asarray(1.0e7 / np.array([nu[0] * 0.99, nu[-1] * 1.01]))
    flux = jax.jit(lambda wl: model.flux_density_jy(wl, jnp.array([1.0]), 10.0))(
        wavelength_nm
    )
    assert np.all(np.isnan(flux))


def test_flux_interpolation_includes_the_declared_wavelength_endpoints(engines):
    """Log-grid roundoff cannot exclude either advertised wavelength limit."""
    model = _model(engines, (288.0,))
    flux = np.asarray(
        model.flux_density_jy(jnp.array([4000.0, 18500.0]), jnp.array([1.0]), 10.0)
    )
    assert np.all(np.isfinite(flux))
    assert np.all(flux > 0.0)


@vv.case("planet-thermal-emission", "code-verification")
def test_flux_scales_as_radius_over_distance_squared(engines):
    """Doubling the radius quadruples the flux; doubling the distance quarters it."""
    model = _model(engines, (230.0,))
    wl = jnp.array([8000.0, 12000.0])
    base = model.flux_density_jy(wl, jnp.array([1.0]), 10.0)
    np.testing.assert_allclose(
        model.flux_density_jy(wl, jnp.array([2.0]), 10.0), 4.0 * base, rtol=1e-12
    )
    np.testing.assert_allclose(
        model.flux_density_jy(wl, jnp.array([1.0]), 20.0), 0.25 * base, rtol=1e-12
    )


@vv.case("planet-thermal-emission", "code-verification")
def test_jit_equals_eager_and_gradient_matches_finite_difference(engines):
    """JIT reproduces eager, and d(flux)/d(ground T) matches a central difference."""
    model = _model(engines, (200.0,), species=_species(1.0e-24))

    def mean_flux(ground_K):
        moved = eqx.tree_at(lambda m: m.surface.temperature_K, model, ground_K)
        return jnp.mean(moved.emergent_flux())

    np.testing.assert_allclose(
        eqx.filter_jit(lambda m: m.emergent_flux())(model),
        model.emergent_flux(),
        rtol=1e-12,
    )
    t0 = jnp.array([280.0])
    step = 1.0e-2
    central = (mean_flux(t0 + step) - mean_flux(t0 - step)) / (2.0 * step)
    np.testing.assert_allclose(jax.grad(mean_flux)(t0)[0], central, rtol=1e-6)


def test_from_engines_constructs_and_emits_inside_jit(engines):
    """Default ground pressure remains static when model assembly is traced."""

    def construct_and_emit(coefficients):
        return _model(engines, coefficients, species=_species(1.0e-24)).emergent_flux()

    coefficients = jnp.array([280.0, 20.0])
    expected = construct_and_emit(coefficients)
    actual = jax.jit(construct_and_emit)(coefficients)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert np.all(np.isfinite(actual))


def test_log_gravity_from_mass_radius_gives_earth_surface_gravity():
    """One Earth mass and radius gives about 9.80 m/s^2."""
    g_cgs = 10.0 ** float(log_gravity_cgs_from_mass_radius(1.0, 1.0))
    assert g_cgs == pytest.approx(980.0, rel=2e-3)


def test_emitting_surface_emissivity_combines_scalar_and_spectrum():
    """The emissivity on the grid is the scalar times the spectral shape."""
    surface = EmittingSurface(
        temperature_K=jnp.array([288.0]),
        emissivity=jnp.array([0.9]),
        spectrum=jnp.array([1.0, 0.5, 0.0]),
    )
    np.testing.assert_allclose(surface.compute_emissivity(0), [0.9, 0.45, 0.0])


# ---------------------------------------------------------------------------
# Earth twin with cached HITRAN line lists
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def earth_twin():
    """Earth-twin model on cached HITRAN lines, or skip when absent."""
    if not _line_lists_cached():
        pytest.skip(
            "data-absent: HITRAN line lists for "
            f"{_LINE_LIST_MOLECULES} not cached in {_DATABASES_DIR}"
        )
    return ExoJaxEmissionModel.from_default_setup(
        log_mmrs={m: jnp.array([v]) for m, v in EARTH_TWIN_LOG_MMRS.items()},
        temperature_coefficients=jnp.array([EARTH_TWIN_COEFFS]),
        log_gravity_cgs=jnp.array([2.9917]),
        surface_pressure_bar=10**EARTH_TWIN_LOG_P0,
    )


def _band(model, lo_um, hi_um):
    um = 1.0e4 / np.asarray(model.nu_grid)
    tb = np.asarray(model.brightness_temperature()[0])
    return tb[(um > lo_um) & (um < hi_um)]


@vv.case("planet-thermal-emission", "code-verification")
def test_earth_twin_window_sees_the_ground(earth_twin):
    """The 11 micron window brightness temperature is near the ground's.

    Tolerance basis: spike (measured 1.3 K below the ground; 4 K allows
    the weak continuum-free line wings).
    """
    wavelengths_um = np.array([6.3, 9.6, 11.0, 15.0])
    temperatures_K = np.interp(
        1.0e4 / wavelengths_um,
        np.asarray(earth_twin.nu_grid),
        np.asarray(earth_twin.brightness_temperature()[0]),
    )
    print(
        "Earth-twin brightness temperatures: "
        + ", ".join(
            f"{wl:g} um = {temperature:.6f} K"
            for wl, temperature in zip(wavelengths_um, temperatures_K, strict=True)
        )
    )
    ground = float(earth_twin.surface.temperature_K[0])
    assert np.median(_band(earth_twin, 10.5, 11.5)) == pytest.approx(ground, abs=4.0)


@vv.case("planet-thermal-emission", "code-verification")
def test_earth_twin_co2_band_probes_the_tropopause(earth_twin):
    """The 15 micron CO2 band minimum sits near the profile's coldest layer.

    Tolerance basis: spike (measured 2.7 K above the minimum; 6 K bound).
    """
    coldest = float(earth_twin.layer_temperature().min())
    band_min = _band(earth_twin, 14.5, 15.5).min()
    assert coldest - 0.5 < band_min < coldest + 6.0


@vv.case("planet-thermal-emission", "code-verification")
def test_earth_twin_ozone_and_water_bands_are_cooler_than_the_window(earth_twin):
    """The 9.6 micron O3 band and 6.3 micron H2O band emit above the ground."""
    window = np.median(_band(earth_twin, 10.5, 11.5))
    assert _band(earth_twin, 9.4, 9.9).min() < window - 25.0
    assert np.median(_band(earth_twin, 6.0, 7.0)) < window - 40.0


def test_ozone_uses_line_lists_in_the_infrared(earth_twin):
    """The thermal builder gives O3 a HITRAN opacity, not the PSG table."""
    o3 = next(s for s in earth_twin.species if s.name == "O3")
    assert isinstance(o3.opa, OpaPremodit)
