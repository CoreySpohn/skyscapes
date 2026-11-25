"""Verification of contrast tables indexed by physical time, not orbital phase."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg, Msun2kg
from hwoutils.epoch import Epoch
from hypothesis import given, settings
from hypothesis import strategies as st

from skyscapes import physical_model
from skyscapes.scene import FlatStar, Planet, Scene

from .test_physical_model_emission import _model as emission_model
from .test_physical_model_emission import engines  # noqa: F401
from .test_physical_model_exojax import make_physical_model
from .test_physical_model_grid import _flat_grid


def _model(dtype=None):
    values = jnp.array([[1.0, 3.0, 15.0], [3.0, 7.0, 19.0]], dtype=dtype)
    return physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=jnp.array([500.0, 700.0], dtype=dtype),
        times_d=jnp.array([0.0, 1.0, 4.0], dtype=dtype),
        contrast_grid=jnp.stack([values, 4.0 * values]) * 1e-10,
    )


def _evaluate(model, time_d, wavelength_nm=500.0):
    times = jnp.atleast_1d(time_d)
    shape = (model.contrast_grid.shape[0], times.size)
    return model.contrast_at_time(
        phase_angle_rad=jnp.full(shape, jnp.pi / 3),
        star_planet_distance_AU=jnp.ones(shape),
        wavelength_nm=jnp.asarray(wavelength_nm),
        Rp_Rearth=jnp.ones(shape[0]),
        time_d=times,
    )


@vv.case("planet-phase", "code-verification")
@pytest.mark.float64_only(reason="float64 contrast table reference")
def test_distinct_times_with_identical_phases_recover_table_knots():
    """Equal phases retain distinct contrasts; tolerance is floating-point roundoff."""
    result = _evaluate(_model(), jnp.array([0.0, 1.0, 4.0]))
    expected = np.array([[1.0, 3.0, 15.0], [4.0, 12.0, 60.0]]) * 1e-10
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0)


@vv.case("planet-phase", "code-verification")
@pytest.mark.float64_only(reason="float64 contrast table reference")
def test_interpolates_physical_time_and_wavelength():
    """Uneven epochs have hand-computed bilinear midpoints, at roundoff tolerance."""
    result = _evaluate(_model(), jnp.array([0.5, 2.5]), 600.0)
    expected = np.array([[3.5, 11.0], [14.0, 44.0]]) * 1e-10
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0)


@vv.case("planet-phase", "code-verification")
@pytest.mark.float64_only(reason="float64 query against a float32 contrast table")
def test_jit_promotes_query_without_changing_float32_storage():
    """A mixed-precision query agrees within the float32 table rounding floor."""
    model = _model(jnp.float32)
    time_d = jnp.array([0.5, 2.5], dtype=jnp.float64)
    result = eqx.filter_jit(_evaluate)(model, time_d, jnp.float64(600.0))
    assert model.contrast_grid.dtype == jnp.float32
    assert result.dtype == jnp.float64
    np.testing.assert_allclose(
        result, np.array([[3.5, 11.0], [14.0, 44.0]]) * 1e-10, rtol=2e-7, atol=0
    )


@vv.case("planet-phase", "code-verification")
def test_subday_offsets_remain_distinct():
    """Fails if seconds-scale offsets merge, even with float32 contrast storage."""
    times = jnp.array([0.0, 100.0 / 86400.0])
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=jnp.array([550.0], dtype=jnp.float32),
        times_d=times,
        contrast_grid=jnp.array([[[1e-10, 3e-10]]], dtype=jnp.float32),
    )
    result = _evaluate(model, times, 550.0)
    np.testing.assert_allclose(result, [[1e-10, 3e-10]], rtol=2e-7, atol=0)


@vv.case("planet-phase", "code-verification")
@pytest.mark.float64_only(reason="float64 contrast table reference")
def test_single_epoch_is_constant_and_spectral_interpolation_remains_bounded():
    """A snapshot has no invented evolution; spectral midpoints are exact algebra."""
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=[500.0, 700.0],
        times_d=[0.0],
        contrast_grid=[[[1e-10], [3e-10]]],
    )
    result = eqx.filter_jit(_evaluate)(model, jnp.array([-1545.0, 8455.0]), 600.0)
    np.testing.assert_allclose(result, [[2e-10, 2e-10]], rtol=1e-14, atol=0)
    assert jnp.isnan(_evaluate(model, jnp.array([0.0]), 701.0)).all()


def test_time_and_wavelength_support_return_nan_outside_the_record():
    """Unsupported time or spectral queries cannot acquire extrapolated evolution."""
    result = _evaluate(_model(), jnp.array([-1.0, 5.0]))
    assert jnp.isnan(result).all()
    for wavelength_nm in [499.0, 701.0]:
        assert jnp.isnan(_evaluate(_model(), jnp.array([1.0]), wavelength_nm)).all()


@pytest.mark.parametrize("times_d", [[0.0], [0.0, 1.0]])
def test_single_wavelength_is_supported_only_at_its_tabulated_wavelength(
    times_d, precision
):
    """A monochromatic table does not imply a flat spectrum outside its support."""
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=[550.0],
        times_d=times_d,
        contrast_grid=np.full((1, 1, len(times_d)), 1e-10),
    )
    np.testing.assert_allclose(
        _evaluate(model, jnp.array(times_d), 550.0),
        precision.type(1e-10),
        rtol=1e-14,
        atol=0,
    )
    assert jnp.isnan(_evaluate(model, jnp.array(times_d), 551.0)).all()


def test_contrast_without_time_raises_clear_error():
    """The legacy call cannot silently replace time with phase."""
    with pytest.raises(ValueError, match=r"contrast_at_time.*time_d"):
        _model().contrast(jnp.zeros((2, 1)), jnp.ones((2, 1)), 550.0, jnp.ones(2))


@pytest.mark.parametrize(
    "kind",
    [
        "lambertian",
        "grid",
        "precomputed",
        "exojax",
        "time_series",
        "blackbody",
        "emission",
        "sum",
    ],
)
def test_default_time_hook_preserves_existing_model_behavior(kind, request):
    """Fails if the flux hook rescales contrast or loses the planet/time axes.

    Reflection is bitwise identical; thermal division allows 2e-15 relative
    float64 roundoff. The legacy phase-only hook must stay exact too.
    """
    if kind == "lambertian":
        model = physical_model.LambertianPhysicalModel(Ag=jnp.array([0.3, 0.5]))
    elif kind == "grid":
        wave, phase, grid = _flat_grid()
        model = physical_model.GridPhysicalModel(wave, phase, jnp.stack([grid, grid]))
    elif kind in ("precomputed", "exojax"):
        model = make_physical_model(K=2, n_nu=10, n_layers=2)
        if kind == "precomputed":
            model = physical_model.PrecomputedPhysicalModel.from_physical_model(model)
    elif kind == "time_series":
        model = _model()
    elif kind == "emission":
        model = emission_model(request.getfixturevalue("engines"), (288.0,))
    else:
        model = physical_model.BlackbodyPhysicalModel(
            teff_K=jnp.array([1560.0, 1700.0])
        )
        if kind == "sum":
            model = physical_model.SumPhysicalModel(
                (
                    physical_model.LambertianPhysicalModel(jnp.array([0.3, 0.5])),
                    model,
                )
            )
    count = 1 if kind == "emission" else 2
    args = (
        jnp.array([[0.0, jnp.pi / 2], [0.3, 0.7]])[:count],
        jnp.ones((count, 2)),
        jnp.asarray(10000.0 if kind == "emission" else 550.0),
        jnp.array([1.0, 2.0])[:count],
    )
    times = jnp.array([0.0, 1.0])
    star = FlatStar(3.0)
    contrast = model.contrast_at_time(*args, time_d=times, star=star, dist_pc=10.0)
    flux = model.photon_flux_density(
        args[2],
        phase_angle_rad=args[0],
        star_planet_distance_AU=args[1],
        Rp_Rearth=args[3],
        time_d=times,
        star=star,
        dist_pc=10.0,
    )
    expected = contrast * star.photon_flux_density(args[2], times, dist_pc=10.0)
    assert isinstance(model, physical_model.AbstractPhysicalModel)
    assert flux.shape == (count, 2)
    if kind in ("blackbody", "emission", "sum"):
        np.testing.assert_allclose(flux, expected, rtol=2e-15, atol=0)
    else:
        np.testing.assert_array_equal(flux, expected)
    if kind in ("lambertian", "grid", "precomputed", "exojax"):
        np.testing.assert_array_equal(contrast, model.contrast(*args))


@vv.case("planet-phase", "code-verification")
@settings(deadline=None, max_examples=30)
@given(step=st.integers(min_value=0, max_value=1024))
@pytest.mark.float64_only(reason="float64 contrast table reference")
def test_linear_interpolation_preserves_nonnegative_contrast(step):
    """Dyadic query coordinates isolate interpolation roundoff from input rounding."""
    fraction = step / 1024
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=[500.0, 700.0],
        times_d=[0.0, 1.0],
        contrast_grid=[[[0.0, 4e-10], [2e-10, 6e-10]]],
    )
    value = _evaluate(model, jnp.array([fraction]), 500.0 + 200.0 * fraction)
    np.testing.assert_allclose(value, 6e-10 * fraction, rtol=1e-14, atol=0)
    assert jnp.all(value >= 0.0)


@vv.case("planet-phase", "code-verification")
@pytest.mark.float64_only(reason="float64 contrast table reference")
def test_time_derivative_matches_piecewise_linear_slope():
    """Autodiff yields the table segment's known slope to roundoff tolerance."""
    model = _model()
    derivative = jax.grad(lambda time: _evaluate(model, jnp.array([time]))[0, 0])(
        jnp.float64(0.5)
    )
    np.testing.assert_allclose(derivative, 2e-10, rtol=1e-14, atol=0)


@vv.case("planet-phase", "code-verification")
@pytest.mark.float64_only(reason="float64 contrast table reference")
def test_system_contrast_and_magnitude_forward_physical_time():
    """Scene accessors recover tabulated ratios and magnitudes to roundoff."""
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=[550.0],
        times_d=[0.0, 1.0],
        contrast_grid=[[[1e-10, 1e-8]]],
    )
    planet = Planet(jnp.ones(1), model)
    star = FlatStar(flux_phot_per_nm_m2=1e9)
    system = Scene.from_classical(
        star,
        (planet,),
        star_mass_Msun=1.989e30 / Msun2kg,
        planet_mass_Msun=jnp.array([Mearth2kg / Msun2kg]),
        dist_pc=10.0,
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([jnp.pi / 3]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.0]),
        epoch_d=0.0,
        epoch=Epoch(2451545.0),
    )
    times = jnp.array([0.0, 1.0])
    np.testing.assert_allclose(
        system.contrasts(550.0, times),
        [[1e-10, 1e-8]],
        rtol=1e-14,
        atol=0,
    )
    _, magnitude = system.alpha_dMag(times, wavelength_nm=550.0)
    np.testing.assert_allclose(magnitude, [[25.0, 20.0]], rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("wavelengths_nm", [], "wavelengths_nm"),
        ("wavelengths_nm", [[500.0, 700.0]], "wavelengths_nm"),
        ("wavelengths_nm", [700.0, 500.0], "wavelengths_nm"),
        ("wavelengths_nm", [500.0, 500.0], "wavelengths_nm"),
        ("wavelengths_nm", [500.0, np.nan], "wavelengths_nm"),
        ("times_d", [], "times_d"),
        ("times_d", [[0.0, 1.0]], "times_d"),
        ("times_d", [1.0, 0.0], "times_d"),
        ("times_d", [0.0, 0.0], "times_d"),
        ("times_d", [0.0, np.inf], "times_d"),
        ("contrast_grid", np.ones((2, 2)), "contrast_grid"),
        ("contrast_grid", np.ones((1, 3, 2)), "contrast_grid"),
        ("contrast_grid", np.ones((1, 2, 3)), "contrast_grid"),
        ("contrast_grid", np.full((1, 2, 2), -1.0), "contrast_grid"),
        ("contrast_grid", np.full((1, 2, 2), np.nan), "contrast_grid"),
        ("contrast_grid", np.full((1, 2, 2), np.inf), "contrast_grid"),
    ],
)
def test_invalid_tables_fail_at_construction(field, value, message):
    """Malformed or unphysical tables fail before a traced evaluation."""
    args = {
        "wavelengths_nm": [500.0, 700.0],
        "times_d": [0.0, 1.0],
        "contrast_grid": np.ones((1, 2, 2)),
    }
    args[field] = value
    with pytest.raises(ValueError, match=message):
        physical_model.TimeSeriesPhysicalModel(**args)
