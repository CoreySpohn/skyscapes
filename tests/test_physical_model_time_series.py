"""Verification of contrast tables indexed by physical time, not orbital phase."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit

from skyscapes import physical_model
from skyscapes.scene import FlatStar, Planet


def _model(dtype=jnp.float64):
    values = jnp.array([[1.0, 3.0, 15.0], [3.0, 7.0, 19.0]], dtype=dtype)
    return physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=jnp.array([500.0, 700.0], dtype=dtype),
        times_jd=jnp.array([2451545.0, 2451546.0, 2451549.0], dtype=dtype),
        contrast_grid=jnp.stack([values, 4.0 * values]) * 1e-10,
    )


def _evaluate(model, time_jd, wavelength_nm=500.0):
    times = jnp.atleast_1d(time_jd)
    shape = (model.contrast_grid.shape[0], times.size)
    return model.contrast_at_time(
        phase_angle_rad=jnp.full(shape, jnp.pi / 3),
        dist_AU=jnp.ones(shape),
        wavelength_nm=jnp.asarray(wavelength_nm),
        Rp_Rearth=jnp.ones(shape[0]),
        time_jd=times,
    )


@vv.case("planet-phase", "code-verification")
def test_distinct_times_with_identical_phases_recover_table_knots():
    """Equal phases retain distinct contrasts; tolerance is floating-point roundoff."""
    result = _evaluate(_model(), jnp.array([2451545.0, 2451546.0, 2451549.0]))
    expected = np.array([[1.0, 3.0, 15.0], [4.0, 12.0, 60.0]]) * 1e-10
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0)


@vv.case("planet-phase", "code-verification")
def test_interpolates_physical_time_and_wavelength():
    """Uneven epochs have hand-computed bilinear midpoints, at roundoff tolerance."""
    result = _evaluate(_model(), jnp.array([2451545.5, 2451547.5]), 600.0)
    expected = np.array([[3.5, 11.0], [14.0, 44.0]]) * 1e-10
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0)


@vv.case("planet-phase", "code-verification")
def test_jit_promotes_query_without_changing_float32_storage():
    """A mixed-precision query agrees within the float32 table rounding floor."""
    model = _model(jnp.float32)
    time_jd = jnp.array([2451545.5, 2451547.5], dtype=jnp.float64)
    result = eqx.filter_jit(_evaluate)(model, time_jd, jnp.float64(600.0))
    assert model.contrast_grid.dtype == jnp.float32
    assert result.dtype == jnp.float64
    np.testing.assert_allclose(
        result, np.array([[3.5, 11.0], [14.0, 44.0]]) * 1e-10, rtol=2e-7, atol=0
    )


@vv.case("planet-phase", "code-verification")
def test_subday_julian_dates_are_not_rounded_to_float32():
    """Seconds-scale samples at modern JD remain distinct to float64 roundoff."""
    times = jnp.array([2451545.0, 2451545.0 + 100.0 / 86400.0])
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=jnp.array([550.0], dtype=jnp.float32),
        times_jd=times,
        contrast_grid=jnp.array([[[1e-10, 3e-10]]], dtype=jnp.float32),
    )
    result = _evaluate(model, times, 550.0)
    np.testing.assert_allclose(result, [[1e-10, 3e-10]], rtol=2e-7, atol=0)


@vv.case("planet-phase", "code-verification")
def test_single_epoch_is_constant_and_spectral_interpolation_remains_bounded():
    """A snapshot has no invented evolution; spectral midpoints are exact algebra."""
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=[500.0, 700.0],
        times_jd=[2451545.0],
        contrast_grid=[[[1e-10], [3e-10]]],
    )
    result = eqx.filter_jit(_evaluate)(model, jnp.array([2450000.0, 2460000.0]), 600.0)
    np.testing.assert_allclose(result, [[2e-10, 2e-10]], rtol=1e-14, atol=0)
    assert jnp.isnan(_evaluate(model, jnp.array([2451545.0]), 701.0)).all()


def test_time_and_wavelength_support_return_nan_outside_the_record():
    """Unsupported time or spectral queries cannot acquire extrapolated evolution."""
    result = _evaluate(_model(), jnp.array([2451544.0, 2451550.0]))
    assert jnp.isnan(result).all()
    for wavelength_nm in [499.0, 701.0]:
        assert jnp.isnan(
            _evaluate(_model(), jnp.array([2451546.0]), wavelength_nm)
        ).all()


@pytest.mark.parametrize("times_jd", [[2451545.0], [2451545.0, 2451546.0]])
def test_single_wavelength_is_supported_only_at_its_tabulated_wavelength(times_jd):
    """A monochromatic table does not imply a flat spectrum outside its support."""
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=[550.0],
        times_jd=times_jd,
        contrast_grid=np.full((1, 1, len(times_jd)), 1e-10),
    )
    np.testing.assert_allclose(
        _evaluate(model, jnp.array(times_jd), 550.0), 1e-10, rtol=1e-14, atol=0
    )
    assert jnp.isnan(_evaluate(model, jnp.array(times_jd), 551.0)).all()


def test_contrast_without_time_raises_clear_error():
    """The legacy call cannot silently replace time with phase."""
    with pytest.raises(ValueError, match=r"contrast_at_time.*time_jd"):
        _model().contrast(jnp.zeros((2, 1)), jnp.ones((2, 1)), 550.0, jnp.ones(2))


def test_default_time_hook_preserves_existing_model_behavior():
    """A phase-only model inherits the hook without implementing a new method."""
    model = physical_model.LambertianPhysicalModel(Ag=jnp.array([0.3]))
    args = (jnp.array([[0.0, jnp.pi / 2]]), jnp.ones((1, 2)), 550.0, jnp.ones(1))
    np.testing.assert_array_equal(
        model.contrast_at_time(*args, time_jd=jnp.array([2451545.0, 2451546.0])),
        model.contrast(*args),
    )


@vv.case("planet-phase", "code-verification")
@settings(deadline=None, max_examples=30)
@given(step=st.integers(min_value=0, max_value=1024))
def test_linear_interpolation_preserves_nonnegative_contrast(step):
    """Dyadic query coordinates isolate interpolation roundoff from input rounding."""
    fraction = step / 1024
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=[500.0, 700.0],
        times_jd=[0.0, 1.0],
        contrast_grid=[[[0.0, 4e-10], [2e-10, 6e-10]]],
    )
    value = _evaluate(model, jnp.array([fraction]), 500.0 + 200.0 * fraction)
    np.testing.assert_allclose(value, 6e-10 * fraction, rtol=1e-14, atol=0)
    assert jnp.all(value >= 0.0)


@vv.case("planet-phase", "code-verification")
def test_time_derivative_matches_piecewise_linear_slope():
    """Autodiff yields the table segment's known slope to roundoff tolerance."""
    model = _model()
    derivative = jax.grad(lambda time: _evaluate(model, jnp.array([time]))[0, 0])(
        jnp.float64(2451545.5)
    )
    np.testing.assert_allclose(derivative, 2e-10, rtol=1e-14, atol=0)


@vv.case("planet-phase", "code-verification")
def test_planet_contrast_and_magnitude_forward_physical_time():
    """Planet accessors recover tabulated ratios and magnitudes to roundoff."""
    model = physical_model.TimeSeriesPhysicalModel(
        wavelengths_nm=[550.0],
        times_jd=[2451545.0, 2451546.0],
        contrast_grid=[[[1e-10, 1e-8]]],
    )
    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([jnp.pi / 3]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.0]),
        t0_d=jnp.array([2451545.0]),
    )
    planet = Planet(jnp.ones(1), jnp.ones(1), orbit, model)
    star = FlatStar(Ms_kg=1.989e30, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    times = jnp.array([2451545.0, 2451546.0])
    np.testing.assert_allclose(
        planet.contrast(solver, 550.0, times, star=star),
        [[1e-10, 1e-8]],
        rtol=1e-14,
        atol=0,
    )
    _, magnitude = planet.alpha_dMag(solver, times, star=star, wavelength_nm=550.0)
    np.testing.assert_allclose(magnitude, [[25.0, 20.0]], rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("wavelengths_nm", [], "wavelengths_nm"),
        ("wavelengths_nm", [[500.0, 700.0]], "wavelengths_nm"),
        ("wavelengths_nm", [700.0, 500.0], "wavelengths_nm"),
        ("wavelengths_nm", [500.0, 500.0], "wavelengths_nm"),
        ("wavelengths_nm", [500.0, np.nan], "wavelengths_nm"),
        ("times_jd", [], "times_jd"),
        ("times_jd", [[0.0, 1.0]], "times_jd"),
        ("times_jd", [1.0, 0.0], "times_jd"),
        ("times_jd", [0.0, 0.0], "times_jd"),
        ("times_jd", [0.0, np.inf], "times_jd"),
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
        "times_jd": [0.0, 1.0],
        "contrast_grid": np.ones((1, 2, 2)),
    }
    args[field] = value
    with pytest.raises(ValueError, match=message):
        physical_model.TimeSeriesPhysicalModel(**args)
