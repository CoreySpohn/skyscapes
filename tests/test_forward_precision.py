"""Mixed-precision checks for the public scene readers and their sources."""

import re
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rsun2m
from hwoutils.conversions import nm_to_wavenumber_cm, wavenumber_cm_to_nm
from hwoutils.leaves import canonical
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from hwoutils.testing._numerical import _first_wide, _subjaxprs, _wide_dtype
from jax.extend.source_info_util import summarize
from optixstuff.spectral import Band, Quadrature, Resolution
from zodi import DensityDust

from skyscapes._sample_time import SampleOrigin, with_sample_origin
from skyscapes._tables import ProductTable
from skyscapes.disk import (
    AYOExozodi,
    CompositeDisk,
    ExovistaDisk,
    density_level,
    surface_brightness_level,
)
from skyscapes.physical_model import (
    BlackbodyPhysicalModel,
    ExoJaxPhysicalModel,
    GridPhysicalModel,
    LambertianPhysicalModel,
    PrecomputedPhysicalModel,
    SumPhysicalModel,
    TimeSeriesPhysicalModel,
)
from skyscapes.physical_model.exojax.components import GrayCloud, NullScattering
from skyscapes.physical_model.lambertian import (
    lambert_phase,
    project_reflected,
    reflected_contrast,
)
from skyscapes.population import (
    draw_population,
    draw_populations,
    earth_like,
    mass_to_radius,
    radius_to_mass,
    sky_elements,
)
from skyscapes.scene import (
    AYOZodi,
    BlackbodyStar,
    FlatStar,
    LeinertZodi,
    Planet,
    PrecomputedZodi,
    Scene,
    Star,
)

from .conftest import sky_grid
from .test_disk import profile, solar_star
from .test_population_process import _hosts
from .test_population_process import process as process

pytestmark = pytest.mark.float64_only(reason="trace float32 inputs with x64 enabled")


class _KnownPromotion(AssertionError):
    """An exact upstream source has widened a float32 trace."""


def _wide_equations(jaxpr):
    """Yield the first wide equation and its parent equations, innermost first."""
    for equation in jaxpr.eqns:
        for child in _subjaxprs(equation.params):
            if _first_wide(child) is not None:
                yield from _wide_equations(child)
                yield equation
                return
        if _wide_dtype((*equation.invars, *equation.outvars)) is not None:
            yield equation
            return


def _promotion_site(jaxpr):
    """Resolve anonymous nested conversions at their first attributable call."""
    for equation in _wide_equations(jaxpr):
        source = summarize(equation.source_info)
        if match := re.fullmatch(r"(.+):(\d+):\d+ \((.+)\)", source):
            path, line, function = match.groups()
            return path, int(line), function
        # JAX's source summary hides frames inside its own source tree. The
        # first call into uniform from the Poisson body is still in the traceback.
        if equation.params.get("name") == "_uniform":
            traceback = equation.source_info.traceback
            for frame in () if traceback is None else traceback.frames:
                if frame.function_name != "uniform" and frame.file_name.endswith(
                    "/jax/_src/random/core.py"
                ):
                    return frame.file_name, frame.line_num, frame.function_name
    return None


def _assert_closed(fn, args, promotion=None):
    narrow = canonical(args, precision="float32")
    try:
        assert_precision_closed(fn, *narrow)
    except AssertionError as error:
        if promotion is None:
            raise
        trace, _, _ = eqx.filter_make_jaxpr(fn)(*narrow)
        hit = _first_wide(trace.jaxpr)
        site = _promotion_site(trace.jaxpr)
        path, line, function = promotion
        if (
            hit is None
            or hit[1] != np.float64
            or site is None
            or not site[0].endswith(path)
            or site[1:] != (line, function)
        ):
            raise
        raise _KnownPromotion(
            f"{path}:{line} ({function}) widens to float64"
        ) from error


def _check(fn, *args, n=64, promotion=None):
    precision_pair(fn, args, bound=float(ulps(n, np.float32(1))))
    _assert_closed(fn, args, promotion)


def _closure_cases(cases, promotions):
    """Bind an exact source and a strict xfail to each dependency case."""
    parameters = []
    for case in cases:
        fields = case if isinstance(case, tuple) else (case,)
        promotion = promotions.get(case)
        marks = ()
        if promotion is not None:
            path, line, function = promotion
            marks = pytest.mark.xfail(
                reason=f"{path}:{line} ({function}) widens to float64",
                raises=_KnownPromotion,
                strict=True,
            )
        parameters.append(
            pytest.param(*fields, promotion, id="-".join(fields), marks=marks)
        )
    return parameters


_GRID_EXTRAP = ("/interpax/_spline.py", 1548, "_extrap.<locals>.loclip")
_POISSON = ("/jax/_src/random/core.py", 1933, "_poisson_knuth.<locals>.body_fn")
_REFLECTION_PROFILE = ("/exojax/atm/atmprof.py", 187, "atmprof_powerlow")
_REFLECTION_BACKEND = ("/exojax/rt/reflect.py", 172, "ArtReflectPure.run")


def test_dependency_matcher_rejects_local_widening():
    """A local wide constant still fails and a closed function returns normally."""
    args = (jnp.float32(1),)
    _assert_closed(lambda x: x, args, _POISSON)
    with pytest.raises(AssertionError, match="precision widened to float64") as exc:
        _assert_closed(lambda x: jnp.asarray(np.float64(1)), args, _POISSON)
    assert type(exc.value) is AssertionError


def _scene(star, planet=None, **kwargs):
    return Scene.from_classical(
        star,
        () if planet is None else (Planet(jnp.array([1.0]), planet),),
        star_mass_Msun=1.0,
        planet_mass_Msun=3e-6,
        a_AU=1.2,
        e=0.2,
        i_rad=0.5,
        W_rad=0.7,
        w_rad=0.6,
        M0_rad=0.4,
        dist_pc=10.0,
        ra_deg=75.0,
        dec_deg=15.0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "method,promotion",
    _closure_cases(
        [
            "ephemeris",
            "positions",
            "automatic_positions",
            "position_angles",
            "star_reflex",
            "star_radial_velocity",
            "contrasts",
            "contrast_cube",
            "planet_flux_densities",
            "alpha_dMag",
            "sources",
            "band_contrasts",
            "monochrome_band_contrasts",
            "empty_contrasts",
            "empty_sources",
            "calibrated_sources",
            "sampled_positions",
            "host_sampled_positions",
            "batched_positions",
        ],
        {},
    ),
)
def test_scene_forward_closure(promotion, method):
    """Fails if a public scene reader widens a float32 trace.

    Allow 64 roundings at each output leaf's scale for the Kepler solve,
    rotations, and subsequent flux, geometry or quadrature arithmetic.
    """
    times = jnp.array([0.3, 1.2])
    wave = jnp.asarray(560.0)
    model = TimeSeriesPhysicalModel(
        [500.0, 550.0, 650.0],
        jnp.array([0.0, 2.0]),
        jnp.array([[[1.0, 1.2], [1.4, 1.6], [1.8, 2.0]]]) * 1e-9,
    )
    empty = method.startswith("empty_")
    star = (
        solar_star()
        if method in {"sources", "band_contrasts", "calibrated_sources"}
        else FlatStar(4.0)
    )
    disk = None
    if method == "calibrated_sources":
        disk = profile("grater")
        disk = profile("grater", replace(disk.amplitude, luminosity_rule="bolometric"))
    scene = _scene(
        star,
        None if empty else model,
        disk=disk,
        local_zodi=PrecomputedZodi([500.0, 650.0], jnp.array([2.0, 3.0])),
    )
    if method == "ephemeris":
        _check(Scene.ephemeris, scene, times, promotion=promotion)
        return
    if method == "automatic_positions":
        _check(Scene.positions, scene, times, promotion=promotion)
        return
    if method in {"sampled_positions", "host_sampled_positions", "batched_positions"}:
        if method == "batched_positions":
            scene = _scene(FlatStar(4.0), model)
            scene = Scene.from_scenes((scene, scene))
        else:
            scene = with_sample_origin(scene, SampleOrigin(20_000.25))
            if method == "host_sampled_positions":
                host_times = 20_000.25 + np.asarray(times)
                _check(lambda s: s.positions(host_times), scene, promotion=promotion)
                return
            # Binary-exact offsets isolate evaluation from rounding the input dates.
            times = jnp.array([20_000.5, 20_001.5])
        _check(Scene.positions, scene, times, promotion=promotion)
        return

    # The public reuse path exercises the readers independently of the solver.
    eph = scene.ephemeris(times)
    if method == "monochrome_band_contrasts":
        rule = Quadrature.monochromatic(560.0)
        _check(
            lambda s, t, q, e: s.band_contrasts(
                q, t, resolution=Resolution(), transmission=jnp.ones_like, ephemeris=e
            ),
            scene,
            times,
            rule,
            eph,
        )
    elif method in {"sources", "band_contrasts", "empty_sources", "calibrated_sources"}:
        rule = Quadrature.build((Band(520.0, 580.0),), order=2)
        calibration = tuple(
            Quadrature.build((band,)) for band in scene.calibration_bands()
        )
        resolution = Resolution.build((rule, *calibration), *scene.spectral_tables())
        if method == "band_contrasts":
            _check(
                lambda s, t, q, r, e: s.band_contrasts(q, t, resolution=r, ephemeris=e),
                scene,
                times,
                rule,
                resolution,
                eph,
            )
        else:
            _check(
                lambda s, t, q, r, e: s.sources(q, r, t, ephemeris=e),
                scene,
                times,
                rule,
                resolution,
                eph,
            )
    elif method in {
        "contrasts",
        "empty_contrasts",
        "planet_flux_densities",
        "contrast_cube",
    }:
        reader = getattr(Scene, method.removeprefix("empty_"))
        wavelengths = jnp.array([530.0, 560.0]) if method == "contrast_cube" else wave
        _check(
            lambda s, w, t, e: reader(s, w, t, ephemeris=e),
            scene,
            wavelengths,
            times,
            eph,
        )
    else:
        reader = getattr(Scene, method)
        _check(
            lambda s, t, e: reader(s, t, ephemeris=e),
            scene,
            times,
            eph,
            promotion=promotion,
        )


@pytest.mark.parametrize("use_host_dates", [True, False])
def test_sample_origin_closure(use_host_dates):
    """Fails if first-sample subtraction transfers a wide host date or offset.

    The offset and observation days are binary fractions, so the split and
    subtraction are exact in both precisions.
    """
    origin = SampleOrigin(20_000.25)
    dates = [20_000.5, 20_001.5]
    if use_host_dates:
        _check(lambda clock: clock.relative(dates), origin, n=0)
    else:
        _check(SampleOrigin.relative, origin, jnp.array(dates), n=0)


@pytest.mark.parametrize("reader", ["star", "contrast", "node_star"])
def test_sources_forward_closure(reader):
    """Fails if an evaluated Sources record recreates float64 metadata or flux.

    Four roundings cover the scale and a possible division at a sampled node.
    """
    scene = _scene(
        FlatStar(jnp.asarray(4.0)), LambertianPhysicalModel(jnp.array([0.3]))
    )
    times = jnp.array([0.3])
    rule = Quadrature.build((Band(520.0, 580.0),), order=2)
    sources = scene.sources(rule, Resolution(), times)
    if reader == "node_star":
        _check(lambda src: src.node_star(0, 0), sources, n=4)
    else:
        _check(lambda src: getattr(src, reader), sources, n=4)


@pytest.fixture(scope="module")
def geometry():
    """A well-conditioned illuminated planet and host at two nearby dates."""
    return (
        jnp.array([[0.7, 1.3]]),
        jnp.array([[1.1, 1.2]]),
        jnp.asarray(900.0),
        jnp.array([1.1]),
        jnp.array([0.3, 1.2]),
        BlackbodyStar(5800.0, Rsun2m),
        jnp.asarray(10.0),
    )


@pytest.mark.parametrize(
    "kind,reader,promotion",
    _closure_cases(
        [
            ("lambert", "contrast"),
            ("lambert", "contrast_at_time"),
            ("lambert", "photon_flux_density"),
            ("lambert", "node_photon_flux_density"),
            ("blackbody", "contrast_at_time"),
            ("blackbody", "photon_flux_density"),
            ("blackbody", "node_photon_flux_density"),
            ("cached", "contrast"),
            ("cached", "contrast_cube"),
            ("cached", "node_photon_flux_density"),
            ("grid", "contrast"),
            ("series", "contrast_at_time"),
            ("sum", "contrast_at_time"),
            ("sum", "photon_flux_density"),
            ("sum", "node_photon_flux_density"),
        ],
        {("grid", "contrast"): _GRID_EXTRAP},
    ),
)
def test_physical_model_forward_closure(promotion, kind, reader, geometry):
    """Fails if a physical-model reader widens a float32 trace.

    Allow 64 relative roundings for interpolation, Lambert phase and geometry
    or the Planck exponent sensitivity (hc/lambda/kT < 14 at these nodes).
    """
    phase, distance, wave, radius, time, star, dist = geometry
    lambert = LambertianPhysicalModel(jnp.array([0.3]))
    thermal = BlackbodyPhysicalModel(jnp.array([1200.0]))
    if kind == "cached":
        model = PrecomputedPhysicalModel(
            jnp.array([[0.2, 0.3, 0.4]]), [8000.0, 11000.0, 14000.0], 3
        )
    elif kind == "grid":
        model = GridPhysicalModel(
            [800.0, 1000.0],
            jnp.array([0.0, 90.0, 180.0]),
            jnp.array([[[1.0, 1.2, 1.4], [1.6, 1.8, 2.0]]]) * 1e-9,
            phase_bounds_deg=jnp.array([[5.0, 175.0]]),
        )
    elif kind == "series":
        model = TimeSeriesPhysicalModel(
            [800.0, 1000.0],
            jnp.array([0.0, 2.0]),
            jnp.array([[[1.0, 1.2], [1.6, 1.8]]]) * 1e-9,
        )
    else:
        model = {
            "lambert": lambert,
            "blackbody": thermal,
            "sum": SumPhysicalModel((lambert, thermal)),
        }[kind]
    if reader == "contrast":
        _check(
            type(model).contrast,
            model,
            phase,
            distance,
            wave,
            radius,
            promotion=promotion,
        )
    elif reader == "contrast_cube":
        _check(type(model).contrast_cube, model, phase, distance, wave[None], radius)
    elif reader == "contrast_at_time":
        _check(
            lambda m, p, d, w, r, t, s, x: m.contrast_at_time(
                p, d, w, r, time_d=t, star=s, dist_pc=x
            ),
            model,
            phase,
            distance,
            wave,
            radius,
            time,
            star,
            dist,
        )
    elif reader == "photon_flux_density":
        _check(
            lambda m, p, d, w, r, t, s, x: m.photon_flux_density(
                w,
                t,
                phase_angle_rad=p,
                star_planet_distance_AU=d,
                Rp_Rearth=r,
                star=s,
                dist_pc=x,
            ),
            model,
            phase,
            distance,
            wave,
            radius,
            time,
            star,
            dist,
        )
    else:
        rule = Quadrature.build((Band(880.0, 920.0),), order=2)
        tables = (ProductTable(model, star),) if kind == "cached" else ()
        resolution = Resolution.build(rule, *tables)
        if kind == "cached":
            assert resolution.rule(rule, tables[0]).kind == "fine"
        _check(
            lambda m, p, d, r, t, s, x, q, res: m.node_photon_flux_density(
                q,
                res,
                t,
                phase_angle_rad=p,
                star_planet_distance_AU=d,
                Rp_Rearth=r,
                star=s,
                star_nodes=s.node_flux_density(q, Resolution(), t, dist_pc=x),
                dist_pc=x,
            ),
            model,
            phase,
            distance,
            radius,
            time,
            star,
            dist,
            rule,
            resolution,
        )


@pytest.mark.parametrize("reader", ["phase", "reflected", "project"])
def test_reflection_forward_closure(reader, geometry):
    """Fails if phase or spectral projection widens a float32 trace.

    Allow 16 roundings for the direct phase away from new and for linear
    wavenumber interpolation followed by radius dilution and phase scaling.
    """
    phase, distance, wave, radius, *_ = geometry
    if reader == "phase":
        _check(lambert_phase, phase, n=16)
    elif reader == "reflected":
        _check(
            reflected_contrast, phase, distance, jnp.asarray(0.3), radius[:, None], n=16
        )
    else:
        _check(
            lambda a, nu, p, d, w, r: project_reflected(
                a,
                nu,
                phase_angle_rad=p,
                star_planet_distance_AU=d,
                wavelength_nm=w,
                radius_Rearth=r,
            ),
            jnp.array([[0.2, 0.3, 0.4]]),
            jnp.array([8000.0, 11000.0, 14000.0]),
            phase,
            distance,
            wave[None],
            radius,
            n=16,
        )


@pytest.mark.parametrize(
    "kind,reader",
    [
        ("flat", "photon_flux_density"),
        ("flat", "host_wavelength_flux"),
        ("flat", "integer_wavelength_flux"),
        ("flat", "angular_diameter_arcsec"),
        ("flat", "host_distance_diameter"),
        ("flat", "calibration_flux_density"),
        ("blackbody", "photon_flux_density"),
        ("blackbody", "host_flux"),
        ("blackbody", "angular_diameter_arcsec"),
        ("blackbody", "node_flux_density"),
        ("blackbody", "calibration_flux_density"),
        ("table", "photon_flux_density"),
        ("series", "photon_flux_density"),
        ("series", "host_flux"),
        ("series", "node_flux_density"),
        ("series", "host_node_flux"),
        ("table", "node_flux_density"),
        ("table", "flux_density_jy"),
        ("table", "angular_diameter_arcsec"),
        ("table", "spectral_interpolant"),
    ],
)
def test_star_forward_closure(kind, reader):
    """Fails if a stellar source, its host knots, or band rule widens the trace.

    Allow 32 roundings for tensor interpolation, a Planck exponent of order
    five and the eight-point calibration mean.
    """
    if kind == "series":
        star = Star(
            wavelengths_nm=[500.0, 550.0, 650.0],
            times_d=jnp.array([0.0, 2.0]),
            flux_density_jy=jnp.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]),
            interpolation_method="cubic",
        )
    else:
        star = {
            "flat": FlatStar(4.25),
            "blackbody": BlackbodyStar(5800.0, Rsun2m),
            "table": solar_star(),
        }[kind]
    t, d, w = (jnp.asarray(x) for x in (0.75, 10.0, 560.0))
    band = Band(520.0, 580.0)
    if reader == "flux_density_jy":
        _check(lambda s: s.flux_density_jy, star, n=4)
    elif reader in {"host_wavelength_flux", "integer_wavelength_flux"}:
        wave = 560.0 if reader == "host_wavelength_flux" else 560
        np.testing.assert_array_equal(
            star.photon_flux_density(wave, t, dist_pc=d), 4.25
        )
        _check(
            lambda s, t, d: s.photon_flux_density(wave, t, dist_pc=d), star, t, d, n=0
        )
    elif reader == "host_distance_diameter":
        _check(lambda s: s.angular_diameter_arcsec(10.0), star, n=0)
    elif reader == "angular_diameter_arcsec":
        _check(type(star).angular_diameter_arcsec, star, d, n=32)
    elif reader == "calibration_flux_density":
        _check(
            lambda s, t, d: s.calibration_flux_density(band, t, dist_pc=d),
            star,
            t,
            d,
            n=32,
        )
    elif reader == "host_flux":
        _check(
            lambda s, d: s.photon_flux_density(560.0, 0.75, dist_pc=d), star, d, n=32
        )
    elif reader in {"node_flux_density", "host_node_flux"}:
        rule = Quadrature.build((band,), order=2)
        tables = (star,) if kind in {"table", "series"} else ()
        resolution = Resolution.build(rule, *tables)
        if kind == "series":
            assert resolution.rule(rule, star).kind == "table"
        if reader == "host_node_flux":
            _check(
                lambda s, d, q, r: s.node_flux_density(q, r, [0.75], dist_pc=d),
                star,
                d,
                rule,
                resolution,
                n=32,
            )
            return
        _check(
            lambda s, t, d, q, r: s.node_flux_density(q, r, t, dist_pc=d),
            star,
            t[None],
            d,
            rule,
            resolution,
            n=32,
        )
    elif reader == "spectral_interpolant":
        _check(star.spectral_interpolant(), jnp.array([1.0, 1.5, 2.0]), w, n=32)
    else:
        _check(
            lambda s, w, t, d: s.photon_flux_density(w, t, dist_pc=d),
            star,
            w,
            t,
            d,
            n=32,
        )


@pytest.mark.parametrize(
    "kind,reader,promotion",
    _closure_cases(
        [
            ("ayo", "photon_radiance"),
            ("leinert", "photon_radiance"),
            ("table", "photon_radiance"),
            ("ayo", "node_radiance"),
            ("leinert", "node_radiance"),
            ("table", "integer_wavelength"),
            ("table", "node_radiance"),
            ("table", "spectral_interpolant"),
        ],
        {},
    ),
)
def test_local_zodi_forward_closure(promotion, kind, reader):
    """Fails if sky brightness or its wavelength tables widen a float32 trace.

    Allow 64 roundings for table interpolation and AB brightness conversion,
    including the sightline and color operations where applicable.
    """
    model = {
        "ayo": AYOZodi(),
        "leinert": LeinertZodi(),
        "table": PrecomputedZodi([500.0, 650.0], jnp.array([2.0, 3.0])),
    }[kind]
    frame = _scene(FlatStar(4.0)).sky_frame
    t, w = jnp.asarray(0.75), jnp.asarray(560.0)
    if reader in {"photon_radiance", "integer_wavelength"}:
        if reader == "integer_wavelength":
            w = w.astype(jnp.int32)
        _check(
            type(model).photon_radiance,
            model,
            w,
            frame,
            t,
            promotion=promotion,
        )
    elif reader == "spectral_interpolant":
        _check(model.spectral_interpolant(), jnp.array([1.0, 2.0]), w)
    else:
        rule = Quadrature.build((Band(520.0, 580.0),), order=2)
        tables = (model,) if kind == "table" else ()
        resolution = Resolution.build(rule, *tables)
        _check(
            lambda m, f, t, q, r: m.node_radiance(q, r, f, t),
            model,
            frame,
            t,
            rule,
            resolution,
            promotion=promotion,
        )


@pytest.mark.parametrize(
    "kind,reader",
    [
        ("ayo", "point"),
        ("ayo", "host_float"),
        ("ayo", "host_int"),
        ("cube", "point"),
        ("exovista", "point"),
        ("fixed_radius", "point"),
        ("grater", "point"),
        ("composite", "point"),
        ("cube", "image"),
        ("exovista", "image"),
        ("grater", "scene_point"),
        ("cube", "scene_image"),
        ("empty", "scene_point"),
        ("empty", "scene_image"),
    ],
)
def test_disk_forward_closure(kind, reader):
    """Fails if cube/LOS sampling, sky coordinates or an empty disk widen.

    Allow 128 roundings for coordinate changes, spline and bilinear weights,
    or 16 line-of-sight nodes and the disk's photometric calibration.
    """
    grid = sky_grid(2, 2, 0.12)
    cube = ExovistaDisk(
        grid,
        [500.0, 550.0, 650.0],
        jnp.linspace(1e-10, 2e-10, 12).reshape(3, 2, 2),
    )
    if kind in {"exovista", "grater", "fixed_radius"}:
        disk = profile("exovista" if kind == "fixed_radius" else kind, order=16)
        if kind == "fixed_radius":
            disk = profile(
                "exovista", replace(disk.amplitude, reference_radius_au=1.0), order=16
            )
    else:
        disk = {
            "ayo": AYOExozodi(),
            "cube": cube,
            "empty": None,
            "composite": CompositeDisk((cube, AYOExozodi())),
        }[kind]
    star = FlatStar(jnp.asarray(4.0))
    wave, time, d, inc, pa = (jnp.asarray(x) for x in (560.0, 0.75, 10.0, 25.0, 35.0))
    east, north = jnp.array([0.035, -0.044]), jnp.array([0.04, 0.025])
    if reader == "scene_point":
        _check(
            Scene.photon_radiance_at,
            _scene(star, disk=disk),
            wave,
            time,
            east,
            north,
            n=128,
        )
    elif reader == "scene_image":
        _check(
            lambda s, w, t: s.photon_radiance(w, t, grid=grid).data,
            _scene(star, disk=disk),
            wave,
            time,
            n=128,
        )
    elif reader == "image":
        _check(
            lambda m, w, t, s, d, i, p: (
                m.photon_radiance(
                    w,
                    t,
                    star=s,
                    dist_pc=d,
                    inclination_deg=i,
                    position_angle_deg=p,
                    grid=grid,
                ).data
            ),
            disk,
            wave,
            time,
            star,
            d,
            inc,
            pa,
            n=128,
        )
    elif reader in {"host_float", "host_int"}:
        host_wave = 556.125 if reader == "host_float" else 556
        _check(
            lambda m, e, n, s, d: m.photon_radiance_at(
                host_wave,
                0.75,
                e,
                n,
                star=s,
                dist_pc=d,
                inclination_deg=25.0,
                position_angle_deg=35.0,
            ),
            disk,
            east,
            north,
            star,
            d,
            n=128,
        )
    else:
        _check(
            lambda m, w, t, e, n, s, d, i, p: m.photon_radiance_at(
                w,
                t,
                e,
                n,
                star=s,
                dist_pc=d,
                inclination_deg=i,
                position_angle_deg=p,
            ),
            disk,
            wave,
            time,
            east,
            north,
            star,
            d,
            inc,
            pa,
            n=128,
        )


@pytest.mark.parametrize("reader", ["density", "surface_brightness"])
def test_disk_calibration_forward_closure(reader):
    """Fails if either public amplitude conversion widens a float32 trace.

    Allow 128 roundings for the 16-node LOS calibration, its photometry
    and the division by the unit-calibration density in the inverse.
    """
    disk = profile("exovista", order=16)
    star = FlatStar(4.25)
    dist = jnp.asarray(10.0)
    if reader == "density":
        _check(
            lambda p, s, d: density_level(p.amplitude, p, star=s, dist_pc=d),
            disk,
            star,
            dist,
            n=128,
        )
    else:
        amplitude = DensityDust(
            level=jnp.asarray(0.02), profile_id=disk.profile_id, calibration_id="test"
        )
        _check(
            lambda a, p, s, d: surface_brightness_level(
                a, p, calibration=p.amplitude, star=s, dist_pc=d
            ),
            amplitude,
            disk,
            star,
            dist,
            n=128,
        )


@pytest.mark.parametrize(
    "reader,promotion",
    _closure_cases(
        ["draw_population", "draw_populations", "sample", "mark_sample"],
        {
            "sample": _POISSON,
        },
    ),
)
def test_population_draw_closure(reader, promotion, process):
    """Fails if sampling creates a wide value in a float32 trace.

    The two PRNG widths produce different variates for the same key; the
    precision-pair comparisons use one fixed draw in the evaluation test.
    """
    key = jax.random.key(0)
    hosts = _hosts()
    if reader == "sample":
        fn, args = lambda p, k, h: p.sample(k, h, 1), (process, key, hosts)
    elif reader == "mark_sample":
        host = jax.tree.map(lambda x: x[:1] if jnp.ndim(x) else x, hosts)
        fn, args = lambda p, k, h: p.mark_sample(k, 1, h), (process, key, host)
    else:
        args = (
            key,
            process.occurrence,
            process.types,
            hosts.luminosity_Lsun,
            hosts.mass_Msun,
            process.config,
        )
        fn = (
            draw_population
            if reader == "draw_population"
            else (
                lambda k, o, t, lum, mass, cfg: draw_populations(
                    k, 1, o, t, lum, mass, cfg
                )
            )
        )
    narrow = fn(*canonical(args, precision="float32"))
    for leaf in jax.tree.leaves(narrow):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == np.float32
    _assert_closed(fn, args, promotion)


@pytest.mark.parametrize(
    "reader,promotion",
    _closure_cases(
        [
            "sky_elements",
            "type_albedo",
            "drawn_contrasts",
            "earth_like",
            "mark_read",
            "mark_build",
            "rate",
            "geometric_albedo",
            "mass_to_radius",
            "radius_to_mass",
            "earth_like_box",
        ],
        {},
    ),
)
def test_population_evaluation_closure(promotion, reader, process):
    """Fails if evaluating a fixed draw introduces wide orbital or flux values.

    Allow 256 roundings for the Cartesian/Keplerian round trip at nonzero
    eccentricity, and 64 for geometry, type interpolation and Lambert phase.
    Allow 16 roundings for the mass-radius powers, eight for the albedo
    interpolation and four for the rate sum. The Earth-like flags are exact.
    """
    key = jax.random.key(0)
    hosts = _hosts()
    wave = jnp.asarray(560.0)
    if reader in {"mass_to_radius", "radius_to_mass"}:
        fn = {"mass_to_radius": mass_to_radius, "radius_to_mass": radius_to_mass}[
            reader
        ]
        _check(fn, jnp.array([0.2, 2.6, 9.0]), n=16)
    elif reader == "earth_like_box":
        _check(earth_like, jnp.asarray(1.0), jnp.asarray(1.1), jnp.asarray(1.0), n=0)
    elif reader == "rate":
        _check(type(process).rate, process, hosts, n=4)
    elif reader == "geometric_albedo":
        _check(
            type(process.types).geometric_albedo,
            process.types,
            jnp.arange(3),
            wave,
            n=8,
        )
    elif reader == "sky_elements":
        draw = draw_population(
            key,
            process.occurrence,
            process.types,
            hosts.luminosity_Lsun,
            hosts.mass_Msun,
            process.config,
        )

        def read(population, mass):
            elements = sky_elements(population, mass)
            return jax.tree.map(
                lambda x: jnp.where(population.present, x, jnp.zeros_like(x)), elements
            )

        _check(
            read,
            draw,
            hosts.mass_Msun,
            n=256,
            promotion=promotion,
        )
    elif reader in {"mark_read", "mark_build"}:
        host = jax.tree.map(lambda x: x[:1] if jnp.ndim(x) else x, hosts)
        draw = process.mark_sample(key, 1, host)
        if reader == "mark_read":
            _check(type(process).mark_read, process, draw, n=64, promotion=promotion)
        else:
            marks = process.mark_read(draw)
            _check(
                type(process).mark_build,
                process,
                host,
                *marks,
                n=64,
                promotion=promotion,
            )
    else:
        draw = process.sample(key, hosts, 1)
        if reader == "type_albedo":
            _check(type(process).type_albedo, process, draw, wave)
        elif reader == "drawn_contrasts":
            times = jnp.array([0.3, 1.2])
            _check(
                lambda s, w, t, e: s.contrasts(w, t, ephemeris=e),
                draw,
                wave,
                times,
                draw.ephemeris(times),
            )
        else:
            _check(
                type(process).earth_like, process, draw, hosts, n=0, promotion=promotion
            )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "case",
    [
        "subpixel",
        "cache64",
        "cache32",
        "cache32_array",
        "repeated_knot",
        "density",
        "zodi_integer",
        "star_integer",
        "phase_integer",
    ],
)
def test_conditioned_forward_regressions(case, jit):
    """Compare against host geometry, analytic interpolation, or exact inputs.

    Eight roundings cover linear interpolation and its contrast scaling, and
    32 cover the AYO colour and inverse-square law. The level is passed through
    exactly. The integer-knot example allows interpolation roundoff at 500 nm.
    """

    def evaluate(fn, *args):
        return (eqx.filter_jit(fn) if jit else fn)(*args)

    if case == "subpixel":
        center = 1 + 2**-26
        grid = replace(sky_grid(2, 2, 0.12), center_x=center, center_y=center)
        disk, star, wave, distance = canonical(
            (AYOExozodi(), FlatStar(4.0), jnp.asarray(560.0), jnp.asarray(10.0)),
            precision="float32",
        )
        offset = (1 - center) * grid.step

        def image(m, s, w, d):
            return m.photon_radiance(
                w,
                d,
                star=s,
                dist_pc=d,
                inclination_deg=d,
                position_angle_deg=d,
                grid=grid,
            ).data

        expected = disk.photon_radiance_at(
            wave,
            distance,
            jnp.asarray(offset, dtype=wave.dtype),
            jnp.asarray(offset, dtype=wave.dtype),
            star=star,
            dist_pc=distance,
            inclination_deg=distance,
            position_angle_deg=distance,
        )
        actual = evaluate(image, disk, star, wave, distance)[1, 1]
        assert np.isfinite(actual)
        np.testing.assert_allclose(
            actual, expected, rtol=float(ulps(32, expected) / expected)
        )
        assert_precision_closed(image, disk, star, wave, distance)
    elif case in {"cache64", "cache32", "cache32_array", "repeated_knot"}:
        phase, distance, radius = (
            jnp.array([[0.7]]),
            jnp.array([[1.2]]),
            jnp.array([1.0]),
        )
        spectrum = jnp.array([[0.2, 0.5, 0.8]], dtype=jnp.float32)
        if case == "repeated_knot":
            knots = np.array([17000.0001, 17000.0001, 18000.0002])
            wave, reflectivity = 600.00001, spectrum[0, 1]
        else:
            knots = np.array([17857.142, 17857.152, 18868.0])
            left = 0
            if case == "cache32_array":
                # A narrow cell near the far end of the full spectral range.
                knots = np.array([8000.0, 17857.142, 17857.152])
                left = 1
            wave = float(wavenumber_cm_to_nm((knots[left] + knots[left + 1]) / 2))
            if case == "cache32_array":
                wave = jnp.float32(wave)
            target = nm_to_wavenumber_cm(float(wave))
            fraction = (target - knots[left]) / (knots[left + 1] - knots[left])
            y0, y1 = map(float, spectrum[0, left : left + 2])
            reflectivity = y0 + fraction * (y1 - y0)
        if case in {"cache32", "cache32_array"}:
            phase, distance, radius = canonical(
                (phase, distance, radius), precision="float32"
            )
        model = PrecomputedPhysicalModel(spectrum, knots, len(knots))

        def read(m, p, d, r, w=wave):
            return m.contrast(p, d, w, r)

        args = (model, phase, distance, radius)
        if case == "cache32_array":
            args += (wave,)

        expected = reflected_contrast(
            phase, distance, float(reflectivity) * 2 / 3, radius
        )
        actual = evaluate(read, *args)
        assert actual.dtype == phase.dtype
        np.testing.assert_allclose(
            actual, expected, rtol=float(ulps(8, np.asarray(1, dtype=phase.dtype)))
        )
        if case in {"cache32", "cache32_array"}:
            assert_precision_closed(read, *args)
    elif case == "density":
        disk = canonical(profile("exovista"), precision="float32")
        value = np.float64(1 + 2**-25)
        amplitude = DensityDust(
            level=value, profile_id=disk.profile_id, calibration_id="precision"
        )
        distance = jnp.asarray(10.0, dtype=jnp.float32)
        actual = evaluate(
            lambda a, p, d: density_level(a, p, star=None, dist_pc=d),
            amplitude,
            disk,
            distance,
        )
        assert isinstance(actual, jax.Array)
        assert actual.dtype == value.dtype
        np.testing.assert_array_equal(actual, value)
    elif case in {"zodi_integer", "star_integer"}:
        knots = [500.4, 502.4]
        values = jnp.array([1, 3], dtype=jnp.int32)
        source = (
            PrecomputedZodi(knots, values)
            if case == "zodi_integer"
            else Star(
                wavelengths_nm=knots,
                times_d=[0.0],
                flux_density_jy=values[:, None],
                interpolation_method="linear",
            )
        )
        actual = evaluate(source.spectral_interpolant(), values, jnp.int32(501))
        assert jnp.issubdtype(actual.dtype, jnp.floating)
        np.testing.assert_allclose(
            actual,
            1.6,
            rtol=0,
            atol=float(ulps(4, jnp.asarray(501, dtype=actual.dtype))),
        )
    else:
        model = GridPhysicalModel(
            [500.0, 600.0], jnp.array([0.0, 90.0, 180.0]), jnp.ones((1, 2, 3))
        )
        actual = evaluate(
            GridPhysicalModel.contrast,
            model,
            jnp.zeros((1, 1), dtype=jnp.int8),
            jnp.ones((1, 1)),
            550.0,
            jnp.ones(1),
        )
        np.testing.assert_array_equal(actual, [[1.0]])


def test_phase_only_sum_requires_context(geometry):
    """A phase-only sum still requires the host and epoch adapter."""
    phase, distance, wave, radius, *_ = geometry
    term = LambertianPhysicalModel(jnp.array([0.3]))
    with pytest.raises(ValueError, match="Use contrast_at_time"):
        SumPhysicalModel((term, term)).contrast(phase, distance, wave, radius)


@pytest.mark.parametrize(
    "reader,promotion",
    _closure_cases(
        ["reflectivity", "contrast", "contrast_cube", "two_stream"],
        {
            "reflectivity": _REFLECTION_PROFILE,
            "contrast": _REFLECTION_PROFILE,
            "contrast_cube": _REFLECTION_PROFILE,
            "two_stream": _REFLECTION_BACKEND,
        },
    ),
)
def test_live_reflection_dependency(reader, promotion, tmp_path):
    """Trace the real, line-free transport and its public spectral projection.

    Allow 64 roundings for layers, stream integration and Lambert projection.
    The backend case passes typed optical depths directly into ExoJAX.
    """
    model = ExoJaxPhysicalModel.from_default_setup(
        log_mmrs={},
        T_eq_K=jnp.array([280.0]),
        T_alpha=jnp.array([0.08]),
        log_surface_albedo=jnp.array([-0.5]),
        log_gravity_cgs=jnp.array([2.99]),
        molecules=(),
        bulk_gas=None,
        n_layers=8,
        n_wavenumbers=20,
        pressure_top_bar=0.01,
        databases_dir=str(tmp_path),
    )
    # A finite, partly absorbing cloud keeps the two-stream limit away from
    # pure scattering and gives every layer nonzero optical depth.
    clouds = GrayCloud(
        log_pressure_bar=jnp.array([-1.0]),
        log_opt_depth=jnp.array([0.0]),
        log_sigma=1.0,
        ssa=0.8,
        g=0.5,
    )
    model = eqx.tree_at(
        lambda m: (m.scattering, m.clouds), model, (NullScattering(), clouds)
    )
    if reader == "two_stream":
        shape = (8, 20)
        _check(
            model.rt_engine.run,
            jnp.full(shape, 0.1),
            jnp.full(shape, 0.8),
            jnp.full(shape, 0.5),
            jnp.full(20, 0.3),
            jnp.ones(20),
            promotion=promotion,
        )
    elif reader == "reflectivity":
        _check(ExoJaxPhysicalModel.reflectivity, model, promotion=promotion)
    else:
        wave = jnp.asarray(560.0)
        if reader == "contrast_cube":
            wave = wave[None]
        _check(
            getattr(ExoJaxPhysicalModel, reader),
            model,
            jnp.array([[0.7]]),
            jnp.array([[1.2]]),
            wave,
            jnp.array([1.0]),
            promotion=promotion,
        )


def test_cached_scalar_adapter_shape(geometry):
    """Accept a singleton array and refuse a multi-wavelength scalar read."""
    phase, distance, wave, radius, *_ = geometry
    model = PrecomputedPhysicalModel(jnp.array([[0.2, 0.5]]), [8000.0, 14000.0], 2)
    expected = model.contrast(phase, distance, wave, radius)
    actual = model.contrast(phase, distance, wave.reshape(1, 1), radius)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(TypeError, match="single wavelength"):
        model.contrast(phase, distance, jnp.stack((wave, wave)), radius)
