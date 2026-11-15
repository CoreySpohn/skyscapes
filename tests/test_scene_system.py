"""scene.System -- star + tuple of planets + optional disk."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import (
    AU2m,
    G_si,
    Mearth2kg,
    Msun2kg,
    Rearth2m,
    c,
    d2s,
    h,
    k_B,
    pc2AU,
    rad2arcsec,
)
from hwoutils.conversions import photons_per_nm_per_m2_to_jy
from orbix.kepler.markley import solve_trig
from orbix.orbit import KeplerianOrbit

from skyscapes.disk import ExovistaDisk
from skyscapes.physical_model import (
    AbstractPhysicalModel,
    BlackbodyPhysicalModel,
    GridPhysicalModel,
    LambertianPhysicalModel,
    SumPhysicalModel,
)
from skyscapes.scene import BlackbodyStar, FlatStar, Planet, Star, System
from skyscapes.spectral import Bandpass, bin_mean


def _system(star, planets, *, a_AU=None, planet_mass_Msun=None, **kwargs):
    """Circular test scenes with explicit, separately owned dynamics."""
    count = sum(p.n_planets for p in planets)
    return System.from_classical(
        star,
        planets,
        star_mass_Msun=kwargs.pop("star_mass_Msun", 1.0),
        planet_mass_Msun=jnp.full(count, Mearth2kg / Msun2kg)
        if planet_mass_Msun is None
        else planet_mass_Msun,
        dist_pc=kwargs.pop("dist_pc", 10.0),
        a_AU=jnp.ones(count) if a_AU is None else a_AU,
        e=0.0,
        i_rad=jnp.pi / 3,
        W_rad=0.0,
        w_rad=0.0,
        M0_rad=0.0,
        **kwargs,
    )


def _make_planet(Ag: float, Rp_Rearth: float) -> Planet:
    return Planet(jnp.array([Rp_Rearth]), LambertianPhysicalModel(jnp.array([Ag])))


def test_system_basic_fields():
    """System stores star and planets tuple; disk defaults to None."""
    star = FlatStar(flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(0.3, 1.0)
    p2 = _make_planet(0.5, 2.0)
    sys_obj = _system(
        star, (p1, p2), a_AU=jnp.array([1.0, 5.0]), star_mass_Msun=1.989e30 / Msun2kg
    )
    assert sys_obj.n_planets == 2
    assert sys_obj.disk is None


def test_system_with_disk_field():
    """System stores an AbstractDisk when provided."""
    star = FlatStar(flux_phot_per_nm_m2=1e9)
    p = _make_planet(0.3, 1.0)
    wl = jnp.linspace(400.0, 1000.0, 5)
    cube = jnp.full((wl.size, 4, 4), 1e-6)
    disk = ExovistaDisk(pixel_scale_arcsec=0.01, wavelengths_nm=wl, contrast_cube=cube)
    sys_obj = _system(star, (p,), disk=disk, star_mass_Msun=1.989e30 / Msun2kg)
    assert sys_obj.disk is disk


def test_system_positions_stacks_planets():
    """System.positions concatenates per-planet (2, K, T) into (2, K_total, T)."""
    star = FlatStar(flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(0.3, 1.0)
    p2 = _make_planet(0.5, 2.0)
    sys_obj = _system(
        star, (p1, p2), a_AU=jnp.array([1.0, 5.0]), star_mass_Msun=1.989e30 / Msun2kg
    )
    pos = sys_obj.positions(jnp.array([0.0, 100.0]))
    assert pos.shape == (2, 2, 2)


def test_system_contrasts_stacks_planets():
    """System.contrasts returns (K_total, T) per-planet contrast."""
    star = FlatStar(flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(0.3, 1.0)
    p2 = _make_planet(0.5, 2.0)
    sys_obj = _system(
        star, (p1, p2), a_AU=jnp.array([1.0, 5.0]), star_mass_Msun=1.989e30 / Msun2kg
    )
    c = sys_obj.contrasts(jnp.array(600.0), jnp.array([0.0]))
    assert c.shape == (2, 1)
    assert jnp.all(c >= 0.0)


def test_system_spec_flux_density_per_planet():
    """Pin flux dispatch to recorded Jacobi/Markley values at rtol 1e-14.

    Truth is the circular classical chain with current masses and elements;
    this float64 tolerance permits rounding without grid interpolation error.
    """
    star = FlatStar(flux_phot_per_nm_m2=1e9)
    p1 = _make_planet(0.3, 1.0)
    p2 = _make_planet(0.5, 2.0)
    sys_obj = _system(
        star, (p1, p2), a_AU=jnp.array([1.0, 5.0]), star_mass_Msun=1.989e30 / Msun2kg
    )
    times = jnp.array([0.0, 100.0])
    f = sys_obj.planet_flux_densities(jnp.array(600.0), times)
    c = sys_obj.contrasts(jnp.array(600.0), times)
    # Recorded from the circular Jacobi chain with Markley propagation.
    np.testing.assert_allclose(
        c,
        [
            [1.7358140773855917e-10, 4.759592439468622e-10],
            [4.628831980300861e-11, 5.6348425008226415e-11],
        ],
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        f,
        [
            [0.17358140773855918, 0.47595924394686223],
            [0.04628831980300861, 0.05634842500822641],
        ],
        rtol=1e-14,
        atol=0,
    )


def test_bin_planet_flux_narrow_limit():
    """Narrow-bin flux equals contrast times star flux to rtol 1e-10, atol 0.

    A blackbody varies smoothly over 1e-4 nm at 500-700 nm, so the symmetric
    bin's second-order correction is below this float64 relative tolerance.
    Include a zero-width bin and multiple planets and dates to check axes.
    """
    from skyscapes.scene import BlackbodyStar

    star = BlackbodyStar(5772.0, 6.957e8)
    planet = _make_planet(0.3, 1.0)
    system = _system(star, (planet, planet))
    centers = jnp.array([500.0, 700.0])
    widths = jnp.array([1e-4, 0.0])
    times = jnp.array([0.0, 100.0])
    actual, expected = eqx.filter_jit(
        lambda: (
            system.bin_mean_planet_flux_densities(centers, widths, times),
            system.contrast_cube(centers, times)
            * star.spec_flux_density(centers, 0.0, dist_pc=system.dist_pc)[
                :, None, None
            ],
        )
    )()
    assert actual.shape == (2, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("target", ["contrast", "system", "planet_flux"])
@pytest.mark.parametrize("wavelength", [[500.0, 600.0], [[500.0, 600.0]]])
@pytest.mark.parametrize("traced", [False, True])
def test_contrast_rejects_wavelength_arrays(target, wavelength, traced):
    """Fails if flux dispatch bypasses the scalar-wavelength guard."""
    star = FlatStar(1e9)
    planet = _make_planet(0.3, 1.0)
    system = _system(star, (planet,))
    times = jnp.array([0.0, 100.0])

    def evaluate(wavelength_nm):
        if target == "contrast":
            return system.contrasts(
                wavelength_nm, times, ephemeris=system.ephemeris(times)
            )
        if target == "planet_flux":
            return system.planet_flux_densities(wavelength_nm, times)
        return system.contrasts(wavelength_nm, times)

    evaluate = jax.jit(evaluate) if traced else evaluate
    with pytest.raises(ValueError, match=r"one wavelength.*contrast_cube"):
        evaluate(jnp.asarray(wavelength))


@pytest.mark.parametrize("tabulated", [False, True])
def test_bin_contrast_weights_flux_and_has_point_limit(tabulated):
    """Fails if contrast is averaged without photon weighting or loses its point limit.

    For F(lambda)=lambda/500 and C(lambda)=lambda/500, the band ratio
    is (center + width**2/(12*center))/500, independent of epoch.
    """
    wave = jnp.array([400.0, 500.0, 600.0, 800.0])
    if tabulated:
        star = Star(
            wavelengths_nm=wave,
            times_jd=jnp.array([0.0, 200.0]),
            flux_density_jy=photons_per_nm_per_m2_to_jy(wave / 500, wave)[:, None]
            * jnp.array([[1.0, 2.0]]),
            interpolation_method="linear",
        )
    else:
        star = FlatStar(1.0)
    p = _make_planet(0.3, 1.0)
    model = GridPhysicalModel(
        wave, jnp.array([0.0, 180.0]), (wave / 500)[None, :, None] * jnp.ones((1, 1, 2))
    )
    p = eqx.tree_at(lambda p: p.physical_model, p, model)
    system = _system(star, (p, p))
    centers = jnp.array([550.0, 650.0, 575.0])
    widths = jnp.array([200.0, 100.0, 0.0])
    times = jnp.array([0.0, 100.0])
    actual = eqx.filter_jit(system.bin_mean_contrast_cube)(centers, widths, times)
    expected = (centers + (widths**2 / (12 * centers) if tabulated else 0)) / 500
    np.testing.assert_allclose(
        actual, jnp.broadcast_to(expected[:, None, None], (3, 2, 2)), rtol=1e-12
    )
    for width in (1e-5, 1e-14):
        tiny = system.bin_mean_contrast_cube(centers, jnp.full(3, width), times)
        np.testing.assert_allclose(
            tiny, system.contrast_cube(centers, times), rtol=1e-8
        )

    @eqx.filter_jit
    def mean(p):
        return system.bin_mean_contrast_cube(
            jnp.array([550.0 * p[0]]), jnp.array([200.0 * p[1]]), times
        )[0, 0, 0]

    point, steps = jnp.ones(2), jnp.eye(2) * 1e-5
    fd = jax.vmap(lambda d: (mean(point + d) - mean(point - d)) / 2e-5)(steps)
    np.testing.assert_allclose(jax.grad(mean)(point), fd, rtol=1e-6, atol=1e-9)


def test_sum_flux_preserves_components_bins_and_gradients():
    """Fails if summing drops a component, its gradient, or stellar bin knots.

    Float64 quadrature and JIT reassociation allow 1e-12 relative error;
    the central difference at relative step 1e-5 allows 1e-7.
    """
    wave = jnp.array([4000.0, 5000.0, 6000.0])
    times = jnp.array([0.0, 100.0])
    star = Star(
        wavelengths_nm=wave,
        times_jd=times,
        flux_density_jy=photons_per_nm_per_m2_to_jy(1e5 * wave / 5000, wave)[:, None]
        * jnp.array([[1.0, 2.0]]),
        interpolation_method="linear",
    )
    reflected = _make_planet(0.3, 1.0)
    emitted = eqx.tree_at(
        lambda p: p.physical_model,
        reflected,
        BlackbodyPhysicalModel(jnp.array([800.0])),
    )
    combined = eqx.tree_at(
        lambda p: p.physical_model,
        reflected,
        SumPhysicalModel((reflected.physical_model, emitted.physical_model)),
    )
    parts = _system(star, (reflected, emitted))
    system = _system(star, (combined,))
    wavelength = jnp.asarray(5000.0)
    expected = jnp.sum(
        parts.planet_flux_densities(wavelength, times), axis=0, keepdims=True
    )
    actual = system.planet_flux_densities(wavelength, times)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        eqx.filter_jit(system.planet_flux_densities)(wavelength, times),
        actual,
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        system.contrasts(wavelength, times),
        actual / star.spec_flux_density(wavelength, times, dist_pc=system.dist_pc),
        rtol=2e-15,
        atol=0,
    )
    centers, widths = jnp.array([5000.0, 5500.0]), jnp.array([800.0, 0.0])
    expected_bins = jnp.sum(
        parts.bin_mean_planet_flux_densities(centers, widths, times),
        axis=1,
        keepdims=True,
    )
    np.testing.assert_allclose(
        system.bin_mean_planet_flux_densities(centers, widths, times),
        expected_bins,
        rtol=1e-12,
        atol=0,
    )
    direct = bin_mean(
        lambda w: system.planet_flux_densities(w, times),
        centers,
        widths,
        knots_nm=star.quadrature_knots_nm,
    )
    np.testing.assert_allclose(direct, expected_bins, rtol=1e-12, atol=0)

    def log_flux(scales):
        varied = eqx.tree_at(
            lambda s: (
                s.planets[0].physical_model.components[0].Ag,
                s.planets[0].physical_model.components[1].teff_K,
            ),
            system,
            (jnp.array([0.3]) * scales[0], jnp.array([800.0]) * scales[1]),
        )
        return jnp.log(varied.planet_flux_densities(wavelength, times)[0, 0])

    point, steps = jnp.ones(2), jnp.eye(2) * 1e-5
    gradient = jax.grad(log_flux)(point)
    finite_difference = jax.vmap(
        lambda d: (log_flux(point + d) - log_flux(point - d)) / 2e-5
    )(steps)
    assert jnp.all(jnp.abs(gradient) > 0)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-7, atol=0)
    dark = FlatStar(0.0)
    np.testing.assert_allclose(
        eqx.tree_at(lambda s: s.star, system, dark).planet_flux_densities(
            wavelength, times
        ),
        emitted.physical_model.spec_flux_density(
            wavelength, emitted.Rp_Rearth, dist_pc=system.dist_pc
        )[:, None]
        * jnp.ones((1, 2)),
        rtol=1e-12,
        atol=0,
    )


def test_sum_requires_a_component_and_stellar_context():
    """Fails if an empty sum constructs or contrast silently lacks a host star."""
    with pytest.raises(ValueError, match="component"):
        SumPhysicalModel(())
    model = SumPhysicalModel((LambertianPhysicalModel(jnp.array([0.3])),))
    args = (jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.asarray(500.0), jnp.ones(1))
    with pytest.raises(ValueError, match="star"):
        model.contrast(*args)
    with pytest.raises(ValueError, match="star"):
        model.contrast_at_time(*args, time_jd=jnp.zeros(1))


def test_planet_flux_preserves_keyword_only_model_overrides():
    """Fails if the default flux hook breaks existing keyword-only time hooks."""

    class KeywordModel(AbstractPhysicalModel):
        def contrast(self, phase_angle_rad, dist_AU, wavelength_nm, Rp_Rearth):
            raise AssertionError("The time-aware override must be used")

        def contrast_at_time(
            self,
            *,
            phase_angle_rad,
            dist_AU,
            wavelength_nm,
            Rp_Rearth,
            time_jd,
            star=None,
            dist_pc=None,
        ):
            return jnp.ones_like(phase_angle_rad) * (time_jd + 1) * 1e-10

    planet = eqx.tree_at(
        lambda p: p.physical_model, _make_planet(0.3, 1.0), KeywordModel()
    )
    star = FlatStar(3.0)
    actual = _system(star, (planet,)).planet_flux_densities(
        jnp.asarray(500.0), jnp.array([0.0, 1.0])
    )
    np.testing.assert_allclose(actual, [[3e-10, 6e-10]], rtol=1e-15, atol=0)


@pytest.mark.parametrize("mass", [1.0, 1000.0])
def test_stellar_observables_scale_sign_and_sum(mass):
    """Fails on stellar sign, mass-ratio, period, or barycentric balance errors.

    Truth is two-body barycentric balance and the circular Kepler RV
    amplitude derived from G, masses, a and inclination; multiple planets
    obey mass-weighted barycentric balance instead of independent summation.
    Float64 algebra
    permits rtol 3e-14, atol 0; sign assertions have no tolerance.
    """
    star = FlatStar(1e9)
    planet = _make_planet(0.3, 1.0)
    mp = mass * Mearth2kg
    period_s = 2 * np.pi * np.sqrt(AU2m**3 / (G_si * (Msun2kg + mp)))
    times = jnp.asarray([0.0, period_s / d2s / 2])
    single = _system(
        star,
        (planet,),
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.array([mass * Mearth2kg / Msun2kg]),
    )
    reflex = single.star_reflex(times)
    np.testing.assert_allclose(
        reflex,
        -mp / (Msun2kg + mp) * single.positions(times)[:, 0],
        rtol=3e-14,
        atol=0,
    )
    assert jnp.all(jnp.sum(reflex * single.positions(times)[:, 0], axis=0) < 0)
    amplitude = (2 * np.pi * G_si / period_s) ** (1 / 3) * mp * np.sin(np.pi / 3)
    amplitude /= (Msun2kg + mp) ** (2 / 3)
    rv = single.star_radial_velocity(times)
    np.testing.assert_allclose(rv, [-amplitude, amplitude], rtol=3e-14, atol=0)
    velocity = _reference_orbit(single).velocity_AU_d(
        solve_trig, times, Ms_kg=Msun2kg, Mp_kg=mp
    )
    assert rv[1] > 0 > velocity[0, 2, 1]
    other = _make_planet(0.5, 2.0)
    pair = _system(
        star,
        (planet, other),
        star_mass_Msun=1.0,
        a_AU=jnp.array([1.0, 5.0]),
        planet_mass_Msun=jnp.array([mass, 1.0]) * Mearth2kg / Msun2kg,
    )
    eph = pair.ephemeris(times)
    mass_ratios = jnp.array([mp, Mearth2kg]) / Msun2kg
    weighted_position = jnp.sum(
        mass_ratios[:, None, None] * eph.position_AU[1:], axis=0
    )
    weighted_velocity = jnp.sum(
        mass_ratios[:, None, None] * eph.velocity_AU_per_d[1:], axis=0
    )
    np.testing.assert_allclose(
        pair.star_reflex(times),
        -weighted_position[:, :2].T * (rad2arcsec / (10.0 * pc2AU)),
        rtol=3e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        pair.star_radial_velocity(times),
        -weighted_velocity[:, 2] * (AU2m / d2s),
        rtol=3e-14,
        atol=0,
    )
    batched = System(
        star,
        (jax.tree.map(lambda a, b: jnp.concatenate((a, b)), planet, other),),
        pair.dynamics,
    )
    for name in ("star_reflex", "star_radial_velocity", "position_angles"):
        np.testing.assert_allclose(
            getattr(batched, name)(times),
            getattr(pair, name)(times),
            rtol=3e-14,
            atol=0,
        )


@pytest.mark.parametrize("tabulated", [False, True])
def test_band_contrast_tophat_and_narrow_limits(tabulated):
    """Fails if a band loses stellar knots, photon weighting, or its point limit.

    Truth is the existing top-hat quadrature (rtol 3e-14 for float64), and
    monochromatic contrast for a 1e-4 nm band (rtol 1e-10 for curvature).
    """
    wave = jnp.array([4000.0, 4700.0, 4900.0, 6000.0])
    star = BlackbodyStar(5772.0, 6.957e8)
    if tabulated:
        star = Star(
            wavelengths_nm=wave,
            times_jd=jnp.array([0.0, 100.0]),
            flux_density_jy=photons_per_nm_per_m2_to_jy(
                jnp.array([1.0, 8.0, 2.0, 5.0]), wave
            )[:, None]
            * jnp.array([[1.0, 2.0]]),
            interpolation_method="linear",
        )
    planet = eqx.tree_at(
        lambda p: p.physical_model,
        _make_planet(0.3, 1.0),
        BlackbodyPhysicalModel(jnp.array([800.0])),
    )
    other = eqx.tree_at(lambda p: p.physical_model.teff_K, planet, jnp.array([1100.0]))
    system = _system(star, (planet, other))
    times = jnp.array([0.0, 100.0])
    band = Bandpass.tophat(5000.0, 1600.0)
    actual = eqx.filter_jit(lambda: system.band_contrasts(band, times))()
    expected = system.bin_mean_contrast_cube(
        jnp.array([5000.0]), jnp.array([1600.0]), times
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    batched = System(
        star,
        (jax.tree.map(lambda a, b: jnp.concatenate((a, b)), planet, other),),
        system.dynamics,
    )
    np.testing.assert_allclose(
        batched.band_contrasts(band, times), actual, rtol=3e-14, atol=0
    )
    np.testing.assert_allclose(
        system.band_contrasts(Bandpass.tophat(5000.0, 1e-4), times),
        system.contrasts(jnp.asarray(5000.0), times),
        rtol=1e-10,
        atol=0,
    )


def test_band_contrast_triangle_and_temperature_gradient():
    """Fails on omitted transmission knots, energy weighting, or broken gradients.

    Truth is NumPy Planck photon spectra with 20001-point trapezoids, rtol
    2e-8 (second-order mesh error plus order-four quadrature on each side).
    The temperature derivative uses central differences at 0.008 K; rtol
    1e-8 allows second-order truncation and float64 cancellation.
    """
    star = BlackbodyStar(5772.0, 6.957e8)
    planet = eqx.tree_at(
        lambda p: p.physical_model,
        _make_planet(0.3, 1.0),
        BlackbodyPhysicalModel(jnp.array([800.0])),
    )
    system = _system(star, (planet,))
    band = Bandpass(jnp.array([4000.0, 4700.0, 6000.0]), jnp.array([0.0, 1.0, 0.0]))
    times = jnp.zeros(1)
    wave = np.linspace(4000.0, 6000.0, 20001)
    response = np.interp(wave, [4000.0, 4700.0, 6000.0], [0.0, 1.0, 0.0])

    def photons(teff, radius):
        lam = wave * 1e-9
        return radius**2 / (lam**4 * np.expm1(h * c / (lam * k_B * teff)))

    reference = np.trapezoid(response * photons(800.0, Rearth2m), wave)
    reference /= np.trapezoid(response * photons(5772.0, star.radius_m), wave)

    @jax.jit
    def contrast(teff):
        varied = eqx.tree_at(
            lambda s: s.planets[0].physical_model.teff_K, system, jnp.array([teff])
        )
        return varied.band_contrasts(band, times)[0, 0]

    np.testing.assert_allclose(contrast(800.0), reference, rtol=2e-8, atol=0)
    fd = (contrast(800.008) - contrast(799.992)) / 0.016
    np.testing.assert_allclose(jax.grad(contrast)(800.0), fd, rtol=1e-8, atol=0)


def _ephemeris_scene(e=0.05, inc=0.7, masses=(0.001, 0.005), reverse=False):
    """Nondegenerate elements with distinct size, phase and physical parameters."""
    n = len(masses)
    indices = list(range(n))[::-1] if reverse else list(range(n))
    planets = tuple(_make_planet(0.3 + i * 0.2, 1.0 + i) for i in indices)
    return System.from_classical(
        FlatStar(1e9),
        planets,
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.asarray(masses)[jnp.array(indices)],
        dist_pc=12.0,
        a_AU=jnp.array([1.0, 5.0])[:n][jnp.array(indices)],
        e=e,
        i_rad=inc,
        W_rad=0.4,
        w_rad=jnp.array([0.7, 1.1])[:n][jnp.array(indices)],
        M0_rad=jnp.array([0.2, 1.7])[:n][jnp.array(indices)],
        epoch_d=jnp.array([23.0, 34.0])[:n][jnp.array(indices)],
    )


def _reference_orbit(system, index=0):
    """Independent two-body orbit, read in scene order for comparisons."""
    i = system.body_index[index]
    elements = {
        key: value[i : i + 1]
        for key, value in system.dynamics.orbits.to_classical().items()
    }
    return KeplerianOrbit(**elements, t0_d=system.dynamics.orbits.epoch_d[i : i + 1])


def _scene_observables(system, times, **kwargs):
    """Read every geometry consumer with fixed, small spectral quadrature."""
    waves, widths = jnp.array([500.0, 700.0]), jnp.array([40.0, 0.0])
    return dict(
        positions=system.positions(times, **kwargs),
        position_angles=system.position_angles(times, **kwargs),
        star_reflex=system.star_reflex(times, **kwargs),
        star_radial_velocity=system.star_radial_velocity(times, **kwargs),
        contrasts=system.contrasts(waves[0], times, **kwargs),
        contrast_cube=system.contrast_cube(waves, times, **kwargs),
        bin_mean_contrast_cube=system.bin_mean_contrast_cube(
            waves, widths, times, n_nodes=2, **kwargs
        ),
        band_contrasts=system.band_contrasts(
            Bandpass.tophat(600.0, 80.0), times, n_nodes=2, **kwargs
        ),
        bin_mean_planet_flux_densities=system.bin_mean_planet_flux_densities(
            waves, widths, times, n_nodes=2, **kwargs
        ),
        planet_flux_densities=system.planet_flux_densities(waves[0], times, **kwargs),
        alpha_dMag=jnp.stack(system.alpha_dMag(times, **kwargs)),
    )


def _independent_observables(system, times, index=0):
    """Direct KeplerianOrbit/Markley truth with independent two-body masses."""
    planet, star = system.planets[index], system.star
    orbit = _reference_orbit(system, index)
    ms, mp = system.star_mass_Msun * Msun2kg, system.planet_mass_Msun[index] * Msun2kg
    r, phase, distance = orbit.propagate(solve_trig, times, Ms_kg=ms + mp)
    geometry = dict(
        phase_angle_rad=phase,
        dist_AU=distance,
        Rp_Rearth=planet.Rp_Rearth,
        time_jd=times,
        star=star,
        dist_pc=system.dist_pc,
    )
    contrast = planet.physical_model.contrast_at_time(
        wavelength_nm=jnp.asarray(500.0), **geometry
    )
    flux = planet.physical_model.spec_flux_density(jnp.asarray(500.0), **geometry)
    positions = r[:, :2, :].transpose(1, 0, 2) * (rad2arcsec / (system.dist_pc * pc2AU))
    alpha = orbit.separation_arcsec(
        solve_trig, times, Ms_kg=ms + mp, dist_pc=system.dist_pc
    )
    return dict(
        positions=positions,
        position_angles=orbit.position_angle_deg(solve_trig, times, Ms_kg=ms, Mp_kg=mp),
        star_reflex=jnp.stack(
            orbit.reflex_position_arcsec(
                solve_trig, times, Ms_kg=ms, Mp_kg=mp, dist_pc=system.dist_pc
            )
        )[:, 0],
        star_radial_velocity=orbit.radial_velocity_m_s(
            solve_trig, times, Ms_kg=ms, Mp_kg=mp
        )[0],
        contrasts=contrast,
        contrast_cube=jnp.stack([contrast, contrast]),
        bin_mean_contrast_cube=jnp.stack([contrast, contrast]),
        band_contrasts=contrast,
        bin_mean_planet_flux_densities=jnp.stack([flux, flux]),
        planet_flux_densities=flux,
        alpha_dMag=jnp.stack(
            [alpha, -2.5 * jnp.log10(contrast + jnp.finfo(contrast.dtype).tiny)]
        ),
    )


@pytest.mark.parametrize("e,inc", [(1e-6, 0.7), (0.6, 1.1), (0.2, 2.6)])
def test_ephemeris_single_planet_conventions(e, inc):
    """Catch epoch, sky, mass, phase and RV-sign changes against KeplerianOrbit/Markley.

    Truth is KeplerianOrbit's independent two-body methods using solve_trig (Markley
    1995). rtol 3e-13 plus 3e-13 absolute allows float64 basis/epoch roundoff;
    contrast and flux use relative tolerance only to protect tiny signals.
    """
    system = _ephemeris_scene(e, inc, masses=(0.01,))
    times = jnp.array([-100.0, 23.0, 200.0, 700.0])
    eph = system.ephemeris(times)
    actual = _scene_observables(system, times, ephemeris=eph)
    expected = _independent_observables(system, times)
    for name, want in expected.items():
        absolute = 0 if "contrast" in name or "flux" in name else 3e-13
        np.testing.assert_allclose(
            actual[name], want, rtol=3e-13, atol=absolute, err_msg=name
        )


@pytest.mark.parametrize("batched", [False, True])
def test_ephemeris_scene_order_and_reuse(batched, monkeypatch):
    """Catch sorting or reuse regressions against a permuted identical scene.

    Truth is the same inner-to-outer Jacobi chain in both scenes; require
    exact arrays for cached/default paths and reordered arrays (no roundoff
    changes are needed). Both tuple modules and K=2 modules are exercised.
    """
    from orbix.kepler import Halley

    system = _ephemeris_scene()
    reverse = _ephemeris_scene(reverse=True)
    assert reverse.body_index == (1, 0)
    np.testing.assert_array_equal(reverse.dynamics.orbits.a_AU, [1.0, 5.0])
    if batched:
        reverse = eqx.tree_at(
            lambda s: s.planets,
            reverse,
            (jax.tree.map(lambda a, b: jnp.concatenate((a, b)), *reverse.planets),),
        )
    times = jnp.array([-100.0, 23.0, 200.0, 700.0])
    defaults = _scene_observables(system, times)
    eph, reversed_eph = system.ephemeris(times), reverse.ephemeris(times)
    np.testing.assert_allclose(
        system.ephemeris(times, solver=Halley()).position_AU,
        eph.position_AU,
        rtol=3e-14,
        atol=3e-14,
    )

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("A supplied ephemeris must never solve again")

    monkeypatch.setattr(System, "ephemeris", unexpected_solve)
    shared = _scene_observables(system, times, ephemeris=eph)
    reordered = _scene_observables(reverse, times, ephemeris=reversed_eph)
    for name, want in defaults.items():
        np.testing.assert_array_equal(shared[name], want, err_msg=name)
        axis = 1 if name in ("positions", "alpha_dMag") or want.ndim == 3 else 0
        want = want if name.startswith("star_") else jnp.flip(want, axis=axis)
        np.testing.assert_array_equal(reordered[name], want, err_msg=name)


@pytest.mark.parametrize("massless", [False, True])
def test_ephemeris_jacobi_balance_and_independent_limit(massless):
    """Catch independent reflex summation or incorrect zero-mass handling.

    Truth: sum(m*r)=sum(m*v)=0 (Jacobi barycentric balance), with atol
    4e-17 Msun AU and 4e-19 Msun AU/day for x64 cancellation. For masses
    .001/.005 Msun at 1/5 AU, the relative maximum stellar-offset change
    from independent KeplerianOrbit/Markley orbits must be nonzero and below .0033
    (over three times the measured .00106742). Massless values use 3e-13
    relative/absolute float64 tolerance, with relative-only photometry.
    """
    masses = (0.0, 0.0) if massless else (0.001, 0.005)
    system = _ephemeris_scene(masses=masses)
    if massless:
        system = _ephemeris_scene(masses=(jnp.nan, -1.0))
    times = jnp.array([-100.0, 23.0, 200.0, 700.0])
    eph = system.ephemeris(times)
    weights = jnp.array([1.0, *masses])
    np.testing.assert_allclose(
        jnp.einsum("k,ktc->tc", weights, eph.position_AU), 0, atol=4e-17
    )
    np.testing.assert_allclose(
        jnp.einsum("k,ktc->tc", weights, eph.velocity_AU_per_d), 0, atol=4e-19
    )
    independent = [
        _independent_observables(system, times, i) for i in range(len(system.planets))
    ]
    if massless:
        actual = _scene_observables(system, times, ephemeris=eph)
        for name, got in actual.items():
            if name.startswith("star_"):
                want = sum(item[name] for item in independent)
            else:
                axis = 1 if name in ("positions", "alpha_dMag") or got.ndim == 3 else 0
                want = jnp.concatenate([item[name] for item in independent], axis=axis)
            absolute = 0 if "contrast" in name or "flux" in name else 3e-13
            np.testing.assert_allclose(
                got, want, rtol=3e-13, atol=absolute, err_msg=name
            )
    else:
        control = sum(item["star_reflex"] for item in independent)
        offsets = system.star_reflex(times, ephemeris=eph)
        relative = jnp.max(jnp.linalg.norm(offsets - control, axis=0)) / jnp.max(
            jnp.linalg.norm(control, axis=0)
        )
        assert 0 < relative < 0.0033


@pytest.mark.parametrize("field", ["star_mass", "planet_mass", "distance", "a"])
def test_ephemeris_shared_jit_gradient(field):
    """Catch host conversion, stale elements, or missing shared-state derivatives.

    Truth is central differences of a position at a +/-1e-5 relative step; rtol
    1e-8, atol 1e-10 covers second-order truncation and x64 cancellation.
    Shared JIT positions and angles agree with eager values to 3e-13.
    """
    system = _ephemeris_scene()
    times = jnp.array([-100.0, 23.0, 200.0, 700.0])

    @jax.jit
    def read(a):
        targets = {
            "star_mass": lambda d: d.star.mass_Msun,
            "planet_mass": lambda d: d.planets.mass_Msun,
            "distance": lambda d: d.frame.dist_pc,
            "a": lambda d: d.orbits.a_AU,
        }
        target = targets[field]
        original = target(system.dynamics)
        replacement = (
            original * a if original.ndim == 0 else original.at[0].set(original[0] * a)
        )
        dynamics = eqx.tree_at(target, system.dynamics, replacement)
        varied = eqx.tree_at(lambda s: s.dynamics, system, dynamics)
        eph = varied.ephemeris(times)
        return varied.positions(times, ephemeris=eph), varied.position_angles(
            times, ephemeris=eph
        )

    position, angle = read(1.0)
    np.testing.assert_allclose(
        position, system.positions(times), rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        angle, system.position_angles(times), rtol=3e-13, atol=3e-13
    )
    derivative = jax.jit(jax.grad(lambda a: read(a)[0][0, 1, 2]))(1.0)
    finite_difference = (read(1.00001)[0][0, 1, 2] - read(0.99999)[0][0, 1, 2]) / 2e-5
    assert derivative != 0
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-8, atol=1e-10)


def test_classical_builder_traced_order_and_ownership():
    """Fails if host sorting traces, explicit hierarchy fails, or owners duplicate."""

    def build(a, order=None, mass=1.0):
        return System.from_classical(
            FlatStar(1e9),
            (Planet(jnp.ones(2), LambertianPhysicalModel(jnp.array([0.3, 0.5]))),),
            star_mass_Msun=mass,
            planet_mass_Msun=jnp.array([0.002, 0.001]),
            dist_pc=10.0,
            a_AU=a,
            e=0.1,
            i_rad=0.7,
            W_rad=0.4,
            w_rad=0.2,
            M0_rad=0.3,
            order=order,
        )

    with pytest.raises(ValueError, match="order"):
        eqx.filter_jit(build)(jnp.array([5.0, 1.0]))
    scene = eqx.filter_jit(lambda a: build(a, (1, 0)))(jnp.array([5.0, 1.0]))
    concrete = eqx.filter_jit(lambda mass: build([5.0, 1.0], mass=mass))(
        jnp.asarray(1.0)
    )
    assert concrete.body_index == (1, 0)
    assert scene.body_index == (1, 0)
    np.testing.assert_array_equal(scene.dynamics.orbits.a_AU, [1.0, 5.0])
    np.testing.assert_array_equal(scene.planet_mass_Msun, [0.002, 0.001])
    paths = [
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(scene)[0]
    ]
    mass_distance_paths = [
        path for path in paths if "mass_Msun" in path or "dist_pc" in path
    ]
    assert {path for path in paths if not path.startswith(".dynamics.")} == {
        ".star.flux_phot_per_nm_m2",
        ".star.diameter_arcsec",
        ".planets[0].Rp_Rearth",
        ".planets[0].physical_model.Ag",
        ".midplane_inc_deg",
        ".midplane_pa_deg",
    }
    assert len(mass_distance_paths) == 3
    assert all(path.startswith(".dynamics.") for path in mass_distance_paths)
    assert not any(
        ".orbit" in path for path in paths if not path.startswith(".dynamics.")
    )


@pytest.mark.parametrize("mapping", [(0, 0), (0,), (0, 2), (0.0, 1), (False, 1)])
def test_scene_rejects_invalid_body_mapping(mapping):
    """Fails if a duplicate, missing, out-of-range or noninteger body is accepted."""
    scene = _ephemeris_scene()
    with pytest.raises(ValueError, match="permutation"):
        System(scene.star, scene.planets, scene.dynamics, body_index=mapping)
    with pytest.raises(ValueError, match="count"):
        System(scene.star, scene.planets[:1], scene.dynamics)


def test_empty_scene_and_reference_epoch():
    """Fails if the zero-body chain breaks or time_jd is silently offset by epoch."""
    from hwoutils.epoch import Epoch

    empty = System.from_classical(
        FlatStar(1.0),
        (),
        star_mass_Msun=1.0,
        planet_mass_Msun=[],
        dist_pc=10.0,
        a_AU=[],
        e=[],
        i_rad=[],
        W_rad=[],
        w_rad=[],
        M0_rad=[],
    )
    times = jnp.array([0.0, 100.0])
    assert empty.positions(times).shape == (2, 0, 2)
    assert empty.contrasts(jnp.asarray(600.0), times).shape == (0, 2)
    np.testing.assert_array_equal(empty.star_reflex(times), 0.0)
    scene = _ephemeris_scene()
    changed_frame = type(scene.dynamics.frame)(scene.dist_pc, Epoch(2451545.0))
    changed = eqx.tree_at(lambda s: s.dynamics.frame, scene, changed_frame)
    np.testing.assert_array_equal(changed.positions(times), scene.positions(times))
