"""Absolute disk photometry, sky orientation and JAX contracts from primitives."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import arcsec2rad, h
from hwoutils.grids import PixelGrid, SkyOrientation, resample
from zodi import DensityDust, SurfaceBrightnessDust

from skyscapes.disk import (
    AbstractDisk,
    ExovistaDisk,
    ExovistaParametricDisk,
    GraterDisk,
    density_level,
    surface_brightness_level,
    zero_magnitude_photon_flux_density,
)
from skyscapes.scene import Star


def solar_star(magnitude=4.83, luminosity=1.0):
    """A host at 10 pc with the stated absolute V magnitude and luminosity."""
    return Star(
        wavelengths_nm=jnp.array([500.0, 550.0, 650.0]),
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.full((3, 1), 3636.0 * 10 ** (-0.4 * magnitude)),
        luminosity_Lsun=luminosity,
        interpolation_method="linear",
    )


def profile(kind="exovista", amplitude=None, order=48):
    """Profiles with support at the reference radius, independently of sampling."""
    amplitude = (
        SurfaceBrightnessDust(level=jnp.asarray(3.0))
        if amplitude is None
        else amplitude
    )
    common = dict(
        amplitude=amplitude,
        n_slices_los=order,
        rmin_AU=jnp.asarray(0.1),
        rmax_AU=jnp.asarray(3.0),
        wavelengths_nm=jnp.array([500.0, 650.0]),
        Ag_grid=jnp.array([0.5, 0.5]),
    )
    if kind == "exovista":
        return ExovistaParametricDisk(
            r0_AU=jnp.asarray(1.0),
            dror=jnp.asarray(0.3),
            rinner_AU=jnp.asarray(0.1),
            hor=jnp.asarray(0.03),
            eta=jnp.asarray(1.0),
            g0=jnp.asarray(0.0),
            g1=jnp.asarray(0.0),
            g2=jnp.asarray(0.0),
            w0=jnp.asarray(1.0),
            w1=jnp.asarray(0.0),
            w2=jnp.asarray(0.0),
            **common,
        )
    return GraterDisk(
        sma_AU=jnp.asarray(1.0),
        alpha_in=jnp.asarray(5.0),
        alpha_out=jnp.asarray(-5.0),
        ksi0_AU=jnp.asarray(0.03),
        gamma=jnp.asarray(2.0),
        beta=jnp.asarray(1.0),
        g_HG_grid=jnp.array([0.0, 0.0]),
        **common,
    )


def point_grid(step=0.01, radius=0.1):
    """The single pixel is at east=radius arcsec for any pixel size."""
    return PixelGrid.regular(
        (1, 1),
        step,
        "arcsec",
        center=(radius / step, 0.0),
        orientation=SkyOrientation(0.0),
    )


@pytest.mark.parametrize("kind", ["exovista", "grater"])
def test_calibration_anchor_and_pixel_area(kind):
    """Fails for missing calibration, host contrast dialect, or pixel-area factors.

    1e-6 relative is the required LOS bound, checked at orders 24, 48, 96;
    common-point pixel equality is roundoff (2e-14), without interpolation.
    The V zero point is the explicit 3636 Jy monochromatic convention.
    """
    amplitude = SurfaceBrightnessDust(level=3)
    disk, star = profile(kind, amplitude), solar_star()
    zero = 3636.0 * 1e-26 / (6.62607015e-34 * 550e-9) * 1e-9
    omega_arcsec2 = (np.pi / (180 * 3600)) ** 2
    assert h == 6.62607015e-34
    np.testing.assert_allclose(arcsec2rad**2, omega_arcsec2, rtol=2e-15)
    np.testing.assert_allclose(
        zero_magnitude_photon_flux_density(amplitude.passband), zero, rtol=2e-15
    )
    expected = 3 * 10 ** (-0.4 * 22)
    image = disk.photon_radiance(550.0, 0.0, star=star, dist_pc=10.0, grid=point_grid())
    actual = image.data.item() / zero * omega_arcsec2
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(
        -2.5 * np.log10(actual), 22 - 2.5 * np.log10(3), rtol=2e-15
    )
    # Hold the calibrated density fixed while doubling the render order; do not
    # recalibrate each rule, which would make this convergence check circular.
    density = DensityDust(
        level=density_level(amplitude, disk, star=star, dist_pc=10.0),
        profile_id=disk.profile_id,
        calibration_id="anchor",
    )
    values = [
        profile(kind, density, order)
        .photon_radiance(550.0, 0.0, star=star, dist_pc=10.0, grid=point_grid())
        .data.item()
        for order in (24, 48, 96)
    ]
    np.testing.assert_allclose(values, image.data.item(), rtol=1e-6)
    print("LOS order 24/48/96:", np.asarray(values) / values[-1] - 1)
    fine = disk.photon_radiance(
        550.0, 0.0, star=star, dist_pc=10.0, grid=point_grid(0.005)
    )
    np.testing.assert_allclose(fine.data, image.data, rtol=2e-14)
    pixel_contrast = (
        image.data.item()
        * point_grid().pixel_area
        * omega_arcsec2
        / star.photon_flux_density(550.0, 0.0, dist_pc=10.0)
    )
    expected_pixel = 3 * 10 ** (-0.4 * 22) / 10 ** (-0.4 * 4.83) * 0.010**2
    np.testing.assert_allclose(pixel_contrast, expected_pixel, rtol=1e-6)
    assert image.measure == "per_unit_area"
    assert image.quantity == "ph s^-1 m^-2 nm^-1 sr^-1"


@pytest.mark.parametrize("kind", ["exovista", "grater"])
def test_amplitude_round_trip_and_profile_identity(kind):
    """Fails if conversions omit the profile or mix profile normalizations.

    Tolerance 2e-15 is the float64 multiplication/division floor.
    """
    amplitude = SurfaceBrightnessDust(level=jnp.asarray(3.0))
    disk, star = profile(kind, amplitude), solar_star()
    density = DensityDust(
        level=density_level(amplitude, disk, star=star, dist_pc=10.0),
        profile_id=disk.profile_id,
        calibration_id="round-trip",
    )
    restored = replace(
        amplitude,
        level=surface_brightness_level(
            density, disk, calibration=amplitude, star=star, dist_pc=10.0
        ),
    )
    np.testing.assert_allclose(restored.level, amplitude.level, rtol=2e-15)
    with pytest.raises(ValueError, match="profile_id"):
        profile(kind, replace(density, profile_id="wrong-profile"))
    with pytest.raises(TypeError, match="amplitude"):
        profile(kind, 3.0)


@pytest.mark.parametrize("kind", ["exovista", "grater"])
def test_host_scaling_once(kind):
    """Fails if a K dwarf's band/bolometric host correction is counted twice.

    K dwarf M_V=7 and L=0.2 at 10 pc; compare at sqrt(0.2) AU.
    Ratio tolerance 3e-14 is float64 cancellation, not a profile approximation.
    """
    star = solar_star(magnitude=7.0, luminosity=0.2)
    band = SurfaceBrightnessDust(level=3, luminosity_rule="band")
    bolo = replace(band, luminosity_rule="bolometric")
    grid = point_grid(radius=np.sqrt(0.2) / 10)
    values = [
        profile(kind, a)
        .photon_radiance(550.0, 0.0, star=star, dist_pc=10.0, grid=grid)
        .data
        for a in (band, bolo)
    ]
    solar = (
        profile(kind, band)
        .photon_radiance(550.0, 0.0, star=solar_star(), dist_pc=10.0, grid=grid)
        .data
    )
    # The same physical profile and point cancel, leaving independently stated
    # host luminosities. This also catches a common extra factor in BOTH rules.
    np.testing.assert_allclose(
        values[0] / solar, 10 ** (-0.4 * (7.0 - 4.83)), rtol=3e-14
    )
    np.testing.assert_allclose(values[1] / solar, 0.2, rtol=3e-14)
    ratio = 0.2 / 10 ** (-0.4 * (7.0 - 4.83))
    np.testing.assert_allclose(values[1] / values[0], ratio, rtol=3e-14)
    np.testing.assert_allclose(
        bolo.at_eeid(host_band_abs_mag=7.0, host_luminosity_lsun=0.2)
        / band.at_eeid(host_band_abs_mag=7.0, host_luminosity_lsun=0.2),
        ratio,
        rtol=2e-15,
    )
    assert abs(ratio - 1) > 0.1  # doubling the correction is distinguishable


def test_orientation_and_rotated_grid():
    """Fails for swapped PA axes, reversed near side, or detector-axis assumptions.

    PA=0 has nodes north and the near half west; PA=90 has nodes east.
    Cubic resampling uses a 0.0015 arcsec reference: 0.4% peak tolerance
    bounds sampling error at the radial slope breaks, far below axis errors.
    """
    disk, star = profile("grater"), solar_star()
    disk = eqx.tree_at(
        lambda d: (d.inclination_deg, d.g_HG_grid), disk, (60.0, jnp.array([0.2, 0.2]))
    )
    grid = PixelGrid.regular((81, 81), 0.003, "arcsec", orientation=SkyOrientation(0.0))
    north = disk.photon_radiance(550.0, 0.0, star=star, dist_pc=10.0, grid=grid)
    east, n = grid.orientation.detector_to_sky(*grid.coordinates())
    assert jnp.sum(north.data * n * n) > 2 * jnp.sum(north.data * east * east)
    assert jnp.sum(north.data * (east < 0)) > jnp.sum(north.data * (east > 0))
    turned = eqx.tree_at(lambda d: d.position_angle_deg, disk, 90.0)
    data = turned.photon_radiance(550.0, 0.0, star=star, dist_pc=10.0, grid=grid).data
    assert jnp.sum(data * east * east) > 2 * jnp.sum(data * n * n)
    rotated = replace(grid, orientation=SkyOrientation(30.0))
    reference = PixelGrid.regular(
        (241, 241), 0.0015, "arcsec", orientation=SkyOrientation(0.0)
    )
    ref = disk.photon_radiance(550.0, 0.0, star=star, dist_pc=10.0, grid=reference)
    direct = disk.photon_radiance(550.0, 0.0, star=star, dist_pc=10.0, grid=rotated)
    np.testing.assert_allclose(
        direct.data,
        resample(ref, rotated, reconstruction="cubic").data,
        rtol=0,
        atol=0.004 * direct.data.max(),
    )
    # Both handednesses describe the identical sky when columns are reversed.
    other = disk.photon_radiance(
        550.0,
        0.0,
        star=star,
        dist_pc=10.0,
        grid=replace(grid, orientation=SkyOrientation(0.0, parity=1)),
    )
    np.testing.assert_allclose(other.data, north.data[:, ::-1], rtol=2e-14, atol=0)


@pytest.mark.parametrize("kind", ["exovista", "grater"])
def test_jax_radiance_contract(kind):
    """Fails for detached amplitude levels or wavelength-axis broadcasting.

    5e-14 relative covers float64 fusion/reduction; linear derivatives have
    the 3e-15 multiplication floor. Two wavelength axes name their dimensions.
    """
    disk, star, grid = profile(kind), solar_star(), point_grid()
    waves = jnp.array([[510.0, 550.0], [580.0, 620.0]])

    def render(d, w):
        return d.photon_radiance(w, 0.0, star=star, dist_pc=10.0, grid=grid)

    eager = render(disk, waves)
    np.testing.assert_allclose(
        eqx.filter_jit(render)(disk, waves).data, eager.data, rtol=5e-14
    )
    assert eager.leading_axes == ("wavelength_0", "wavelength_1")
    mapped = jax.vmap(lambda w: render(disk, w).data)(waves.ravel())
    loop = jnp.stack([render(disk, w).data for w in waves.ravel()])
    np.testing.assert_allclose(mapped, loop, rtol=5e-14)

    def scalar(level):
        return render(
            eqx.tree_at(lambda d: d.amplitude.level, disk, level), 550.0
        ).data.sum()

    value, gradient = jax.value_and_grad(scalar)(disk.amplitude.level)
    np.testing.assert_allclose(gradient, value / disk.amplitude.level, rtol=3e-15)


def test_exovista_disk_flux_shape():
    """Fails if stored pixel contrast is treated as a density or re-area-scaled.

    Native-grid recovery has 3e-15 relative algebra roundoff. Near zero,
    allow one float64 epsilon of peak flux for the rotated coordinate round trip.
    """
    wave = jnp.array([500.0, 550.0, 650.0])
    cube = jnp.arange(3 * 4 * 6, dtype=float).reshape(3, 4, 6) * 1e-10
    grid = PixelGrid.regular((4, 6), 0.01, "arcsec", orientation=SkyOrientation(30.0))
    disk = ExovistaDisk(grid, wave, cube)
    star = solar_star()
    image = disk.photon_radiance(wave, 0.0, star=star, dist_pc=10.0, grid=grid)
    expected = cube * star.photon_flux_density(wave, 0.0, dist_pc=10.0)[:, None, None]
    omega = 0.01**2 * (np.pi / (180 * 3600)) ** 2
    np.testing.assert_allclose(
        image.data * omega,
        expected,
        rtol=3e-15,
        atol=np.finfo(expected.dtype).eps * expected.max(),
    )
    assert isinstance(disk, AbstractDisk)


def test_point_disk_planet_centroid():
    """No supported profile can represent a compact Gaussian clump."""
    pytest.skip(
        "Both profiles are star-centered axisymmetric annuli; neither supports "
        "a compact Gaussian density/clump."
    )


def test_system_radiance_owns_midplane_override():
    """Fails if System ignores its angle or multiplies stellar flux a second time.

    The density-normalized solar anchor at zero inclination is independent of
    system plumbing; tolerance 1e-6 is the LOS calibration convergence bound.
    The zero-disk branch and nested composite use the identical requested grid.
    """
    from skyscapes.disk import CompositeDisk
    from skyscapes.scene import System

    disk = eqx.tree_at(
        lambda d: (d.inclination_deg, d.position_angle_deg), profile(), (75.0, 43.0)
    )
    empty = jnp.empty(0)
    system = System.from_classical(
        solar_star(),
        (),
        star_mass_Msun=1.0,
        planet_mass_Msun=empty,
        a_AU=empty,
        e=empty,
        W_rad=empty,
        i_rad=empty,
        w_rad=empty,
        M0_rad=empty,
        dist_pc=10.0,
        disk=CompositeDisk((CompositeDisk((disk,)), disk)),
        midplane_inc_deg=0.0,
        midplane_position_angle_deg=17.0,
    )
    grid = point_grid()
    data = system.photon_radiance(550.0, 0.0, grid=grid).data
    zero = 3636.0 * 1e-26 / (6.62607015e-34 * 550e-9) * 1e-9
    expected = 2 * 3 * 10 ** (-0.4 * 22) * zero / (np.pi / (180 * 3600)) ** 2
    np.testing.assert_allclose(data, expected, rtol=1e-6)
    assert float(disk.inclination_deg) == 75.0  # override is functional
    no_disk = eqx.tree_at(lambda s: s.disk, system, None)
    np.testing.assert_array_equal(
        no_disk.photon_radiance(550.0, 0.0, grid=grid).data, 0.0
    )
    with pytest.raises(ValueError, match="SkyOrientation"):
        system.photon_radiance(550.0, 0.0, grid=replace(grid, orientation=None))


@pytest.mark.parametrize("kind", ["exovista", "grater", "cube", "composite", None])
def test_system_point_radiance_matches_grid(kind):
    """Fails if a disk point uses a different sky frame or wavelength sampling.

    Relative tolerance 5e-14 allows float64 coordinate round trips and JIT
    fusion. The rotated target samples the stored cube between native pixels.
    """
    from skyscapes.disk import CompositeDisk
    from skyscapes.scene import System

    grid = PixelGrid.regular(
        (3, 4),
        0.005,
        "arcsec",
        center=(19.5, 1.1),
        orientation=SkyOrientation(20.0),
    )
    wave = jnp.array([500.0, 575.0, 650.0])
    if kind in ("cube", "composite"):
        native = replace(
            grid,
            shape=(13, 13),
            center_x=18.0,
            center_y=-9.0,
            orientation=SkyOrientation(70.0),
        )
        cube = 1e-10 * jnp.linspace(1.0, 3.0, 3 * 13 * 13).reshape(3, 13, 13)
        disk = ExovistaDisk(native, wave, cube)
        if kind == "composite":
            disk = CompositeDisk((profile(), disk))
    else:
        disk = None if kind is None else profile(kind)
    empty = jnp.empty(0)
    scene = System.from_classical(
        solar_star(),
        (),
        star_mass_Msun=1.0,
        planet_mass_Msun=empty,
        a_AU=empty,
        e=empty,
        W_rad=empty,
        i_rad=empty,
        w_rad=empty,
        M0_rad=empty,
        dist_pc=10.0,
        disk=disk,
        midplane_inc_deg=20.0,
        midplane_position_angle_deg=31.0,
    )
    # Non-knot wavelengths exercise the cube's spectral interpolant.
    query = jnp.array([[520.0], [580.0]])
    east, north = grid.orientation.detector_to_sky(*grid.coordinates())
    expected = scene.photon_radiance(query, 0.0, grid=grid).data
    actual = eqx.filter_jit(scene.photon_radiance_at)(
        query, jnp.asarray(0.0), east, north
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=0)
    if kind == "cube":
        # The input is affine in wavelength, so interpax reproduces linear
        # interpolation. Resample the independent native radiance as before.
        from hwoutils.grids import Image

        contrast = cube[0] + (query - wave[0])[..., None, None] * (
            (cube[-1] - cube[0]) / (wave[-1] - wave[0])
        )
        flux = scene.star.photon_flux_density(query, 0.0, dist_pc=scene.dist_pc)
        radiance = (
            contrast * flux[..., None, None] / (native.pixel_area * arcsec2rad**2)
        )
        native_image = Image(
            radiance,
            native,
            ("wave_row", "wave_column"),
            "per_unit_area",
            "ph s^-1 m^-2 nm^-1 sr^-1",
        )
        np.testing.assert_allclose(
            actual, resample(native_image, grid).data, rtol=5e-14
        )
    if kind is None:
        np.testing.assert_array_equal(actual, 0.0)
