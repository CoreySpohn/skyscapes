"""Disk amplitude calibration is a band quantity: the star is averaged, not sampled.

Under the bolometric luminosity rule the level goes as the inverse of the host
flux in the calibration band, so a point read on a spectral line puts the line
into the level. Under the default band rule that flux cancels, and the level
must not depend on the star at all. The truths here are independent of the
operator that reduces a tabulated star: Gauss-Legendre points on every cell
between the table knots (exact for the linear PHOENIX table) and
``scipy.integrate.quad`` for a Planck spectrum. The PHOENIX 5800 K table is the
offline test fixture, block-averaged to 57 knots in 500 to 555 nm, so it holds
fewer and shallower lines than the full 1427-knot table.
"""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import c, h, k_B, pc2m
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from hwoutils.numerics import eps
from numpy.polynomial.legendre import leggauss
from optixstuff.grids import PixelGrid, SkyOrientation
from optixstuff.spectral import Band, Quadrature, Resolution
from scipy.integrate import quad
from zodi import DensityDust, Passband, SurfaceBrightnessDust

from skyscapes.disk import (
    CompositeDisk,
    density_level,
    zero_magnitude_photon_flux_density,
)
from skyscapes.scene import BlackbodyStar, FlatStar, Star
from tests.test_disk import profile
from tests.test_scene_sources import DIST_PC, TIMES, _phoenix, _scene

V_BAND = Band(506.0, 594.0)
WIDE_BAND = Band(450.0, 650.0)
MAIN_BANDS = [Band(500.0, 555.0), Band(735.0, 785.0)]


class _LuminousFlat(FlatStar):
    """A flat-spectrum host with a luminosity, so the bolometric rule can read it."""

    luminosity_Lsun: float = eqx.field(default=1.0, kw_only=True)


class _LuminousBlackbody(BlackbodyStar):
    """A blackbody host with a luminosity, so the bolometric rule can read it."""

    luminosity_Lsun: float = eqx.field(default=1.0, kw_only=True)


def _dust(rule, **kwargs):
    return SurfaceBrightnessDust(level=3, luminosity_rule=rule, **kwargs)


def _scene_with(star, disk):
    return _scene(star, (), disk=disk)


def _resolved(scene):
    """The exposure rule beside one calibration rule per calibration band."""
    main = Quadrature.build(MAIN_BANDS, breaks_nm=(520.0,))
    cals = tuple(Quadrature.build([band]) for band in scene.calibration_bands())
    return main, Resolution.build((main, *cals), *scene.spectral_tables())


def _table_mean(star, band):
    """Band mean of a linear table: 8 Gauss points on every cell between knots."""
    knots = np.asarray(star.wavelengths_nm)
    inner = knots[(knots > band.low_nm) & (knots < band.high_nm)]
    edges = np.concatenate([[band.low_nm], inner, [band.high_nm]])
    x, w = leggauss(8)
    length = np.diff(edges)[:, None]
    points = (edges[:-1, None] + length * (x + 1) / 2).ravel()
    weights = (length * w / 2).ravel()
    values = np.asarray(
        star.photon_flux_density(jnp.asarray(points), jnp.zeros(()), dist_pc=DIST_PC)
    )
    return np.sum(weights * values) / band.width_nm


def _planck_mean(star, band):
    """Band mean of the diluted Planck photon density by adaptive quadrature."""
    dilution = np.pi * (star.radius_m / (DIST_PC * pc2m)) ** 2

    def photons(nm):
        wave = nm * 1e-9
        radiance = 2 * c / wave**4 / np.expm1(h * c / (wave * k_B * star.teff_K))
        return dilution * radiance * 1e-9

    return quad(photons, band.low_nm, band.high_nm, epsabs=0, epsrel=1e-13)[0] / (
        band.width_nm
    )


def _level(disk, star):
    return float(density_level(disk.amplitude, disk, star=star, dist_pc=DIST_PC))


def _flat_level(disk, luminosity, flux=1.0):
    """Level of the same disk under a flat host: the star's band mean is its flux."""
    return _level(disk, _LuminousFlat(flux, luminosity_Lsun=luminosity)) * flux


@vv.case("stellar-photometry", "code-verification")
@pytest.mark.float64_only(reason="float64 photometry reference")
def test_bolometric_level_divides_by_the_band_mean_of_a_phoenix_star():
    """Fails if the bolometric calibration point-samples the star or mis-keys a band.

    PHOENIX 5800 K under a bolometric disk, with a second disk of a wider band
    (450 to 650 nm) in a composite, both centred on 550 nm. Truths: the band
    mean of the photon table in each band, eight Gauss points on every cell
    between knots (exact for the linear table; independent of the operator),
    compared with ``Sources.calibration`` to 1e-12 relative, basis
    floating-point (measured 4e-16). The level times the band mean is the same
    constant for any star (the level goes as the inverse of the flux), taken
    from a flat host whose band mean is its own flux; 1e-12 relative, basis
    floating-point (measured 4e-16). Mutants, each measured on this fixture: a
    point sample at 550 nm in place of the mean misses the level by 7.6e-3 in
    the 88 nm band and 1.3e-2 in the 200 nm band (on the full 1427-knot table
    the 88 nm figure is about 7e-2), and a lookup that ignores the band reads
    the first band's mean for both; each is far above the tolerance.
    """
    star = _phoenix()
    narrow = profile("grater", _dust("bolometric"))
    wide = profile("grater", _dust("bolometric"), calibration_band=WIDE_BAND)
    scene = _scene_with(star, CompositeDisk((narrow, wide)))
    assert scene.calibration_bands() == (V_BAND, WIDE_BAND)
    rule, resolution = _resolved(scene)
    sources = scene.sources(rule, resolution, TIMES)
    luminosity = star.luminosity_Lsun
    node_star = sources.node_star()
    point = float(
        star.photon_flux_density(jnp.asarray(550.0), jnp.zeros(()), dist_pc=DIST_PC)
    )
    for disk, band, calibration in zip(
        (narrow, wide),
        (V_BAND, WIDE_BAND),
        np.asarray(sources.calibration),
        strict=True,
    ):
        mean = _table_mean(star, band)
        np.testing.assert_allclose(calibration, mean, rtol=1e-12)
        constant = _flat_level(disk, luminosity)
        level = _level(disk, node_star)
        np.testing.assert_allclose(level * mean, constant, rtol=1e-12)
        assert abs(point / mean - 1) > 1e-4
        assert abs(level * point / constant - 1) > 1e-4
    # The two bands are different numbers, so a lookup that ignores the band fails.
    assert abs(sources.calibration[1] / sources.calibration[0] - 1) > 1e-3


@vv.case("stellar-photometry", "code-verification")
@pytest.mark.float64_only(reason="float64 photometry reference")
def test_bolometric_level_divides_by_the_band_mean_of_a_blackbody():
    """Fails if a smooth star's band mean is a point sample or the wrong band.

    A 5772 K blackbody host with a luminosity, bolometric disk. Truth:
    ``scipy.integrate.quad`` of the diluted Planck photon density over 506 to
    594 nm at relative tolerance 1e-13, against ``Sources.calibration`` (the
    eight-point rule of a smooth star, which needs no calibration quadrature)
    and the level times the band mean against the flat-host constant. The
    ruling's bound is 1e-9 (an order-four rule on a smooth source); the
    eight-point rule measures 4e-16, so the test asserts the ruling's 1e-9. A
    point sample at 550 nm misses the band mean by 4.9e-3 here, far above it.
    """
    star = _LuminousBlackbody(5772.0, 6.957e8, luminosity_Lsun=1.0)
    disk = profile("grater", _dust("bolometric"))
    scene = _scene_with(star, disk)
    rule, resolution = _resolved(scene)
    sources = scene.sources(rule, resolution, TIMES)
    mean = _planck_mean(star, V_BAND)
    np.testing.assert_allclose(sources.calibration[0], mean, rtol=1e-9)
    np.testing.assert_allclose(
        _level(disk, sources.node_star()) * mean, _flat_level(disk, 1.0), rtol=1e-9
    )
    # The star's own method, with no scene, is the same number.
    direct = star.calibration_flux_density(V_BAND, 0.0, dist_pc=DIST_PC)
    np.testing.assert_allclose(direct, mean, rtol=1e-9)
    point = float(
        star.photon_flux_density(jnp.asarray(550.0), jnp.zeros(()), dist_pc=DIST_PC)
    )
    assert abs(point / mean - 1) > 1e-3


@vv.case("stellar-photometry", "code-verification")
def test_band_rule_level_does_not_read_the_star():
    """Fails if the band rule reads the star on one side of the calibration only.

    The default rule on PHOENIX with the table values scaled by 1.07, so the
    point flux at the band center is 7 percent higher: the level equals the
    unscaled level to 4e-16 relative, a few ULP (basis floating-point; the
    star is not read, so the measured difference is zero). A mutant that
    divides by the point flux and builds the magnitude from one, which is the
    band rule routed through the calibration flux on one side only, moves the
    level by 0.065 (a factor of 1/1.07), more than the 1e-9 asserted. The rule
    also holds for a star with no flux at the calibration wavelength, where a
    magnitude of the point flux would be 0/0, and for a bare table that
    covers only part of the calibration band.
    """
    base = _phoenix()
    scaled = Star(
        wavelengths_nm=base.wavelengths_nm,
        times_d=jnp.zeros(1),
        flux_density_jy=1.07 * base.flux_density_jy,
        luminosity_Lsun=base.luminosity_Lsun,
    )
    disk = profile("grater", _dust("band"))
    unscaled_level = _level(disk, base)
    np.testing.assert_allclose(_level(disk, scaled), unscaled_level, rtol=4e-16)
    zero = Star(
        wavelengths_nm=jnp.array([600.0, 650.0]),
        times_d=jnp.zeros(1),
        flux_density_jy=jnp.zeros((2, 1)),
        luminosity_Lsun=1.0,
    )
    assert _level(disk, zero) == unscaled_level
    # The band rule on a PHOENIX scene needs no calibration quadrature.
    scene = _scene_with(base, disk)
    assert scene.calibration_bands() == ()
    rule, resolution = _resolved(scene)
    assert scene.sources(rule, resolution, TIMES).calibration.shape == (0,)


@vv.case("stellar-photometry", "code-verification")
def test_a_band_rule_disk_carries_the_node_effective_star():
    """Fails if the disk is read against a point sample of the star at the nodes.

    The all-node star on a PHOENIX scene with a band-rule disk (the S1 read
    that raised, because the level read the star at one wavelength): the disk
    at every node equals the disk against the star divided by the star's point
    value and multiplied by its node-effective value, since the level does not
    depend on the star. The ratio allows 16 rounded operations for the
    spectral interpolation, node weights and flux scaling on all nodes at once
    and on node (0, 2) alone (a transposed index reads another node). A node differs
    from its point sample by up to 8e-2 on this fixture.
    """
    star = _phoenix()
    scene = _scene_with(star, profile("grater", _dust("band")))
    rule, resolution = _resolved(scene)
    sources = scene.sources(rule, resolution, TIMES)
    nodes = rule.nodes_nm
    grid = PixelGrid.regular((1, 1), 0.01, "arcsec", orientation=SkyOrientation(0.0))
    east, north = grid.coordinates()
    east = east + 0.1
    every = scene.photon_radiance_at(nodes, 0.0, east, north, star=sources.node_star())
    point = scene.photon_radiance_at(nodes, 0.0, east, north)
    point_star = star.photon_flux_density(nodes, jnp.zeros(()), dist_pc=DIST_PC)
    expected = point * (sources.unscaled_star[..., 0] / point_star)[..., None, None]
    # The first band only: the disk's scattering table ends at 650 nm.
    np.testing.assert_allclose(every[0], expected[0], rtol=16 * eps(every), atol=0)
    assert np.max(np.abs(every[0] / point[0] - 1)) > 1e-2
    one = scene.photon_radiance_at(
        nodes[0, 2], 0.0, east, north, star=sources.node_star(0, 2)
    )
    np.testing.assert_allclose(one, every[0, 2], rtol=16 * eps(every), atol=0)


def test_calibration_without_the_band_quadrature_is_refused():
    """Fails if a tabulated star's calibration can be read without its operator.

    Exact refusals: a bare ``Star`` names ``Scene.sources``; ``Scene.sources``
    without the calibration quadrature names it; a node star asked for a band
    that was not computed names ``Scene.calibration_bands``; a smooth host
    needs no quadrature.
    """
    star = _phoenix()
    disk = profile("grater", _dust("bolometric"))
    scene = _scene_with(star, disk)
    with pytest.raises(LookupError, match=r"Scene\.sources"):
        _level(disk, star)
    main = Quadrature.build(MAIN_BANDS, breaks_nm=(520.0,))
    with pytest.raises(LookupError, match="calibration quadrature"):
        scene.sources(main, Resolution.build(main, *scene.spectral_tables()), TIMES)
    _, resolution = _resolved(scene)
    sources = scene.sources(main, resolution, TIMES)
    with pytest.raises(LookupError, match="calibration_bands"):
        sources.node_star().calibration_flux_density(WIDE_BAND, 0.0, dist_pc=DIST_PC)
    smooth = _LuminousBlackbody(5772.0, 6.957e8)
    smooth_scene = _scene_with(smooth, disk)
    smooth_scene.sources(main, Resolution(), TIMES)


def test_calibration_band_declaration_is_checked():
    """Fails if an off-center band or a bolometric rule with no band passes.

    Exact. The default for Johnson V is the 506 to 594 nm top hat; a band not
    centred on the passband center, a non-V passband under the bolometric rule
    with no band, and a band that is not a ``Band`` are refused at
    construction, while an AB passband with a centred band is accepted.
    """
    default = profile("grater", _dust("bolometric"))
    assert default.calibration_bands() == (V_BAND,)
    assert profile("grater", _dust("band")).calibration_bands() == ()
    density = DensityDust(level=1.0, profile_id="grater-1999", calibration_id="x")
    assert profile("grater", density).calibration_bands() == ()
    with pytest.raises(ValueError, match="centred"):
        profile("grater", _dust("bolometric"), calibration_band=Band(500.0, 620.0))
    ab = Passband("g", 480.0, "ab")
    with pytest.raises(ValueError, match="calibration_band"):
        profile("grater", _dust("bolometric", passband=ab))
    centred = profile(
        "grater", _dust("bolometric", passband=ab), calibration_band=Band(440.0, 520.0)
    )
    assert centred.calibration_bands() == (Band(440.0, 520.0),)
    # A passband read in float32 still identifies its host-defined band.
    host_center = 480.123456
    rounded = Passband("g", float(np.float32(host_center)), "ab")
    band = Band(host_center - 40.0, host_center + 40.0)
    assert profile(
        "grater", _dust("bolometric", passband=rounded), calibration_band=band
    ).calibration_bands() == (band,)
    with pytest.raises(TypeError, match="Band"):
        profile("grater", _dust("bolometric"), calibration_band=(506.0, 594.0))


@vv.case("stellar-photometry", "code-verification")
@pytest.mark.float64_only(reason="float64 photometry reference")
def test_zero_point_is_the_band_mean_of_the_zero_magnitude_density():
    """Fails if the zero point is read at the center or averaged in the wrong variable.

    Truth: ``scipy.integrate.quad`` of 3636 Jy as a photon density over 506 to
    594 nm, divided by the width, at relative tolerance 1e-13 (the integrand is
    smooth: basis floating-point, measured 2e-16). A call with no band is the
    monochromatic value at 550 nm, and the band mean is lower by
    1 - 1/(550 mean(1/lambda)) = 2.1e-3, which a center read misses by far more
    than the tolerance.
    """
    passband = _dust("band").passband

    def photons(nm):
        return jy_to_photons_per_nm_per_m2(3636.0, nm)

    truth = quad(photons, 506.0, 594.0, epsabs=0, epsrel=1e-13)[0] / 88.0
    mean = zero_magnitude_photon_flux_density(passband, band=V_BAND)
    np.testing.assert_allclose(mean, truth, rtol=1e-13)
    centre = zero_magnitude_photon_flux_density(passband)
    np.testing.assert_allclose(centre, photons(550.0), rtol=1e-15)
    assert abs(centre / mean - 1) > 1e-3


def test_sources_calibration_is_one_traced_program():
    """Fails if the calibration is a host constant or breaks under jit.

    ``Scene.sources`` under ``eqx.filter_jit`` equals the eager call (1e-14
    relative, basis floating-point) and the calibration follows a rescaled
    table: a star with every value doubled returns twice the band mean (1e-12),
    so the value is read in the graph, not frozen at the build.
    """
    star = _phoenix()
    disk = profile("grater", _dust("bolometric"))
    scene = _scene_with(star, disk)
    rule, resolution = _resolved(scene)
    eager = scene.sources(rule, resolution, TIMES).calibration
    jitted = eqx.filter_jit(lambda s: s.sources(rule, resolution, TIMES))(scene)
    np.testing.assert_allclose(jitted.calibration, eager, rtol=1e-14)
    doubled = eqx.tree_at(
        lambda s: s.star._flux_density_phot, scene, 2 * star._flux_density_phot
    )
    np.testing.assert_allclose(
        doubled.sources(rule, resolution, TIMES).calibration, 2 * eager, rtol=1e-12
    )
    grad = jax.grad(
        lambda values: (
            eqx.tree_at(lambda s: s.star._flux_density_phot, scene, values)
            .sources(rule, resolution, TIMES)
            .calibration[0]
        )
    )(star._flux_density_phot)
    # The mean is linear in the table, so its gradient sums to the band mean
    # over the table's own scale: one unit of every value moves it by one.
    np.testing.assert_allclose(jnp.sum(grad), 1.0, rtol=1e-12)
