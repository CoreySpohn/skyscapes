"""Import contracts for the legacy and spotted-star ExoVista layouts."""

import warnings

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.constants import R_sun
from astropy.io import fits
from hwoutils.constants import (
    AU2m,
    G_si,
    Jy,
    Mearth2kg,
    Msun2kg,
    c,
    d2s,
    h,
    k_B,
    pc2m,
    um2nm,
    yr2s,
)
from hwoutils.conversions import flux_jy_to_mag
from hwoutils.epoch import Epoch
from hwoutils.numerics import roundoff
from hwoutils.precision import precision_of
from hwoutils.testing import ulps
from optixstuff.spectral import Band, Knots, Quadrature, Resolution

from skyscapes.io import (
    exovista_header,
    from_exovista,
    get_earth_like_planet_indices,
)
from skyscapes.io.exovista import _phase_model
from skyscapes.physical_model import TimeSeriesPhysicalModel
from skyscapes.scene import Scene


@pytest.fixture
def scene_file(tmp_path):
    """Write small scenes with distinct planet spectra at identical phases."""

    def write(*, version=2.6, n_times=3, n_spots=2, wave_dtype=np.float64, header=None):
        wave = np.array([0.5, 0.6, 0.7], dtype=wave_dtype)
        times = np.arange(n_times) * 0.001
        primary = fits.PrimaryHDU(wave)
        primary.header["VERSION"] = version
        # Recent ExoVista writers undercount extensions; use actual body HDUs.
        primary.header["N_EXT"] = 5 if version >= 2.5 else 6
        hdus = [primary, fits.ImageHDU(wave), fits.ImageHDU(), fits.ImageHDU()]
        if version >= 2.5:
            spots = np.zeros((n_times, n_spots, 4), dtype=np.float32)
            spots[..., 0] = 15.0
            spots[..., 1] = 25.0
            spots[..., 2] = 1e18
            spots[..., 3] = 5.0
            hdus.append(fits.ImageHDU(spots))
        star = np.zeros((n_times, 19))
        star[:, 0] = times
        star[:, 16:] = (1.0 + np.arange(n_times))[:, None]
        star_hdu = fits.ImageHDU(star)
        for key, value in {
            "MASS": 1.0,
            "DIST": 10.0,
            "ANGDIAM": 0.9,
            # A catalog target's radius, derived from its 0.9 mas diameter.
            "RSTAR": 10.0 * 0.9e-3 / 0.00465047 / 2.0,
            "LSTAR": 1.0,
            "I": 0.0,
            "PA": 0.0,
            "TEFF": 5778.0,
            "SPECUNIT": "Jy",
            "HIERARCH TeffUmbra": 2500.0,
            "HIERARCH TeffPenumbra": 2700.0,
        }.items():
            star_hdu.header[key] = value
        for key, value in (header or {}).items():
            star_hdu.header[key] = value
        hdus.append(star_hdu)
        for index, (radius, mass, axis) in enumerate(
            ((1.0, 1.0, 1.0), (11.0, 318.0, 5.0))
        ):
            data = np.zeros_like(star)
            data[:, 0] = times
            data[:, 9] = axis
            data[:, 13] = (
                np.sqrt(G_si * (Msun2kg + mass * Mearth2kg) / (axis * AU2m))
                * yr2s
                / AU2m
            )
            data[:, 16:] = (index + 1) * (1.0 + np.arange(n_times))[:, None] * 1e-10
            body = fits.ImageHDU(data)
            for key, value in {"R": radius, "M": mass, "A": axis, "E": 0.0}.items():
                body.header[key] = value
            hdus.append(body)
        tag = "-".join(str(v) for v in (header or {}).values())
        path = tmp_path / f"scene-{version}-{n_times}-{n_spots}-{tag}.fits"
        fits.HDUList(hdus).writeto(path)
        return str(path)

    return write


@pytest.mark.parametrize("wave_dtype", [np.float32, np.float64])
def test_loaded_knots_are_the_file_grid(scene_file, wave_dtype):
    """Fails if the loader rounds the stored micron grid on its way to the knots.

    Exact digest. The star's and every planet's knots are the stored micron
    grid (float32 or float64 on disk), widened to float64 and then scaled to
    nm, computed here from the file.
    """
    path = scene_file(wave_dtype=wave_dtype)
    truth = Knots(np.asarray(fits.getdata(path, 0), dtype=np.float64) * um2nm)
    scene = from_exovista(path, load_disk=False)
    assert scene.star.knots.digest == truth.digest
    assert [p.physical_model.knots.digest for p in scene.planets] == [truth.digest] * 2


@pytest.mark.parametrize("version", [2.33, 2.5, 2.6])
def test_layout_selects_actual_bodies_and_earths(scene_file, version):
    """Header identity determines body count and selection, not N_EXT."""
    path = scene_file(version=version)
    scene = from_exovista(path, load_disk=False)
    assert scene.n_planets == 2
    assert float(scene.star.angular_diameter_arcsec(scene.dist_pc)) == pytest.approx(
        0.0009
    )
    assert get_earth_like_planet_indices(path) == [0]
    assert from_exovista(path, only_earths=True, load_disk=False).n_planets == 1
    selected = from_exovista(path, planet_indices=[1, 0], load_disk=False)
    np.testing.assert_array_equal([p.Rp_Rearth[0] for p in selected.planets], [11, 1])


def test_earth_selection_uses_file_header_precision(scene_file):
    """The half-open habitable zone uses the float64 values from the file.

    One planet is just inside the outer limit and the other is just below
    the inner limit. Both would round onto the corresponding limit in float32.
    """
    path = scene_file()
    with fits.open(path, mode="update") as hdul:
        hdul[6].header["A"] = 1.67 - 1e-9
        hdul[7].header["A"] = 0.95 - 1e-9
        hdul[7].header["R"] = 1.0
    assert get_earth_like_planet_indices(path) == [0]
    assert from_exovista(path, only_earths=True, load_disk=False).n_planets == 1


def test_new_layout_preserves_spot_history_and_stellar_temperatures(scene_file):
    """Spot areas convert from cm^2 to m^2; temperatures survive import."""
    scene = from_exovista(scene_file(), load_disk=False)
    spots = scene.star.spots
    assert spots.latitude_deg.shape == (3, 2)
    np.testing.assert_array_equal(spots.longitude_deg, 25.0)
    np.testing.assert_allclose(spots.area_m2, 1e14, rtol=1e-7, atol=0)
    np.testing.assert_array_equal(spots.area_ratio, 5.0)
    assert spots.umbra_temperature_K == 2500
    assert spots.penumbra_temperature_K == 2700


def test_header_is_host_data_beside_a_batchable_scene(scene_file):
    """Fails if header values ride on the scene and block batching.

    Truth: the values written into the fixture's star extension, read back
    through ``exovista_header`` as plain Python data; two scenes loaded from
    files that differ only in header values (RA, DEC, TEFF, a target id) have
    equal structure, so they batch, and the batch keeps each file's sky
    position. The tolerance is exact equality.
    """
    a = scene_file(header={"RA": 10.0, "DEC": -5.0, "TEFF": 5000.0, "STARID": 1})
    b = scene_file(header={"RA": 200.0, "DEC": 40.0, "TEFF": 6000.0, "STARID": 2})
    header = exovista_header(a)
    assert type(header) is dict
    assert header["TEFF"] == 5000.0
    assert header["STARID"] == 1
    assert header["RA"] == 10.0
    assert header["MASS"] == 1.0
    assert not {"COMMENT", "HISTORY", ""} & header.keys()
    shared = Epoch(2451545.0)
    cadence = np.arange(3) * 2.0**-26
    for path, start in ((a, 0.0), (b, 30.0)):
        with fits.open(path, mode="update") as hdul:
            for hdu in hdul[5:]:
                hdu.data[:, 0] = start + cadence
    scenes = [
        from_exovista(
            p, planet_indices=[0], load_disk=False, local_zodi=None, epoch=shared
        )
        for p in (a, b)
    ]
    assert not hasattr(scenes[0].star, "metadata")
    batch = Scene.from_scenes(scenes)
    assert batch.batch_shape == (2,)
    np.testing.assert_array_equal(batch.ra_deg, [10.0, 200.0])
    np.testing.assert_array_equal(batch.dec_deg, [-5.0, 40.0])
    dates = (np.array([0.0, 30.0])[:, None] + cadence) * (yr2s / d2s)
    expected = [np.array([1.0, 2.0, 3.0], dtype=np.float32) * np.float32(1e-10)]
    reads = batch.contrasts(600.0, dates)
    for row, clock, scene, other in zip(
        reads, dates, scenes, reversed(scenes), strict=True
    ):
        np.testing.assert_array_equal(row, expected)
        restored = Scene.from_table(scene.to_table(), template=other)
        np.testing.assert_array_equal(restored.contrasts(600.0, clock), expected)


def test_new_layout_keeps_time_dependent_contrast_at_equal_phase(scene_file):
    """Phase-only regridding must not erase a factor-three illumination change."""
    scene = from_exovista(scene_file(), load_disk=False)
    times = scene.star._times_d
    np.testing.assert_allclose(
        scene.contrasts(600.0, times)[0], [1e-10, 2e-10, 3e-10], rtol=1e-6, atol=0
    )
    # At the table knots: Jy -> photons = 1e-26 / (h * wavelength_nm).
    expected = np.array([1, 4, 9]) * 1e-10 * 1e-26 / (h * 600.0)
    np.testing.assert_allclose(
        scene.planet_flux_densities(600.0, times)[0], expected, rtol=1e-6, atol=0
    )


@pytest.mark.parametrize("version", [2.33, 2.5, 2.6])
@pytest.mark.parametrize("planet_spectra", ["auto", "time", "phase"])
def test_single_snapshot_and_empty_spots_are_supported(
    scene_file, version, planet_spectra
):
    """A snapshot is constant in time; zero spots is distinct from no metadata."""
    scene = from_exovista(
        scene_file(version=version, n_times=1, n_spots=0),
        load_disk=False,
        planet_spectra=planet_spectra,
    )
    if version >= 2.5:
        assert scene.star.spots.latitude_deg.shape == (1, 0)
    else:
        assert scene.star.spots is None
    times = scene.star._times_d + jnp.array([0.0, 1.0])
    np.testing.assert_allclose(
        scene.star.photon_flux_density(600.0, times, dist_pc=scene.dist_pc),
        1e-26 / (h * 600.0),
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_allclose(
        scene.contrasts(600.0, times)[0], 1e-10, rtol=1e-6, atol=0
    )


@pytest.mark.parametrize(
    ("version", "inclination_deg", "amplitudes", "expected_amplitude"),
    [
        (2.33, 0.0, [1.0], 1.0),
        (2.5, 0.0, [1.0], 1.0),
        (2.6, 0.0, [1.0], 1.0),
        (2.33, 0.0, [1.0, 1.0], 1.0),
        (2.33, 0.1, [1.0, 3.0], 2.0),
        (2.33, 4.0, [1.0, 3.0, 5.0], [1.0, 1.0, 1.0, 4.0]),
    ],
)
def test_phase_model_preserves_face_on_and_snapshot_spectra(
    scene_file, version, inclination_deg, amplitudes, expected_amplitude
):
    """Fails if repeated phases erase a spectrum or extrapolate past its edges."""
    path = scene_file(version=version, n_times=len(amplitudes))
    with fits.open(path, mode="update") as hdul:
        rows = hdul[-2].data
        # Start at quadrature. On the 4 deg orbit the last two phases differ
        # by less than a microdegree, so their mean sets the upper plateau.
        rows[:, 0] = [0.0, 0.125, 0.125 + 1e-8][: len(amplitudes)]
        speed = rows[0, 13]
        tilt = np.deg2rad(inclination_deg)
        rows[0, 13:15] = speed * np.array([np.cos(tilt), np.sin(tilt)])
        rows[:, 16:] = np.outer(amplitudes, [1.0, 2.0, 4.0]) * 1e-10
    scene = from_exovista(
        path, planet_indices=[0], load_disk=False, planet_spectra="phase"
    )
    model = scene.planets[0].physical_model
    phases = jnp.deg2rad(jnp.array([[0.0, 30.0, 90.0, 180.0]]))
    for wavelength, expected in [(500.0, 1.0), (550.0, 1.5), (700.0, 4.0)]:
        np.testing.assert_allclose(
            model.contrast(phases, jnp.ones_like(phases), wavelength, jnp.ones(1))[0],
            expected * np.asarray(expected_amplitude) * 1e-10,
            rtol=1e-6,
            atol=0,
        )


def test_one_phase_grid_interval_resolves_the_spectrum(precision):
    """A span of one grid interval retains its two endpoint spectra in both lanes."""
    phases = np.deg2rad([90.0, 90.0 + 180.0 / 99]).astype(precision)
    table = TimeSeriesPhysicalModel(
        wavelengths_nm=[500.0, 600.0],
        times_d=[0.0, 1.0],
        contrast_grid=np.array([[[1.0, 3.0], [2.0, 4.0]]]),
    )
    grid = _phase_model(table, phases)
    np.testing.assert_array_equal(grid.contrast_grid[0, :, 0], [1.0, 2.0])
    np.testing.assert_array_equal(grid.contrast_grid[0, :, -1], [3.0, 4.0])


def test_legacy_has_no_imported_spots(scene_file):
    """Older files do not pretend to carry measured zero spot coverage."""
    scene = from_exovista(scene_file(version=2.33), load_disk=False)
    assert scene.star.spots is None


def test_negative_planet_index_does_not_select_star(scene_file):
    """Reject indices that would address a non-planet extension."""
    with pytest.raises(ValueError, match="planet_indices"):
        from_exovista(scene_file(), planet_indices=[-1], load_disk=False)


def test_elapsed_years_use_fixed_duration_and_shared_epoch(scene_file):
    """Half a simulation year is 182.625 days, independent of leap years."""
    path = scene_file()
    with fits.open(path, mode="update") as hdul:
        for hdu in hdul[5:]:
            hdu.data[:, 0] = [0.0, 0.5, 1.0]
    scene = from_exovista(path, load_disk=False)
    expected = np.array([0.0, 182.625, 365.25])
    np.testing.assert_array_equal(scene.star._times_d, expected)
    np.testing.assert_array_equal(scene.star.spots.times_d, expected)
    for planet in scene.planets:
        np.testing.assert_array_equal(planet.physical_model.times_d, expected)
    np.testing.assert_array_equal(scene.dynamics.orbits.epoch_d, expected[0])


def test_imported_positive_flux_cannot_overshoot_below_zero(scene_file):
    """A deep positive flux minimum must not become negative between samples."""
    path = scene_file(n_times=4)
    with fits.open(path, mode="update") as hdul:
        hdul[5].data[:, 16:] = np.array([1.0, 0.01, 0.01, 1.0])[:, None]
    scene = from_exovista(path, load_disk=False)
    midpoint = jnp.mean(scene.star._times_d[1:3])
    flux = scene.star.photon_flux_density(600.0, midpoint, dist_pc=scene.dist_pc)
    expected = 0.01 * 1e-26 / (h * 600.0)
    np.testing.assert_allclose(flux, expected, rtol=1e-6, atol=0)
    assert np.all(np.asarray(scene.planet_flux_densities(600.0, midpoint)) > 0)


@pytest.mark.parametrize("epoch", [None, Epoch(2451545.0)])
@pytest.mark.parametrize("mode", ["time", "phase"])
def test_import_centers_states_and_times_before_narrowing(
    scene_file, precision, epoch, mode
):
    """Recover a one-AU separation and sub-second samples from float64 FITS rows.

    Raw coordinates and years round together in float32. Allow one rounding
    of each host day offset, and 32 operations at the orbital radius for the
    Cartesian-to-frame conversion, Kepler evaluation and sky projection.
    At the knots, allow two roundings for linear interpolation of the recorded
    float32 contrasts. The stellar spectrum is stored as float32 in either
    process, so its photon flux allows four float32 roundings (storage, the
    wavelength interpolation, and the Jy-to-photon conversion).
    """
    stored = float(roundoff(1.0, n_ops=4, dtype=np.float32))
    path = scene_file()
    with fits.open(path, mode="update") as hdul:
        years = 30.0 + np.arange(3) * 2.0**-26
        for body in hdul[5:]:
            body.data[:, 0] = years
            body.data[:, 9:12] += 2.0**27
        expected_east = -(hdul[6].data[0, 9] - hdul[5].data[0, 9])
        expected_east /= hdul[5].header["DIST"]
    scene = from_exovista(
        path,
        planet_indices=[0],
        load_disk=False,
        epoch=epoch,
        planet_spectra=mode,
    )
    offsets = (years - years[0]) * (yr2s / d2s)
    assert precision_of(scene) == precision
    assert scene.star._times_d.dtype == precision
    np.testing.assert_array_equal(scene.star._times_d, offsets.astype(precision))
    np.testing.assert_array_equal(scene.star.spots.times_d, offsets.astype(precision))
    np.testing.assert_array_equal(scene.dynamics.orbits.epoch_d, 0)
    first_d = 0.0 if epoch is None else years[0] * (yr2s / d2s)
    dates = first_d + offsets
    amplitudes = np.arange(1, 4)
    if mode == "time":
        np.testing.assert_array_equal(
            scene.planets[0].physical_model.times_d, offsets.astype(precision)
        )
        expected = amplitudes * 1e-10
    else:
        # The face-on orbit has one unresolved phase: its spectrum is the mean.
        expected = np.full(3, amplitudes.mean() * 1e-10)

    def read(scene):
        return scene.contrasts(600.0, dates), scene.star.photon_flux_density(
            600.0, dates, dist_pc=scene.dist_pc
        )

    for contrast, flux in (read(scene), eqx.filter_jit(read)(scene)):
        np.testing.assert_allclose(
            contrast[0], expected.astype(np.float32), rtol=float(ulps(2, 1.0)), atol=0
        )
        np.testing.assert_allclose(
            flux, amplitudes * Jy / (h * 600.0), rtol=stored, atol=0
        )
    # The star table holds the photon flux density at the 600 and 700 nm knots
    # and interpolates it linearly, so a band between them integrates
    # analytically whatever the node layout.
    low, high = 610.0, 640.0
    slope = (1 / 700 - 1 / 600) / 100
    integral = (high - low) / 600 + slope * ((high - 600) ** 2 - (low - 600) ** 2) / 2
    rule = Quadrature.build([Band(low, high)])
    resolution = Resolution.build(rule, *scene.spectral_tables())
    sources = scene.sources(rule, resolution, dates)
    weights = np.asarray(rule.weights(), dtype=np.float64)[0]
    band_flux = np.einsum("q,qt->t", weights, np.asarray(sources.unscaled_star[0]))
    np.testing.assert_allclose(
        band_flux,
        amplitudes * Jy / h * integral,
        rtol=stored + float(roundoff(1.0, n_ops=16, dtype=precision)),
        atol=0,
    )
    np.testing.assert_allclose(
        sources.contrast[0, 0, 0],
        expected.astype(np.float32),
        rtol=stored,
        atol=0,
    )
    positions = scene.positions(dates)
    np.testing.assert_array_equal(
        scene.ephemeris(dates).position_AU, scene.dynamics.solve(dates).position_AU
    )
    actual = positions[:, 0, 0]
    assert actual.dtype == precision
    np.testing.assert_allclose(
        actual, [expected_east, 0.0], rtol=0, atol=float(ulps(32, expected_east))
    )


def _writer_mask_area(mapsize=512, u1=0.0473, u2=0.0841):
    """Area of the ExoVista v2.5/v2.6 limb-darkening mask, from its definition."""
    j, k = np.indices((mapsize, mapsize))
    half = mapsize / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.sqrt(half**2 - (j - half) ** 2 - (k - half) ** 2) / half
        mu2 = np.maximum(1 - mu, 0)
        mask = np.maximum((1 - (u1 + 1) * mu2 - u2 * mu2**2) / mu, 0)
    return np.nan_to_num(mask).sum() * 4 / np.pi / mapsize**2


def _blackbody_surface_flux(wavelength_um, temperature_K):
    """Emergent surface flux pi B_lambda in W m^-2 m^-1."""
    lam = wavelength_um * 1e-6
    x = h * c / (lam * k_B * temperature_K)
    return np.pi * 2 * h * c**2 / lam**5 / np.expm1(x)


_PHOENIX_STAR = {"temperature_K": 5800.0, "radius_Rsun": 1.2, "dist_pc": 15.0}


def _write_phoenix_star(path, header_updates=None):
    """Store a blackbody star the way the v2.5 and v2.6 writers do, unmarked.

    Returns the channel wavelengths in microns and the star's observer flux
    density in Jy at each, computed from the Planck function alone.
    """
    star = _PHOENIX_STAR
    with fits.open(path, mode="update") as hdul:
        wave_um = np.asarray(hdul[0].data, dtype=np.float64)
        surface_cgs = _blackbody_surface_flux(wave_um, star["temperature_K"]) * 10.0
        hdul[5].data[:, 16:] = _writer_mask_area() * surface_cgs[None, :]
        header = hdul[5].header
        del header["SPECUNIT"]
        header["RSTAR"] = star["radius_Rsun"]
        header["DIST"] = star["dist_pc"]
        for key, value in (header_updates or {}).items():
            if value is None:
                del header[key]
            else:
                header[key] = value

    return wave_um, _phoenix_star_observer_jy(wave_um)


def _phoenix_star_observer_jy(wave_um):
    """Observer flux density in Jy of the blackbody star at each wavelength."""
    star = _PHOENIX_STAR
    lam = wave_um * 1e-6
    radius_m = star["radius_Rsun"] * R_sun.to_value("m")
    dilution = (radius_m / (star["dist_pc"] * pc2m)) ** 2
    f_nu_jy = _blackbody_surface_flux(wave_um, star["temperature_K"]) * lam**2 / c
    return f_nu_jy * dilution / Jy


def test_unmarked_recent_spectra_are_observer_flux_of_the_phoenix_star(scene_file):
    """A v2.5+ writer's surface flux becomes the star's flux at the observer.

    The stored spectrum is what the v2.5 and v2.6 writers record for a
    blackbody star: the surface flux density in erg s^-1 cm^-2 cm^-1 times the
    area of their limb-darkening mask. The star at the observer is that surface
    flux diluted by (R / d)^2, expressed per unit frequency, which is computed
    here from the Planck function alone. With no VMAG to check against, the
    reader converts and says it could not check.
    """
    path = scene_file()
    wave_um, f_nu_jy = _write_phoenix_star(path)
    with pytest.warns(RuntimeWarning, match="without checking"):
        scene = from_exovista(path, load_disk=False)

    for wavelength_um, expected_jy in zip(wave_um, f_nu_jy, strict=True):
        wavelength_nm = wavelength_um * 1e3
        # Jy -> photons s^-1 m^-2 nm^-1 = F_nu / (h * wavelength_nm).
        expected = expected_jy * Jy / (h * wavelength_nm)
        got = scene.star.photon_flux_density(
            wavelength_nm, scene.star._times_d, dist_pc=scene.dist_pc
        )
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=0)


def test_unmarked_spectrum_consistent_with_its_vmag_converts_silently(scene_file):
    """A converted spectrum that matches the header's VMAG passes the check.

    The VMAG written here is the AB magnitude of the star's own observer flux
    at the channel nearest 550 nm, so the converted spectrum agrees with it
    and the reader neither raises nor warns.
    """
    path = scene_file()
    with fits.open(path) as hdul:
        wave_um = np.asarray(hdul[0].data, dtype=np.float64)
    f_nu_jy = _phoenix_star_observer_jy(wave_um)
    v_channel = int(np.argmin(np.abs(wave_um - 0.55)))
    vmag = float(flux_jy_to_mag(f_nu_jy[v_channel]))
    _write_phoenix_star(path, {"VMAG": vmag})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = from_exovista(path, load_disk=False)

    wavelength_nm = wave_um[v_channel] * 1e3
    expected = f_nu_jy[v_channel] * Jy / (h * wavelength_nm)
    got = scene.star.photon_flux_density(
        wavelength_nm, scene.star._times_d, dist_pc=scene.dist_pc
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize(
    "updates",
    [{"RSTAR": None}, {"RSTAR": "NaN"}, {"DIST": "NaN"}, {"DIST": "Invalid tag"}],
)
def test_unmarked_spectrum_without_a_numeric_radius_or_distance_is_rejected(
    scene_file, updates
):
    """The dilution needs RSTAR and DIST as numbers; anything else is refused."""
    path = scene_file()
    _write_phoenix_star(path, updates)
    with pytest.raises(ValueError, match="RSTAR and DIST"):
        from_exovista(path, load_disk=False)


def test_vmag_check_is_skipped_when_no_channel_is_near_v(scene_file):
    """A grid that does not reach the V band is converted, with a warning.

    The VMAG here is far from the star's flux, so comparing it with the
    nearest (near-infrared) channel would wrongly reject a correct file.
    """
    path = scene_file()
    with fits.open(path, mode="update") as hdul:
        hdul[0].data[:] = [1.0, 1.1, 1.2]
    _write_phoenix_star(path, {"VMAG": 20.0})
    with pytest.warns(RuntimeWarning, match="without checking"):
        from_exovista(path, load_disk=False)


def test_unmarked_spectrum_that_contradicts_its_vmag_is_rejected(scene_file):
    """A spectrum already in Jy, left unmarked, is not converted a second time."""
    path = scene_file()
    with fits.open(path, mode="update") as hdul:
        header = hdul[5].header
        del header["SPECUNIT"]
        header["RSTAR"] = 1.0
        header["VMAG"] = 4.83
    with pytest.raises(ValueError, match="VMAG"):
        from_exovista(path, load_disk=False)


def test_unmarked_legacy_spectra_are_read_as_jy(scene_file):
    """Layouts before v2.5 predate the PHOENIX writer and stay in Jy."""
    path = scene_file(version=2.33)
    with fits.open(path, mode="update") as hdul:
        del hdul[4].header["SPECUNIT"]
    scene = from_exovista(path, load_disk=False)
    np.testing.assert_allclose(
        scene.star.photon_flux_density(
            600.0, scene.star._times_d[0], dist_pc=scene.dist_pc
        ),
        1e-26 / (h * 600.0),
        rtol=1e-6,
        atol=0,
    )


def test_declared_surface_flux_cannot_be_read_as_observer_jy(scene_file):
    """Reject explicitly incompatible units rather than reinterpret the numbers."""
    path = scene_file()
    with fits.open(path, mode="update") as hdul:
        hdul[5].header["SPECUNIT"] = "erg/s/cm^2/cm"
    with pytest.raises(ValueError, match="SPECUNIT must be Jy"):
        from_exovista(path, load_disk=False)


def test_from_exovista_places_the_scene_from_the_star_header(scene_file):
    """RA and DEC in the stellar header place the scene; without them it is unplaced."""
    path = scene_file()
    assert from_exovista(path, load_disk=False).ra_deg is None
    with fits.open(path, mode="update") as hdul:
        star_ext = next(i for i, h in enumerate(hdul) if "MASS" in h.header)
        hdul[star_ext].header["RA"] = 101.25
        hdul[star_ext].header["DEC"] = -16.7
    scene = from_exovista(path, load_disk=False)
    np.testing.assert_allclose(scene.ra_deg, 101.25)
    np.testing.assert_allclose(scene.dec_deg, -16.7)
    assert not hasattr(scene.star, "ra_deg")
