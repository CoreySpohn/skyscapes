"""Import contracts for the legacy and spotted-star ExoVista layouts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.constants import AU2m, G_si, Mearth2kg, Msun2kg, yr2s

from skyscapes.io import from_exovista, get_earth_like_planet_indices


@pytest.fixture
def scene_file(tmp_path):
    """Write small scenes with distinct planet spectra at identical phases."""

    def write(*, version=2.6, n_times=3, n_spots=2):
        wave = np.array([0.5, 0.6, 0.7])
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
            "LSTAR": 1.0,
            "I": 0.0,
            "PA": 0.0,
            "TEFF": 5778.0,
            "SPECUNIT": "Jy",
            "HIERARCH TeffUmbra": 2500.0,
            "HIERARCH TeffPenumbra": 2700.0,
        }.items():
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
        path = tmp_path / f"scene-{version}-{n_times}-{n_spots}.fits"
        fits.HDUList(hdus).writeto(path)
        return str(path)

    return write


@pytest.mark.parametrize("version", [2.33, 2.5, 2.6])
def test_layout_selects_actual_bodies_and_earths(scene_file, version):
    """Header identity determines body count and selection, not N_EXT."""
    path = scene_file(version=version)
    scene = from_exovista(path, load_disk=False)
    assert scene.n_planets == 2
    assert float(scene.star.diameter_arcsec) == pytest.approx(0.0009)
    assert get_earth_like_planet_indices(path) == [0]
    assert from_exovista(path, only_earths=True, load_disk=False).n_planets == 1
    selected = from_exovista(path, planet_indices=[1, 0], load_disk=False)
    np.testing.assert_array_equal([p.Rp_Rearth[0] for p in selected.planets], [11, 1])


def test_new_layout_preserves_spot_history_and_stellar_metadata(scene_file):
    """Spot areas convert from cm^2 to m^2; temperatures survive import."""
    scene = from_exovista(scene_file(), load_disk=False)
    spots = scene.star.spots
    assert spots.latitude_deg.shape == (3, 2)
    np.testing.assert_array_equal(spots.longitude_deg, 25.0)
    np.testing.assert_allclose(spots.area_m2, 1e14, rtol=1e-7, atol=0)
    np.testing.assert_array_equal(spots.area_ratio, 5.0)
    assert spots.umbra_temperature_k == 2500
    assert spots.penumbra_temperature_k == 2700
    assert scene.star.metadata["TEFF"] == 5778


def test_new_layout_keeps_time_dependent_contrast_at_equal_phase(scene_file):
    """Phase-only regridding must not erase a factor-three illumination change."""
    scene = from_exovista(scene_file(), load_disk=False)
    times = scene.star._times_jd
    np.testing.assert_allclose(
        scene.contrasts(600.0, times)[0], [1e-10, 2e-10, 3e-10], rtol=1e-6, atol=0
    )
    # At the table knots: Jy -> photons = 1e-26 / (h * wavelength_nm).
    expected = np.array([1, 4, 9]) * 1e-10 * 1e-26 / (6.62607015e-34 * 600.0)
    np.testing.assert_allclose(
        scene.planet_flux_densities(600.0, times)[0], expected, rtol=1e-6, atol=0
    )


def test_single_snapshot_and_empty_spots_are_supported(scene_file):
    """A snapshot is constant in time; zero spots is distinct from no metadata."""
    scene = from_exovista(scene_file(n_times=1, n_spots=0), load_disk=False)
    assert scene.star.spots.latitude_deg.shape == (1, 0)
    times = scene.star._times_jd + jnp.array([0.0, 1.0])
    np.testing.assert_allclose(
        scene.star.spec_flux_density(600.0, times),
        1e-26 / (6.62607015e-34 * 600.0),
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_allclose(
        scene.contrasts(600.0, times)[0], 1e-10, rtol=1e-6, atol=0
    )


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
    expected = 2451545.0 + np.array([0.0, 182.625, 365.25])
    np.testing.assert_allclose(scene.star._times_jd, expected, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(scene.star.spots.times_jd, expected)
    for planet in scene.planets:
        np.testing.assert_array_equal(planet.physical_model.times_jd, expected)
        assert float(planet.orbit.t0_d[0]) == expected[0]


def test_imported_positive_flux_cannot_overshoot_below_zero(scene_file):
    """A deep positive flux minimum must not become negative between samples."""
    path = scene_file(n_times=4)
    with fits.open(path, mode="update") as hdul:
        hdul[5].data[:, 16:] = np.array([1.0, 0.01, 0.01, 1.0])[:, None]
    scene = from_exovista(path, load_disk=False)
    midpoint = jnp.mean(scene.star._times_jd[1:3])
    flux = scene.star.spec_flux_density(600.0, midpoint)
    expected = 0.01 * 1e-26 / (6.62607015e-34 * 600.0)
    np.testing.assert_allclose(flux, expected, rtol=1e-6, atol=0)
    assert np.all(np.asarray(scene.planet_flux_densities(600.0, midpoint)) > 0)


def test_import_requires_precision_for_absolute_julian_dates(scene_file):
    """Fail explicitly instead of silently merging closely spaced epochs."""
    path = scene_file()
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="jax_enable_x64"):
            from_exovista(path, load_disk=False)


def test_unmarked_recent_spectra_warn_about_known_writer_units(scene_file):
    """Old spotted scenes cannot silently inherit corrected-writer provenance."""
    path = scene_file()
    with fits.open(path, mode="update") as hdul:
        del hdul[5].header["SPECUNIT"]
    with pytest.warns(RuntimeWarning, match="surface flux"):
        from_exovista(path, load_disk=False)


def test_declared_surface_flux_cannot_be_read_as_observer_jy(scene_file):
    """Reject explicitly incompatible units rather than reinterpret the numbers."""
    path = scene_file()
    with fits.open(path, mode="update") as hdul:
        hdul[5].header["SPECUNIT"] = "erg/s/cm^2/cm"
    with pytest.raises(ValueError, match="SPECUNIT must be Jy"):
        from_exovista(path, load_disk=False)
