"""Import contracts for the legacy and spotted-star ExoVista layouts."""

import warnings

import jax
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
    h,
    k_B,
    pc2m,
    yr2s,
)
from hwoutils.conversions import flux_jy_to_mag

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
    expected = np.array([1, 4, 9]) * 1e-10 * 1e-26 / (h * 600.0)
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
        1e-26 / (h * 600.0),
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
    expected = 0.01 * 1e-26 / (h * 600.0)
    np.testing.assert_allclose(flux, expected, rtol=1e-6, atol=0)
    assert np.all(np.asarray(scene.planet_flux_densities(600.0, midpoint)) > 0)


def test_import_requires_precision_for_absolute_julian_dates(scene_file):
    """Fail explicitly instead of silently merging closely spaced epochs."""
    path = scene_file()
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="jax_enable_x64"):
            from_exovista(path, load_disk=False)


def _writer_mask_area(mapsize=512, u1=0.0473, u2=0.0841):
    """Area of the ExoVista v2.5/v2.6 limb-darkening mask, from its definition."""
    j, k = np.indices((mapsize, mapsize))
    half = mapsize / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.sqrt(half**2 - (j - half) ** 2 - (k - half) ** 2) / half
        mu2 = np.maximum(1 - mu, 0)
        mask = np.maximum((1 - (u1 + 1) * mu2 - u2 * mu2**2) / mu, 0)
    return np.nan_to_num(mask).sum() * 4 / np.pi / mapsize**2


def _blackbody_surface_flux(wavelength_um, temperature_k):
    """Emergent surface flux pi B_lambda in W m^-2 m^-1."""
    lam = wavelength_um * 1e-6
    x = h * c / (lam * k_B * temperature_k)
    return np.pi * 2 * h * c**2 / lam**5 / np.expm1(x)


_PHOENIX_STAR = {"temperature_k": 5800.0, "radius_rsun": 1.2, "dist_pc": 15.0}


def _write_phoenix_star(path, header_updates=None):
    """Store a blackbody star the way the v2.5 and v2.6 writers do, unmarked.

    Returns the channel wavelengths in microns and the star's observer flux
    density in Jy at each, computed from the Planck function alone.
    """
    star = _PHOENIX_STAR
    with fits.open(path, mode="update") as hdul:
        wave_um = np.asarray(hdul[0].data, dtype=np.float64)
        surface_cgs = _blackbody_surface_flux(wave_um, star["temperature_k"]) * 10.0
        hdul[5].data[:, 16:] = _writer_mask_area() * surface_cgs[None, :]
        header = hdul[5].header
        del header["SPECUNIT"]
        header["RSTAR"] = star["radius_rsun"]
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
    radius_m = star["radius_rsun"] * R_sun.to_value("m")
    dilution = (radius_m / (star["dist_pc"] * pc2m)) ** 2
    f_nu_jy = _blackbody_surface_flux(wave_um, star["temperature_k"]) * lam**2 / c
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
        got = scene.star.spec_flux_density(wavelength_nm, scene.star._times_jd)
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
    got = scene.star.spec_flux_density(wavelength_nm, scene.star._times_jd)
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
        scene.star.spec_flux_density(600.0, scene.star._times_jd[0]),
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
