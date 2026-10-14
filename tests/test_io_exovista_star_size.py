"""Stellar angular size of ExoVista scenes from either writer path.

ExoVista writes the same stellar header for both of its inputs, but the
ANGDIAM card means different things: a hand-written system file stores the
angular radius (0.465 mas per solar radius at 10 pc), while a catalog target
stores the angular diameter from the Boyajian et al. (2014) color relation and
derives RSTAR from it. RSTAR and DIST mean the same thing on both paths, so
the diameter comes from them, and a header without them is rejected.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import R_sun
from astropy.io import fits

from skyscapes.io import from_exovista

# Solar radius in AU used by ExoVista's catalog loader to derive RSTAR.
_EXOVISTA_RSUN_AU = 0.00465047


def _expected_diameter_arcsec(rstar_rsun, dist_pc):
    """Angular diameter 2 R / d of a star, computed independently in astropy."""
    angle = 2 * rstar_rsun * R_sun / (dist_pc * u.pc)
    return angle.to_value(u.arcsec, equivalencies=u.dimensionless_angles())


def _write_scene(path, star_header):
    """Write a one-planet legacy-layout scene with the given star header."""
    wave = np.array([0.5, 0.6, 0.7])
    primary = fits.PrimaryHDU(wave)
    primary.header["VERSION"] = 2.33
    primary.header["N_EXT"] = 5
    star = np.zeros((1, 19))
    star[:, 16:] = 1.0
    star_hdu = fits.ImageHDU(star)
    for key, value in {"MASS": 1.0, "SPECUNIT": "Jy", **star_header}.items():
        star_hdu.header[key] = value
    planet = np.zeros((1, 19))
    planet[:, 9] = 1.0
    planet[:, 13] = 2 * np.pi
    planet[:, 16:] = 1e-10
    planet_hdu = fits.ImageHDU(planet)
    for key, value in {"R": 1.0, "M": 1.0, "A": 1.0, "E": 0.0}.items():
        planet_hdu.header[key] = value
    hdus = [primary, fits.ImageHDU(wave), fits.ImageHDU(), fits.ImageHDU()]
    fits.HDUList([*hdus, star_hdu, planet_hdu]).writeto(path)
    return str(path)


@pytest.mark.parametrize(("rstar", "dist"), [(1.0, 10.0), (0.8, 3.2), (1.7, 25.0)])
def test_system_file_star_has_full_angular_diameter(tmp_path, rstar, dist):
    """A system-file ANGDIAM is a radius; the Star still gets the diameter."""
    header = {"RSTAR": rstar, "DIST": dist, "ANGDIAM": 0.465 * rstar / (dist / 10.0)}
    scene = from_exovista(_write_scene(tmp_path / "s.fits", header), load_disk=False)
    expected = _expected_diameter_arcsec(rstar, dist)
    np.testing.assert_allclose(scene.star.diameter_arcsec, expected, rtol=1e-6)
    # Twice the stored ANGDIAM; the tolerance covers ExoVista's rounded 0.465 mas.
    np.testing.assert_allclose(
        scene.star.diameter_arcsec, 2 * header["ANGDIAM"] / 1e3, rtol=1e-3
    )


def test_sun_at_ten_parsecs_spans_0p93_mas(tmp_path):
    """The Sun at 10 pc from a system file, whose ANGDIAM card reads 0.465."""
    header = {"RSTAR": 1.0, "DIST": 10.0, "ANGDIAM": 0.465}
    scene = from_exovista(_write_scene(tmp_path / "s.fits", header), load_disk=False)
    assert float(scene.star.diameter_arcsec) == pytest.approx(9.3009e-4, rel=1e-4)


def test_system_file_star_without_a_radius_is_a_point(tmp_path):
    """ExoVista's default system-file RSTAR is 0, with ANGDIAM 0 to match."""
    header = {"RSTAR": 0.0, "DIST": 10.0, "ANGDIAM": 0.0}
    scene = from_exovista(_write_scene(tmp_path / "p.fits", header), load_disk=False)
    assert float(scene.star.diameter_arcsec) == 0.0


@pytest.mark.parametrize(("theta_mas", "dist"), [(0.93, 10.0), (2.4, 3.6)])
def test_catalog_star_keeps_its_angular_diameter(tmp_path, theta_mas, dist):
    """A catalog ANGDIAM is a diameter and RSTAR is derived from it.

    The 1e-6 tolerance covers the 5.9e-7 offset between ExoVista's solar
    radius in AU and astropy's.
    """
    rstar = dist * theta_mas / 1000.0 / _EXOVISTA_RSUN_AU / 2.0
    header = {"RSTAR": rstar, "DIST": dist, "ANGDIAM": theta_mas}
    scene = from_exovista(_write_scene(tmp_path / "c.fits", header), load_disk=False)
    expected = _expected_diameter_arcsec(rstar, dist)
    np.testing.assert_allclose(scene.star.diameter_arcsec, expected, rtol=1e-6)
    np.testing.assert_allclose(scene.star.diameter_arcsec, theta_mas / 1e3, rtol=1e-6)


@pytest.mark.parametrize(
    "updates",
    [
        {"RSTAR": None},
        {"RSTAR": "NaN"},
        {"RSTAR": "Invalid tag"},
        {"RSTAR": -1.0},
        {"DIST": "NaN"},
        {"DIST": 0.0},
    ],
)
def test_star_without_a_usable_radius_or_distance_is_rejected(tmp_path, updates):
    """ANGDIAM is a radius or a diameter by writer path, so it is no fallback."""
    header = {"RSTAR": 1.0, "DIST": 10.0, "ANGDIAM": 0.465}
    for key, value in updates.items():
        if value is None:
            del header[key]
        else:
            header[key] = value
    path = _write_scene(tmp_path / "f.fits", header)
    with pytest.raises(ValueError, match="RSTAR"):
        from_exovista(path, load_disk=False)
