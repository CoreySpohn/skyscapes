"""A Jupiter-mass ExoVista system keeps its sky positions at later epochs.

The FITS states come from a direct numerical integration of the two-body
problem for both bodies about their barycenter (Murray and Dermott 1999,
Solar System Dynamics, ch. 2, eqs. 2.1 to 2.4), not from orbix's orbit
matrices or propagator. The relative orbit of two bodies has the
gravitational parameter G (M_star + M_planet), so a loader that derives the
orbit from a state vector and a propagator that advances it must use the same
sum. Using G M_star at runtime makes a Jupiter-mass planet lag its true
position by about 0.145 degrees after ten years.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.constants import (
    J2000_JD,
    AU2m,
    Mearth2kg,
    Msun2kg,
    d2s,
    yr2s,
)
from hwoutils.constants import G as G_AU3_PER_KG_D2
from orbix.kepler import markley_solve_trig
from scipy.integrate import solve_ivp

from skyscapes.io import from_exovista
from skyscapes.scene import Planet

# The same value of G as the library, in SI units.
G_SI = G_AU3_PER_KG_D2 * AU2m**3 / d2s**2

DIST_PC = 10.0
JUPITER_MEARTH = 317.83
A_AU, ECC = 5.2, 0.048
EPOCH_YEARS = np.linspace(0.0, 10.0, 41)


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _initial_relative_state(mu):
    """Planet position and velocity [m, m/s] at true anomaly 40 deg.

    Perifocal state from the vis-viva and orbit equations (Murray and
    Dermott 1999, eqs. 2.20 and 2.122), rotated by 3-1-3 angles (their
    eq. 2.119) into an inclined orbit in the (x, y, z) = (west, north,
    toward) chart.
    """
    a = A_AU * AU2m
    nu = np.deg2rad(40.0)
    p = a * (1.0 - ECC**2)
    r = p / (1.0 + ECC * np.cos(nu))
    r_pf = r * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), ECC + np.cos(nu), 0.0])
    rot = (
        _rot_z(np.deg2rad(71.0)) @ _rot_x(np.deg2rad(52.0)) @ _rot_z(np.deg2rad(113.0))
    )
    return rot @ r_pf, rot @ v_pf


def _integrate_two_bodies(m_star, m_planet):
    """Barycentric positions and velocities of both bodies at EPOCH_YEARS."""
    mu = G_SI * (m_star + m_planet)
    r_rel, v_rel = _initial_relative_state(mu)
    f_star = -m_planet / (m_star + m_planet)
    f_planet = m_star / (m_star + m_planet)
    y0 = np.concatenate(
        [f_planet * r_rel, f_star * r_rel, f_planet * v_rel, f_star * v_rel]
    )

    def rhs(_t, y):
        r_p, r_s, v_p, v_s = y[:3], y[3:6], y[6:9], y[9:]
        sep = r_s - r_p
        dist3 = np.linalg.norm(sep) ** 3
        a_p = G_SI * m_star * sep / dist3
        a_s = -G_SI * m_planet * sep / dist3
        return np.concatenate([v_p, v_s, a_p, a_s])

    sol = solve_ivp(
        rhs,
        (0.0, EPOCH_YEARS[-1] * yr2s),
        y0,
        t_eval=EPOCH_YEARS * yr2s,
        method="DOP853",
        rtol=1e-13,
        atol=1e-3,
    )
    assert sol.success
    y = sol.y.T
    return y[:, :3], y[:, 3:6], y[:, 6:9], y[:, 9:]


def _write_fits(path, m_planet_earth):
    m_star = Msun2kg
    m_planet = m_planet_earth * Mearth2kg
    r_p, r_s, v_p, v_s = _integrate_two_bodies(m_star, m_planet)
    to_au = 1.0 / AU2m
    to_au_yr = yr2s / AU2m

    wavelengths = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    primary = fits.PrimaryHDU(wavelengths)
    primary.header["VERSION"] = 2.6
    n_t = len(EPOCH_YEARS)
    star = np.zeros((n_t, 19), dtype=np.float32)
    star[:, 0] = EPOCH_YEARS
    star[:, 9:12] = r_s * to_au
    star[:, 12:15] = v_s * to_au_yr
    star[:, 16:] = 1.0
    planet = np.zeros_like(star)
    planet[:, 0] = EPOCH_YEARS
    planet[:, 9:12] = r_p * to_au
    planet[:, 12:15] = v_p * to_au_yr
    planet[:, 16:] = 1e-10
    star_header = fits.Header(
        {
            "MASS": 1.0,
            "DIST": DIST_PC,
            "RSTAR": 1.0,
            "ANGDIAM": 0.93,
            "SPECUNIT": "Jy",
            "I": 54.0,
            "PA": 32.0,
        }
    )
    hdus = [
        primary,
        fits.ImageHDU(wavelengths),
        fits.ImageHDU(),
        fits.ImageHDU(),
        fits.ImageHDU(np.zeros((n_t, 0, 4), dtype=np.float32)),
        fits.ImageHDU(star, header=star_header),
        fits.ImageHDU(
            planet,
            header=fits.Header({"A": A_AU, "E": ECC, "M": m_planet_earth, "R": 11.2}),
        ),
    ]
    fits.HDUList(hdus).writeto(path)
    # Sky offsets of the planet from the star: (east, north) = (-x, y).
    rel = (r_p - r_s) * to_au
    return np.stack([-rel[:, 0], rel[:, 1]]) / DIST_PC


@pytest.fixture(scope="module")
def jupiter_system(tmp_path_factory):
    """Load the integrated Jupiter-mass system and its true sky offsets."""
    path = tmp_path_factory.mktemp("massive") / "jupiter.fits"
    expected = _write_fits(path, JUPITER_MEARTH)
    return from_exovista(path, load_disk=False, planet_spectra="time"), expected


def test_massive_planet_positions_follow_the_two_body_integration(jupiter_system):
    """Sky positions at 41 epochs over ten years match the integrated orbit."""
    system, expected = jupiter_system
    times = J2000_JD + EPOCH_YEARS * (yr2s / d2s)
    positions = system.planets[0].position_arcsec(
        markley_solve_trig, times, star=system.star
    )
    error = np.abs(np.asarray(positions[:, 0, :]) - expected)
    # float32 FITS storage allows ~1e-6 arcsec; the omitted planet mass in the
    # runtime parameter gave 1.3e-3 arcsec by year ten.
    assert error.max() < 2e-6


def test_unknown_planet_mass_falls_back_to_the_stellar_mass(jupiter_system):
    """NaN, zero or negative masses add nothing; known masses add G M_planet."""
    system, _ = jupiter_system
    base = system.planets[0]
    masses = jnp.array([JUPITER_MEARTH, jnp.nan, 0.0, -3.0])
    planet = Planet(
        Rp_Rearth=jnp.ones(4),
        Mp_Mearth=masses,
        orbit=base.orbit,
        physical_model=base.physical_model,
    )
    total = jax.jit(lambda p: p.central_mass_kg(star=system.star))(planet)
    ms = float(system.star.Ms_kg)
    np.testing.assert_allclose(total[0], ms + JUPITER_MEARTH * Mearth2kg, rtol=1e-14)
    np.testing.assert_allclose(total[1:], ms, rtol=1e-14)
