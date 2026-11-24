"""ExoVista Cartesian-state regressions against a Newtonian three-body solution."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.constants import AU2m, G, G_si, Mearth2kg, Msun2kg, d2s, yr2s
from hwoutils.conversions import au_per_yr_to_m_per_s
from orbix.equations.orbit import state_vector_to_keplerian
from orbix.system import FrameOrbit, phase_angle_rad
from scipy.integrate import solve_ivp

from skyscapes.io import exovista, from_exovista
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene.scene import _state_a_e
from tests.test_io_exovista_massive import (
    DIST_PC,
    EPOCH_YEARS,
    JUPITER_MEARTH,
    _rot_x,
    _rot_z,
)
from tests.test_io_exovista_massive import _write_fits as write_one_planet_fits

_STAR_MSUN = 2.5
_JUPITER_KG = JUPITER_MEARTH * Mearth2kg
_MASSES_KG = np.array([_STAR_MSUN * Msun2kg, _JUPITER_KG, _JUPITER_KG])
_DAYS_PER_YEAR = yr2s / d2s
_AXES = np.array([-1.0, 1.0, -1.0])


def _initial_state():
    """Build barycentric positions [AU] and velocities [AU/day] in file axes.

    Each planet starts on a specified star-relative osculating ellipse. The
    mutual inclination and stellar mass keep the actual three-body perturbation
    small over ten years. A common rotation gives both orbits visible sky and
    line-of-sight motion.
    """
    sky = _rot_z(np.deg2rad(35)) @ _rot_x(np.deg2rad(30)) @ _rot_z(np.deg2rad(25))
    outer_plane = _rot_z(np.deg2rad(45)) @ _rot_x(np.deg2rad(100))
    state = np.zeros((2, 3, 3))
    for j, (a, anomaly, rotation) in enumerate(
        [(1.9, 0.0, sky), (5.4, 40.0, sky @ outer_plane)], start=1
    ):
        e = 0.05
        p = a * (1.0 - e**2)
        nu = np.deg2rad(anomaly)
        c, s = np.cos(nu), np.sin(nu)
        mu = G * (_MASSES_KG[0] + _MASSES_KG[j])
        state[0, j] = rotation @ np.array([c, s, 0]) * (p / (1 + e * c))
        state[1, j] = rotation @ np.array([-s, e + c, 0]) * np.sqrt(mu / p)
    return state - np.average(state, axis=1, weights=_MASSES_KG)[:, None, :]


@pytest.fixture(scope="module")
def jacobi_file(tmp_path_factory):
    """Integrate all three interacting bodies and store float32 ExoVista rows."""

    def rhs(_time, flat_state):
        position, velocity = flat_state.reshape(2, 3, 3)
        separation = position[None, :, :] - position[:, None, :]
        distance = np.linalg.norm(separation, axis=-1)
        np.fill_diagonal(distance, np.inf)
        acceleration = G * np.sum(
            separation * (_MASSES_KG / distance**3)[:, :, None], axis=1
        )
        return np.stack([velocity, acceleration]).ravel()

    days = EPOCH_YEARS * _DAYS_PER_YEAR
    solution = solve_ivp(
        rhs,
        (days[0], days[-1]),
        _initial_state().ravel(),
        t_eval=days,
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
    )
    assert solution.success, solution.message
    position, velocity = solution.y.reshape(2, 3, 3, -1).transpose(0, 1, 3, 2)

    wavelengths = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    primary = fits.PrimaryHDU(wavelengths)
    primary.header["VERSION"] = 2.6
    hdus = [
        primary,
        fits.ImageHDU(wavelengths),
        fits.ImageHDU(),
        fits.ImageHDU(),
        fits.ImageHDU(np.zeros((len(days), 0, 4), dtype=np.float32)),
    ]
    star_header = {
        "MASS": _STAR_MSUN,
        "DIST": DIST_PC,
        "RSTAR": 2.0,
        "ANGDIAM": 1.86,
        "SPECUNIT": "Jy",
        "I": 30.0,
        "PA": 35.0,
    }
    planet_header = {"M": JUPITER_MEARTH, "R": 11.2, "A": 99.0, "E": 0.9}
    for j, header in enumerate([star_header, planet_header, planet_header]):
        rows = np.zeros((len(days), 16 + wavelengths.size), dtype=np.float32)
        rows[:, 0] = EPOCH_YEARS
        rows[:, 9:12] = position[j]
        rows[:, 12:15] = velocity[j] * _DAYS_PER_YEAR
        rows[:, 16:] = 1 if j == 0 else 1e-9 * (j + EPOCH_YEARS[:, None]) * wavelengths
        hdus.append(fits.ImageHDU(rows, header=fits.Header(header)))
    path = tmp_path_factory.mktemp("jacobi") / "jupiters.fits"
    fits.HDUList(hdus).writeto(path)
    return path


@pytest.fixture(scope="module")
def jacobi_scene(jacobi_file):
    """Load the two-planet file in its native inner-first order."""
    return from_exovista(jacobi_file, load_disk=False)


def _file_relative_states(path):
    """Read star-relative orbix-axis positions [AU] and velocities [AU/day]."""
    with fits.open(path) as hdul:
        states = np.stack([hdu.data[:, 9:15].astype(float) for hdu in hdul[5:]])
    relative = states[1:] - states[0]
    return relative[..., :3] * _AXES, relative[..., 3:] * _AXES / _DAYS_PER_YEAR


def test_first_epoch_reproduces_all_relative_states(jacobi_file, jacobi_scene):
    """Detect star-relative elements installed as Jacobi elements (outer reflex)."""
    position, velocity = _file_relative_states(jacobi_file)
    eph = jacobi_scene.ephemeris(np.array([0.0]))
    targets = eph.scene_planet_indices
    # Installing individual star-relative elements as Jacobi misses the inner
    # barycenter correction in both the outer position and velocity.
    for actual, expected in [
        (eph.position_AU, position),
        (eph.velocity_AU_per_d, velocity),
    ]:
        relative = actual[targets, 0] - actual[0, 0]
        np.testing.assert_allclose(relative, expected[:, 0], rtol=0, atol=1e-9)


def test_outer_first_offset_survives_inner_deselection(jacobi_file, jacobi_scene):
    """The outer planet starts at the same sky offset in a one-planet chain."""
    outer = from_exovista(jacobi_file, load_disk=False, planet_indices=[1])
    times = np.array([0.0])
    np.testing.assert_allclose(
        1000 * outer.positions(times)[:, 0],
        1000 * jacobi_scene.positions(times)[:, 1],
        rtol=0,
        atol=1e-6,
    )


def test_ten_year_sky_residuals(jacobi_file, jacobi_scene):
    """Measure the maximum absolute coordinate residual at all 41 stored dates."""
    position, _ = _file_relative_states(jacobi_file)
    expected = position[..., :2].transpose(2, 0, 1) / DIST_PC
    actual = jacobi_scene.positions(EPOCH_YEARS * _DAYS_PER_YEAR)
    residual_mas = np.max(np.abs(actual - expected), axis=(0, 2)) * 1000
    print("Planet      Max abs sky residual [mas]")
    for name, residual in zip(("inner", "outer"), residual_mas, strict=True):
        print(f"{name:5s}       {residual:.8f}")
    assert np.all(residual_mas < 0.05), residual_mas


def test_reverse_hdus_preserve_physical_planet_positions(
    jacobi_file, jacobi_scene, tmp_path
):
    """Removing the dynamical ordering changes the outer-first ephemeris."""
    path = tmp_path / "outer-first.fits"
    with fits.open(jacobi_file) as hdul:
        fits.HDUList([*hdul[:6], hdul[7], hdul[6]]).writeto(path)
    reverse = from_exovista(path, load_disk=False)
    times = EPOCH_YEARS * _DAYS_PER_YEAR
    # positions maps the chain ephemeris back into scene order via body_index.
    assert jacobi_scene.body_index == (0, 1)
    assert reverse.body_index == (1, 0)
    np.testing.assert_array_equal(
        reverse.positions(times)[:, ::-1, :], jacobi_scene.positions(times)
    )


def test_state_derived_elements_match_independent_osculating_ellipses(jacobi_file):
    """Check the state-based a and e against the ellipses used to initialize DOP853."""
    mu = G * (_MASSES_KG[0] + _MASSES_KG[1:])
    with fits.open(jacobi_file) as hdul:
        position, velocity = np.stack(
            [exovista._first_state(hdu.data[0]) for hdu in hdul[5:]], axis=1
        )
        a, e = _state_a_e(position[1:] - position[0], velocity[1:] - velocity[0], mu)
        print("Planet   State a [AU]    State e    Header A    Header E")
        for j, name in enumerate(("inner", "outer")):
            header = hdul[6 + j].header
            print(
                f"{name:5s}       {a[j]:.6f}   {e[j]:.6f}"
                f"   {header['A']:.1f}   {header['E']:.1f}"
            )
    # Treating AU/year as AU/day would make both osculating ellipses unbound
    # (negative a); the comparison below also catches it.
    assert np.isfinite(a).all() and np.isfinite(e).all() and (a > 0).all()
    # Compute the reference from the integrator's unrounded initial states
    # using specific orbital energy and angular momentum in the file frame.
    initial = _initial_state()
    r, v = initial[:, 1:] - initial[:, :1]
    energy = np.sum(v**2, axis=-1) / 2 - mu / np.linalg.norm(r, axis=-1)
    h_squared = np.sum(np.cross(r, v) ** 2, axis=-1)
    np.testing.assert_allclose(a, -mu / (2 * energy), atol=0, rtol=0.01)
    np.testing.assert_allclose(
        e, np.sqrt(1 + 2 * energy * h_squared / mu**2), atol=0, rtol=0.01
    )


def test_first_epoch_mismatch_is_rejected(jacobi_file, tmp_path):
    """A planet cannot enter a common Cartesian state at a different time."""
    path = tmp_path / "mismatched-epoch.fits"
    with fits.open(jacobi_file) as hdul:
        hdul[7].data[0, 0] += 0.125
        hdul.writeto(path)
    with pytest.raises(ValueError, match=r"(?i)(time|epoch)"):
        from_exovista(path, load_disk=False)


# A cold run compiles both the classical reference and the Cartesian phase grid.
@pytest.mark.medium
def test_one_planet_phase_samples_match_classical_reconstruction(tmp_path, monkeypatch):
    """Preserve the old phase samples when the header matches the stored state."""
    path = tmp_path / "one_planet.fits"
    write_one_planet_fits(path, JUPITER_MEARTH)
    with fits.open(path, mode="update") as hdul:
        star, planet = hdul[5], hdul[6]
        relative = planet.data[0, 9:15].astype(float) - star.data[0, 9:15].astype(float)
        r_m = jnp.asarray(relative[:3] * _AXES) * AU2m
        v_m_s = au_per_yr_to_m_per_s(jnp.asarray(relative[3:] * _AXES))
        mass_Msun = planet.header["M"] * Mearth2kg / Msun2kg
        mu_si = G_si * Msun2kg * (star.header["MASS"] + mass_Msun)
        a_m, e, inc, node, peri, mean = state_vector_to_keplerian(r_m, v_m_s, mu_si)

        # The old loader took A/E from the header. Match them to the actual
        # float32 FITS state so rounding does not give two different orbits.
        planet.header["A"] = float(a_m / AU2m)
        planet.header["E"] = float(e)
        times_d = planet.data[:, 0].astype(float) * (yr2s / d2s)
        base = FrameOrbit.from_classical(
            a_AU=planet.header["A"],
            e=planet.header["E"],
            i_rad=inc,
            W_rad=node,
            w_rad=peri,
            M0_rad=mean,
            epoch_d=times_d[0],
            mu_AU3_per_d2=mu_si * d2s**2 / AU2m**3,
        )
    positions, _ = base.at(times_d)
    expected_beta = phase_angle_rad(positions)

    phase_samples = []
    regrid = exovista._phase_model

    def record_phase_samples(model, beta_rad):
        phase_samples.append(np.asarray(beta_rad))
        return regrid(model, beta_rad)

    monkeypatch.setattr(exovista, "_phase_model", record_phase_samples)
    scene = from_exovista(path, load_disk=False, planet_spectra="phase")
    assert isinstance(scene.planets[0].physical_model, GridPhysicalModel)
    (actual_beta,) = phase_samples
    np.testing.assert_allclose(actual_beta, expected_beta, rtol=0, atol=1e-9)


def test_phase_grid_uses_each_planets_assembled_ephemeris(jacobi_file, tmp_path):
    """Reproduce the stored contrast grid from scene-order star-relative phases."""
    path = tmp_path / "different-cadences.fits"
    with fits.open(jacobi_file) as hdul:
        hdul[7].data = hdul[7].data[::2]
        hdul.writeto(path)
    scene = from_exovista(
        path, planet_indices=[1, 0], load_disk=False, planet_spectra="phase"
    )
    with fits.open(path) as hdul:
        for planet, body, hdu in zip(
            scene.planets, np.add(scene.body_index, 1), (hdul[7], hdul[6]), strict=True
        ):
            times_yr = hdu.data[:, 0].astype(float)
            times_d = (times_yr - times_yr[0]) * (yr2s / d2s)
            beta = np.rad2deg(scene.ephemeris(times_d).phase_angle_rad(body))
            order = np.argsort(beta)
            model = planet.physical_model
            grid = np.asarray(model.phase_angle_deg)
            covered = (beta.min() <= grid) & (grid <= beta.max())
            assert covered.any()
            expected = np.interp(grid[covered], beta[order], hdu.data[order, 16])
            np.testing.assert_allclose(
                model.contrast_grid[0, 0, covered], expected, rtol=2e-7, atol=0
            )
