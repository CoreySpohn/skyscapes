"""ExoVista FITS loader — builds a ``skyscapes.scene.System`` directly.

Mirrors the legacy ``skyscapes._legacy.loaders.from_exovista`` semantics
for Star and Disk, but rebuilds each planet as a 1-planet
``KeplerianOrbit`` + ``GridAtmosphere`` pair stored in the
``System.planets`` tuple. No ghost padding — variadic tuples make that
unnecessary.
"""

from __future__ import annotations

from collections.abc import Sequence

import interpax
import jax.numpy as jnp
import numpy as np
from astropy.io.fits import getdata, getheader
from hwoutils.constants import (
    AU2m,
    G,
    G_si,
    Mearth2kg,
    Msun2kg,
    mas2arcsec,
    two_pi,
    um2nm,
)
from hwoutils.conversions import au_per_yr_to_m_per_s, decimal_year_to_jd
from orbix.equations.orbit import (
    mean_anomaly_tp,
    mean_motion,
    period_n,
    state_vector_to_keplerian,
)
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.system.orbit import KeplerianOrbit

from ..atmosphere import GridAtmosphere
from ..disk import ExovistaDisk
from ..scene import Planet, SpectrumStar, System


def _load_star(fits_file: str, fits_ext: int = 4) -> SpectrumStar:
    """Load the FITS star extension into a SpectrumStar."""
    with open(fits_file, "rb") as f:
        obj_data, obj_header = getdata(f, ext=fits_ext, header=True, memmap=False)
        wavelengths_um = getdata(f, ext=0, header=False, memmap=False)

    wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)
    times_year = jnp.asarray(2000.0 + obj_data[:, 0])
    times_jd = decimal_year_to_jd(times_year)
    flux_density_jy = jnp.asarray(obj_data[:, 16:].T.astype(np.float32))

    diameter_arcsec = obj_header["ANGDIAM"] * mas2arcsec
    Ms_kg = obj_header.get("MASS") * Msun2kg
    dist_pc = obj_header.get("DIST")
    ra_deg = obj_header.get("RA", 0.0)
    dec_deg = obj_header.get("DEC", 0.0)
    luminosity_lsun = obj_header.get("LSTAR", 1.0)

    return SpectrumStar(
        Ms_kg=Ms_kg,
        dist_pc=dist_pc,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        diameter_arcsec=diameter_arcsec,
        luminosity_lsun=luminosity_lsun,
        wavelengths_nm=wavelengths_nm,
        times_jd=times_jd,
        flux_density_jy=flux_density_jy,
    )


def _load_single_planet(
    fits_file: str,
    idx: int,
    star: SpectrumStar,
    wavelengths_nm: jnp.ndarray,
) -> tuple[Planet, float]:
    """Load one planet and return ``(Planet, t0_d)``.

    WARNING — ExoVista frame convention (see Plan-2 "Known Limitations"):
    ExoVista FITS columns 9-11 / 12-14 are **barycentric** position/velocity
    vectors, NOT on-sky vectors. Header orbital elements (A, E, INC, LAN, ARGP)
    are expressed in the system **midplane** frame. This loader preserves the
    legacy behavior by passing those vectors straight through, which is
    bit-exact vs. the legacy skyscapes Planet for systems where the midplane
    coincides with the sky frame. When it doesn't, downstream orbit
    propagators (orbix, EXOSIMS) will silently place the planet at the wrong
    (x, y) on sky. Plan 5 fixes this by applying a midplane→sky rotation
    (cf. ``exoverses.exovista.system.ExovistaSystem(convert=True)`` and
    ``exoverses.util.misc.gen_rotate_to_sky_coords``) and recomputing
    (Ω, i, ω, M) from the rotated state vectors before constructing the
    Keplerian orbit.
    """
    with open(fits_file, "rb") as f:
        obj_data, obj_header = getdata(f, ext=5 + idx, header=True, memmap=False)

    times_year = jnp.asarray(2000.0 + obj_data[:, 0])
    times_jd = decimal_year_to_jd(times_year)
    t0 = float(times_jd[0])

    contrast_data = jnp.asarray(obj_data[:, 16:].T.astype(np.float32))

    r_sky_au = obj_data[0, 9:12]
    v_sky_au_yr = obj_data[0, 12:15]
    r_sky_m = jnp.array(r_sky_au * AU2m)
    v_sky_m_s = jnp.array(au_per_yr_to_m_per_s(v_sky_au_yr))
    planet_mass_kg = float(obj_header.get("M")) * Mearth2kg
    mu_SI = G_si * (star.Ms_kg + planet_mass_kg)
    _a_m, _e, i_rad, W_rad, w_rad, M_rad = state_vector_to_keplerian(
        r_sky_m, v_sky_m_s, mu_SI
    )

    a_AU = float(obj_header.get("A"))
    e_val = float(obj_header.get("E"))

    orbit = KeplerianOrbit(
        a_AU=jnp.array([a_AU]),
        e=jnp.array([e_val]),
        W_rad=jnp.array([float(W_rad)]),
        i_rad=jnp.array([float(i_rad)]),
        w_rad=jnp.array([float(w_rad)]),
        M0_rad=jnp.array([float(M_rad)]),
        t0_d=jnp.array([t0]),
    )

    n = mean_motion(orbit.a_AU, G * star.Ms_kg)
    T_d = period_n(n)
    tp_d = orbit.t0_d - T_d * orbit.M0_rad / two_pi
    M_at_t = mean_anomaly_tp(times_jd, n[0], tp_d[0]) % two_pi
    mean_anom_deg = jnp.rad2deg(M_at_t)

    sort_idx = jnp.argsort(mean_anom_deg)
    mean_anom_sorted = mean_anom_deg[sort_idx]
    contrast_sorted = contrast_data[:, sort_idx]
    regular_grid = jnp.linspace(0.0, 360.0, 100)
    xq, yq = jnp.meshgrid(wavelengths_nm, regular_grid, indexing="ij")
    contrast_grid = interpax.interp2d(
        xq.flatten(),
        yq.flatten(),
        wavelengths_nm,
        mean_anom_sorted,
        contrast_sorted,
        method="linear",
        extrap=True,
    ).reshape(xq.shape)

    atmosphere = GridAtmosphere(
        Rp_Rearth=jnp.array([float(obj_header.get("R"))]),
        wavelengths_nm=wavelengths_nm,
        phase_angle_deg=regular_grid,
        contrast_grid=contrast_grid[None, ...],
    )

    return Planet(orbit=orbit, atmosphere=atmosphere), t0


def _load_disk(fits_file: str, fits_ext: int) -> ExovistaDisk:
    """Load the ExoVista disk extension into an ExovistaDisk.

    Records the system midplane (PA + I from the star header) on the disk as
    metadata. The contrast cube itself is already rendered in the on-sky frame
    by ExoVista, so no projection is applied here — the fields exist so that
    planet loaders (or future Plan-5 frame-conversion code) can align
    midplane-frame orbital elements with the disk's on-sky geometry.
    """
    with open(fits_file, "rb") as f:
        obj_data, header = getdata(f, ext=fits_ext, header=True, memmap=False)
        wavelengths_um = getdata(f, ext=fits_ext - 1, header=False, memmap=False)
        _, star_header = getdata(f, ext=4, header=True, memmap=False)

    wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)
    contrast_cube = jnp.asarray(obj_data[:-1].astype(np.float32))
    pixel_scale_arcsec = header["PXSCLMAS"] * mas2arcsec
    inclination_deg = float(star_header.get("I", 0.0))
    position_angle_deg = float(star_header.get("PA", 0.0))

    return ExovistaDisk(
        pixel_scale_arcsec=pixel_scale_arcsec,
        wavelengths_nm=wavelengths_nm,
        contrast_cube=contrast_cube,
        inclination_deg=inclination_deg,
        position_angle_deg=position_angle_deg,
    )


def get_earth_like_planet_indices(fits_file: str) -> list[int]:
    """Identify Earth-like planets in an ExoVista FITS file.

    Classification criteria:
      - Scaled semi-major axis: 0.95 <= a / sqrt(L_star) < 1.67 AU
      - Planet radius: 0.8 / sqrt(a_scaled) <= R < 1.4 R_earth
    """
    with open(fits_file, "rb") as f:
        h = getheader(f, ext=0, memmap=False)
        _, star_header = getdata(f, ext=4, header=True, memmap=False)

    n_ext = h["N_EXT"]
    n_planets_total = n_ext - 4
    star_luminosity_lsun = star_header.get("LSTAR", 1.0)

    earth_indices: list[int] = []
    for i in range(n_planets_total):
        with open(fits_file, "rb") as f:
            _, planet_header = getdata(f, ext=5 + i, header=True, memmap=False)
        a_au = planet_header.get("A", 1.0)
        radius_rearth = planet_header.get("R", 1.0)
        a_scaled = a_au / np.sqrt(star_luminosity_lsun)
        lower_r = 0.8 / np.sqrt(a_scaled)
        if (0.95 <= a_scaled < 1.67) and (lower_r <= radius_rearth < 1.4):
            earth_indices.append(i)
    return earth_indices


def from_exovista(
    fits_file: str,
    planet_indices: Sequence[int] | None = None,
    only_earths: bool = False,
) -> System:
    """Load an ExoVista FITS file into a ``scene.System``.

    Args:
        fits_file: Path to ExoVista FITS file.
        planet_indices: Planet indices to load (0-based). ``None`` = all.
        only_earths: If True and *planet_indices* is None, auto-filter Earths.

    Returns:
        ``scene.System`` with star, planets (tuple), and disk.
    """
    disk_ext = 2

    with open(fits_file, "rb") as f:
        h = getheader(f, ext=0, memmap=False)
    n_ext = h["N_EXT"]
    n_planets_total = n_ext - 4

    if planet_indices is None:
        if only_earths:
            planet_indices = get_earth_like_planet_indices(fits_file)
        else:
            planet_indices = list(range(n_planets_total))

    with open(fits_file, "rb") as f:
        wavelengths_um = getdata(f, ext=0, header=False, memmap=False)
    wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)

    star = _load_star(fits_file, fits_ext=4)
    planets = tuple(
        _load_single_planet(fits_file, i, star, wavelengths_nm)[0]
        for i in planet_indices
    )
    disk = _load_disk(fits_file, disk_ext)

    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    return System(star=star, planets=planets, disk=disk, trig_solver=solver)
