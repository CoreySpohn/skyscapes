"""ExoVista FITS loader -- builds a ``skyscapes.scene.System`` directly.

Mirrors the legacy ``skyscapes._legacy.loaders.from_exovista`` semantics
for Star and Disk, but rebuilds each planet as a 1-planet
``KeplerianOrbit`` + ``GridAtmosphere`` pair stored in the
``System.planets`` tuple. No ghost padding -- variadic tuples make that
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
    G_si,
    Mearth2kg,
    Msun2kg,
    mas2arcsec,
    um2nm,
)
from hwoutils.conversions import au_per_yr_to_m_per_s, decimal_year_to_jd
from orbix.equations.orbit import state_vector_to_keplerian
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.system.orbit import KeplerianOrbit

from ..atmosphere import GridAtmosphere
from ..disk import ExovistaDisk
from ..scene import Planet, SpectrumStar, System
from ._frames import rotate_to_sky_coords


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
    trig_solver,
    midplane_inc_deg: float,
    midplane_pa_deg: float,
) -> tuple[Planet, float]:
    """Load one planet and return ``(Planet, t0_d)``.

    The barycentric ``(r, v)`` state vectors from the FITS file are
    rotated into the on-sky frame at load time using the system midplane
    angles before being converted to Keplerian elements. This means
    ``KeplerianOrbit.propagate`` produces on-sky positions directly --
    no frame metadata is consulted at runtime.

    The contrast grid is indexed by phase angle beta = arccos(r_z / |r|),
    computed at load time with the same ``trig_solver`` the runtime
    ``System`` uses.
    """
    with open(fits_file, "rb") as f:
        obj_data, obj_header = getdata(f, ext=5 + idx, header=True, memmap=False)

    times_year = jnp.asarray(2000.0 + obj_data[:, 0])
    times_jd = decimal_year_to_jd(times_year)
    t0 = float(times_jd[0])

    contrast_data = jnp.asarray(obj_data[:, 16:].T.astype(np.float32))

    # Rotate barycentric state vectors into the sky frame BEFORE
    # computing Keplerian elements. FITS data is big-endian ('>f8');
    # convert to native float64 before passing to JAX.
    r_bary_au = jnp.asarray(obj_data[0:1, 9:12].astype(np.float64))  # (1, 3)
    v_bary_au_yr = jnp.asarray(obj_data[0:1, 12:15].astype(np.float64))  # (1, 3)
    r_sky_au = rotate_to_sky_coords(
        r_bary_au, inc_deg=midplane_inc_deg, pa_deg=midplane_pa_deg
    )[0]
    v_sky_au_yr = rotate_to_sky_coords(
        v_bary_au_yr, inc_deg=midplane_inc_deg, pa_deg=midplane_pa_deg
    )[0]

    r_sky_m = r_sky_au * AU2m
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

    # beta(t_i) per FITS input time, using the same solver the runtime uses.
    _r_AU, phase_angle_rad, _dist_AU = orbit.propagate(
        trig_solver, times_jd, Ms_kg=star.Ms_kg
    )
    beta_deg = jnp.rad2deg(phase_angle_rad[0])  # (T,), already in [0, 180]

    # Sort by beta. Duplicates are fine for interpolation (axisymmetric
    # atmosphere => same beta => same contrast).
    sort_idx = jnp.argsort(beta_deg)
    beta_sorted = beta_deg[sort_idx]
    contrast_sorted = contrast_data[:, sort_idx]

    # Regular beta axis on [0, 180].
    regular_grid = jnp.linspace(0.0, 180.0, 100)
    xq, yq = jnp.meshgrid(wavelengths_nm, regular_grid, indexing="ij")
    contrast_grid = interpax.interp2d(
        xq.flatten(),
        yq.flatten(),
        wavelengths_nm,
        beta_sorted,
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

    The contrast cube is already rendered in the on-sky frame by
    ExoVista; the system midplane angles (I, PA) live on
    ``System.midplane_inc_deg`` / ``System.midplane_pa_deg`` (populated
    by ``from_exovista`` from the same FITS star header), not on the
    disk itself.
    """
    with open(fits_file, "rb") as f:
        obj_data, header = getdata(f, ext=fits_ext, header=True, memmap=False)
        wavelengths_um = getdata(f, ext=fits_ext - 1, header=False, memmap=False)

    wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)
    contrast_cube = jnp.asarray(obj_data[:-1].astype(np.float32))
    pixel_scale_arcsec = header["PXSCLMAS"] * mas2arcsec

    return ExovistaDisk(
        pixel_scale_arcsec=pixel_scale_arcsec,
        wavelengths_nm=wavelengths_nm,
        contrast_cube=contrast_cube,
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

    The system's midplane inclination and position angle (FITS star
    header keys ``I`` and ``PA``) are applied at load time as a frame
    rotation of each planet's state vector before its Keplerian
    elements are computed. The same angles are stored as
    ``System.midplane_inc_deg`` and ``System.midplane_pa_deg`` for
    downstream diagnostic use.

    Args:
        fits_file: Path to ExoVista FITS file.
        planet_indices: Planet indices to load (0-based). ``None`` = all.
        only_earths: If True and *planet_indices* is None, auto-filter Earths.

    Returns:
        ``scene.System`` with star, planets (tuple), disk, and midplane metadata.
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
        star_header = getheader(f, ext=4, memmap=False)
    wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)

    midplane_inc_deg = float(star_header.get("I", 0.0))
    midplane_pa_deg = float(star_header.get("PA", 0.0))

    star = _load_star(fits_file, fits_ext=4)
    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    planets = tuple(
        _load_single_planet(
            fits_file,
            i,
            star,
            wavelengths_nm,
            solver,
            midplane_inc_deg=midplane_inc_deg,
            midplane_pa_deg=midplane_pa_deg,
        )[0]
        for i in planet_indices
    )
    disk = _load_disk(fits_file, disk_ext)

    return System(
        star=star,
        planets=planets,
        disk=disk,
        trig_solver=solver,
        midplane_inc_deg=midplane_inc_deg,
        midplane_pa_deg=midplane_pa_deg,
    )
