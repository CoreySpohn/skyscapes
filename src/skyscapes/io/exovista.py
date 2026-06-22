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
from astropy.io import fits
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

from ..disk import ExovistaDisk
from ..physical_model import GridPhysicalModel
from ..scene import Planet, Star, System


def _load_star(hdul: fits.HDUList, fits_ext: int = 4) -> Star:
    """Load the FITS star extension into a Star."""
    obj_data = hdul[fits_ext].data
    obj_header = hdul[fits_ext].header
    wavelengths_um = hdul[0].data

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

    return Star(
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
    hdul: fits.HDUList,
    idx: int,
    star: Star,
    wavelengths_nm: jnp.ndarray,
    trig_solver,
) -> tuple[Planet, float]:
    """Load one planet and return ``(Planet, t0_d)``.

    ExoVista stores the planet ``(r, v)`` state vectors already in the
    on-sky ("bary-sky") frame -- the same frame the disk cube is rendered
    in -- so they are converted to Keplerian elements directly, with NO
    additional sky rotation. Rotating them again (as an earlier version
    did) double-applies the system inclination and PA and tilts every
    planetary system off its disk.

    The contrast grid is indexed by phase angle beta = arccos(r_z / |r|),
    computed at load time with the same ``trig_solver`` the runtime
    ``System`` uses; this stays self-consistent with rendering because the
    grid is keyed by the same ``orbit.propagate`` beta used at render time.
    """
    obj_data = hdul[5 + idx].data
    obj_header = hdul[5 + idx].header

    times_year = jnp.asarray(2000.0 + obj_data[:, 0])
    times_jd = decimal_year_to_jd(times_year)
    t0 = float(times_jd[0])

    contrast_data = jnp.asarray(obj_data[:, 16:].T.astype(np.float32))

    # ExoVista's state vectors are already in the on-sky ("bary-sky") frame;
    # use them directly. FITS data is big-endian ('>f8'); convert to native
    # float64 before passing to JAX.
    r_sky_au = jnp.asarray(obj_data[0, 9:12].astype(np.float64))  # (3,)
    v_sky_au_yr = jnp.asarray(obj_data[0, 12:15].astype(np.float64))  # (3,)

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

    # Contrast is a flux ratio and cannot be negative. Linear extrapolation past
    # the orbit's sampled beta coverage (extrap=True above) can dip a few 1e-11
    # below zero near beta = 0/180; clip so the stored grid stays physical for
    # every consumer (contrast, alpha_dMag, image rendering).
    contrast_grid = jnp.clip(contrast_grid, 0.0, None)

    physical_model = GridPhysicalModel(
        wavelengths_nm=wavelengths_nm,
        phase_angle_deg=regular_grid,
        contrast_grid=contrast_grid[None, ...],
    )

    return (
        Planet(
            Rp_Rearth=jnp.array([float(obj_header.get("R"))]),
            Mp_Mearth=jnp.array([float(obj_header.get("M"))]),
            orbit=orbit,
            physical_model=physical_model,
        ),
        t0,
    )


def _load_disk(hdul: fits.HDUList, fits_ext: int) -> ExovistaDisk:
    """Load the ExoVista disk extension into an ExovistaDisk.

    The contrast cube is already rendered in the on-sky frame by
    ExoVista; the system midplane angles (I, PA) live on
    ``System.midplane_inc_deg`` / ``System.midplane_pa_deg`` (populated
    by ``from_exovista`` from the same FITS star header), not on the
    disk itself.
    """
    obj_data = hdul[fits_ext].data
    header = hdul[fits_ext].header
    wavelengths_um = hdul[fits_ext - 1].data

    wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)
    contrast_cube = jnp.asarray(obj_data[:-1].astype(np.float32))
    pixel_scale_arcsec = header["PXSCLMAS"] * mas2arcsec

    return ExovistaDisk(
        pixel_scale_arcsec=pixel_scale_arcsec,
        wavelengths_nm=wavelengths_nm,
        contrast_cube=contrast_cube,
    )


def _earth_like_planet_indices_from_hdul(hdul: fits.HDUList) -> list[int]:
    """Earth-filter logic operating on an already-open HDUList."""
    n_ext = hdul[0].header["N_EXT"]
    n_planets_total = n_ext - 4
    star_luminosity_lsun = hdul[4].header.get("LSTAR", 1.0)

    earth_indices: list[int] = []
    for i in range(n_planets_total):
        planet_header = hdul[5 + i].header
        a_au = planet_header.get("A", 1.0)
        radius_rearth = planet_header.get("R", 1.0)
        a_scaled = a_au / np.sqrt(star_luminosity_lsun)
        lower_r = 0.8 / np.sqrt(a_scaled)
        if (0.95 <= a_scaled < 1.67) and (lower_r <= radius_rearth < 1.4):
            earth_indices.append(i)
    return earth_indices


def get_earth_like_planet_indices(fits_file: str) -> list[int]:
    """Identify Earth-like planets in an ExoVista FITS file.

    Classification criteria:
      - Scaled semi-major axis: 0.95 <= a / sqrt(L_star) < 1.67 AU
      - Planet radius: 0.8 / sqrt(a_scaled) <= R < 1.4 R_earth
    """
    with fits.open(fits_file, memmap=False) as hdul:
        return _earth_like_planet_indices_from_hdul(hdul)


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

    with fits.open(fits_file, memmap=False) as hdul:
        n_ext = hdul[0].header["N_EXT"]
        n_planets_total = n_ext - 4

        if planet_indices is None:
            if only_earths:
                planet_indices = _earth_like_planet_indices_from_hdul(hdul)
            else:
                planet_indices = list(range(n_planets_total))

        wavelengths_um = hdul[0].data
        wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)
        star_header = hdul[4].header
        midplane_inc_deg = float(star_header.get("I", 0.0))
        midplane_pa_deg = float(star_header.get("PA", 0.0))

        star = _load_star(hdul, fits_ext=4)
        solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
        planets = tuple(
            _load_single_planet(hdul, i, star, wavelengths_nm, solver)[0]
            for i in planet_indices
        )
        disk = _load_disk(hdul, disk_ext)

    return System(
        star=star,
        planets=planets,
        disk=disk,
        trig_solver=solver,
        midplane_inc_deg=midplane_inc_deg,
        midplane_pa_deg=midplane_pa_deg,
    )
