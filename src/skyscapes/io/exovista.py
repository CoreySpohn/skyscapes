"""ExoVista FITS loader -- builds a ``skyscapes.scene.System`` directly.

Supports legacy and spotted-star layouts. Each planet has a Keplerian
orbit and either a time-indexed contrast table or a phase-indexed model.
New layouts preserve recorded activity-dependent spectra by default.
Unmarked spotted-star spectra are converted from the PHOENIX surface flux
the v2.5 and v2.6 writers store to observer flux density in Jy.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import astropy.units as u
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import R_sun
from astropy.io import fits
from hwoutils.constants import (
    J2000_JD,
    AU2m,
    G_si,
    Mearth2kg,
    Msun2kg,
    cm22m2,
    d2s,
    mas2arcsec,
    pc2m,
    rad2arcsec,
    um2nm,
    yr2s,
)
from hwoutils.conversions import au_per_yr_to_m_per_s, mag_to_flux_jy
from orbix.equations.orbit import state_vector_to_keplerian
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit

from ..disk import ExovistaDisk
from ..physical_model import GridPhysicalModel, TimeSeriesPhysicalModel
from ..scene import Planet, Star, StarSpots, System

# Area of the quiet-disk limb-darkening mask that the ExoVista v2.5 and v2.6
# writers multiply into every stellar spectrum, in units of the geometric disk:
# the sum of the writer's mask (solar quadratic coefficients u1 = 0.0473,
# u2 = 0.0841, hard-coded there) times 4 / (pi * mapsize^2). It changes by less
# than 1e-5 between mapsize 512 and 3600 (default 1800).
EXOVISTA_LD_MASK_AREA = 0.9109100

# Observer flux at the V band (Vega magnitude, read here on the AB scale) that
# a converted spectrum must match to within this factor, which leaves room for
# spot dimming, the model atmosphere and the Vega-AB offset but not for a unit
# error. The check uses the channel nearest 550 nm, and only when that channel
# lies within the tolerance below; a grid that does not reach V is not checked.
_V_BAND_UM = 0.55
_V_BAND_TOLERANCE_UM = 0.06
_V_BAND_FACTOR = 3.0


def _finite_header_number(header, key):
    """Return a numeric, finite header value, or None if it is not one."""
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _phoenix_surface_to_observer_jy(stored, wavelengths_um, header) -> np.ndarray:
    """Convert a v2.5/v2.6 stellar spectrum to observer flux density in Jy.

    Those writers store the PHOENIX emergent surface flux density
    ``F_lambda`` in erg s^-1 cm^-2 cm^-1, weighted by the area of their
    limb-darkening mask, in columns documented as observer Jy. The observer
    flux density is the surface flux diluted by ``(R_star / d)^2``, with no
    further factor of pi because PHOENIX tabulates flux rather than intensity.

    Args:
        stored: Stellar spectra as written, shape ``(n_times, n_wavelengths)``.
        wavelengths_um: Channel wavelengths in microns.
        header: The stellar extension header, which must carry ``RSTAR``
            (solar radii) and ``DIST`` (pc).

    Returns:
        Observer flux density in Jy, the shape of ``stored``.

    Raises:
        ValueError: ``RSTAR`` or ``DIST`` is missing or not a finite number,
            or the converted spectrum disagrees with the header's ``VMAG`` by
            more than a factor of three.

    Warns:
        RuntimeWarning: The conversion could not be checked against ``VMAG``,
            because the header has no finite numeric ``VMAG`` or no channel
            lies near 550 nm.
    """
    rstar = _finite_header_number(header, "RSTAR")
    dist = _finite_header_number(header, "DIST")
    if rstar is None or dist is None:
        raise ValueError(
            "Converting an unmarked ExoVista v2.5+ spectrum needs finite numeric "
            f"RSTAR and DIST; got RSTAR={header.get('RSTAR')!r}, "
            f"DIST={header.get('DIST')!r}."
        )
    dilution = float(((rstar * R_sun) / (dist * u.pc)).decompose() ** 2)
    surface = np.asarray(stored, dtype=np.float64) / EXOVISTA_LD_MASK_AREA
    observer_jy = (
        (surface * dilution * u.erg / u.s / u.cm**2 / u.cm)
        .to(u.Jy, equivalencies=u.spectral_density(wavelengths_um * u.um))
        .value
    )

    vmag = _finite_header_number(header, "VMAG")
    offsets_um = np.abs(np.asarray(wavelengths_um, dtype=np.float64) - _V_BAND_UM)
    k = int(np.argmin(offsets_um))
    if vmag is None or offsets_um[k] > _V_BAND_TOLERANCE_UM:
        warnings.warn(
            "Converted an unmarked ExoVista v2.5+ stellar spectrum from PHOENIX "
            "surface flux to observer Jy without checking it: the header has no "
            "numeric VMAG or no channel lies near 550 nm. Mark the stellar "
            "extension with SPECUNIT='Jy' if the spectrum is already observer "
            "flux density.",
            RuntimeWarning,
            stacklevel=3,
        )
        return observer_jy
    ratio = float(np.median(observer_jy[:, k])) / float(mag_to_flux_jy(vmag))
    if not 1.0 / _V_BAND_FACTOR < ratio < _V_BAND_FACTOR:
        raise ValueError(
            "An unmarked ExoVista v2.5+ stellar spectrum, read as PHOENIX "
            f"surface flux, is {ratio:.3g} times the flux of its VMAG; its "
            "units are unknown. Mark the stellar extension with "
            "SPECUNIT='Jy' if the spectrum is already observer flux density."
        )
    return observer_jy


def _angular_diameter_arcsec(header) -> float:
    """Stellar angular diameter, ``2 R / d``, from RSTAR and DIST.

    ExoVista writes the same stellar header for its two inputs, but ANGDIAM
    differs between them: a hand-written system file stores the angular
    radius (0.465 mas per solar radius at 10 pc), while a catalog target
    stores the Boyajian et al. (2014) angular diameter and derives RSTAR from
    it. ANGDIAM alone cannot say which it holds, so the diameter comes from
    RSTAR (solar radii) and DIST (pc), which mean the same on both paths and
    which every ExoVista version this loader reads records.

    Raises:
        ValueError: RSTAR is not a finite number at least zero, or DIST is
            not a finite positive number.
    """
    rstar = _finite_header_number(header, "RSTAR")
    dist = _finite_header_number(header, "DIST")
    if rstar is None or rstar < 0 or dist is None or dist <= 0:
        raise ValueError(
            "The stellar angular diameter needs a finite RSTAR >= 0 and a "
            f"finite DIST > 0; got RSTAR={header.get('RSTAR')!r}, "
            f"DIST={header.get('DIST')!r}. ANGDIAM cannot stand in: ExoVista "
            "system files store the angular radius there, catalog targets the "
            "diameter."
        )
    return 2 * rstar * R_sun.to_value(u.m) / (dist * pc2m) * rad2arcsec


def _simulation_times_jd(elapsed_years) -> jnp.ndarray:
    """Assign the arbitrary simulation origin to J2000, using Julian years."""
    return J2000_JD + jnp.asarray(elapsed_years.astype(np.float64)) * (yr2s / d2s)


def _star_extension(hdul: fits.HDUList) -> int:
    """Locate the star by its header, including files with stale N_EXT cards."""
    candidates = [
        i
        for i, hdu in enumerate(hdul)
        if all(key in hdu.header for key in ("ANGDIAM", "MASS", "DIST"))
    ]
    if len(candidates) != 1 or candidates[0] not in (4, 5):
        raise ValueError("Expected one ExoVista star in extension 4 or 5.")
    return candidates[0]


def _body_extensions(hdul: fits.HDUList, star_ext: int) -> list[int]:
    """Identify planets after the stellar extension by their physical headers."""
    return [
        i
        for i in range(star_ext + 1, len(hdul))
        if all(key in hdul[i].header for key in ("A", "E", "M", "R"))
    ]


def _load_spots(hdul: fits.HDUList, star_ext: int, times_jd) -> StarSpots | None:
    """Preserve the optional v2.5+ spot extension without inventing evolution."""
    if star_ext == 4:
        return None
    data = hdul[4].data
    if data is None:
        data = np.empty((len(times_jd), 0, 4))
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or data.shape[0] != len(times_jd) or data.shape[2] != 4:
        raise ValueError("ExoVista spots must have shape (n_times, n_spots, 4).")
    if not np.all(np.isfinite(data)) or np.any(data[..., 2] < 0):
        raise ValueError("ExoVista spot records must be finite with nonnegative areas.")
    if np.any(data[..., 3] < 1):
        raise ValueError("ExoVista total-spot/umbra area ratios must be at least one.")
    header = {key.upper(): value for key, value in hdul[star_ext].header.items()}
    return StarSpots(
        times_jd=times_jd,
        latitude_deg=jnp.asarray(data[..., 0]),
        longitude_deg=jnp.asarray(data[..., 1]),
        area_m2=jnp.asarray(data[..., 2] * cm22m2),
        area_ratio=jnp.asarray(data[..., 3]),
        umbra_temperature_k=header.get("TEFFUMBRA"),
        penumbra_temperature_k=header.get("TEFFPENUMBRA"),
    )


def _load_star(hdul: fits.HDUList, fits_ext: int = 4) -> Star:
    """Load the FITS star extension into a Star."""
    obj_data = hdul[fits_ext].data
    obj_header = hdul[fits_ext].header
    wavelengths_um = hdul[0].data

    spectral_unit = obj_header.get("SPECUNIT")
    if spectral_unit is not None and spectral_unit.strip().lower() != "jy":
        raise ValueError("ExoVista stellar SPECUNIT must be Jy.")

    wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)
    times_jd = _simulation_times_jd(obj_data[:, 0])
    spectra = obj_data[:, 16:]
    if fits_ext == 5 and spectral_unit is None:
        spectra = _phoenix_surface_to_observer_jy(spectra, wavelengths_um, obj_header)
    flux_density_jy = jnp.asarray(np.asarray(spectra).T.astype(np.float32))

    diameter_arcsec = _angular_diameter_arcsec(obj_header)
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
        interpolation_method="linear",
        spots=_load_spots(hdul, fits_ext, times_jd),
        metadata={
            key.upper(): value
            for key, value in obj_header.items()
            if key and key not in ("COMMENT", "HISTORY")
        },
    )


def _exovista_to_orbix(vector: jnp.ndarray) -> jnp.ndarray:
    """Convert (west, north, toward) to orbix's (north, east, away).

    ExoVista's ``src/exovista/generate_scene.py`` ``generate_scene``
    docstring defines +x as west, +y as north and +z toward the observer.
    For either a position or velocity vector, the conversion is (y, -x, -z).
    """
    return jnp.stack([vector[1], -vector[0], -vector[2]])


def _load_single_planet(
    hdul: fits.HDUList,
    fits_ext: int,
    star: Star,
    wavelengths_nm: jnp.ndarray,
    trig_solver,
    *,
    planet_spectra: Literal["time", "phase"],
) -> tuple[Planet, float]:
    """Load one planet and return ``(Planet, t0_d)``.

    ExoVista stores the planet ``(r, v)`` state vectors already in the
    on-sky ("bary-sky") frame -- the same frame the disk cube is rendered
    in. Subtract the star's state at the same epoch to obtain star-relative
    states, then convert their (west, north, toward) axes to orbix's
    (north, east, away) axes before deriving Keplerian elements. The system
    inclination and PA are already applied by the writer.

    With ``planet_spectra="time"`` the contrasts stay indexed by the FITS
    epochs, which preserves the stellar-activity history. With
    ``planet_spectra="phase"`` they are regridded onto the standard phase
    angle returned by orbix (0 = full phase), computed at load time with
    the same ``trig_solver`` the runtime ``System`` uses; this stays
    self-consistent with rendering because the grid is keyed by the same
    phase angle ``Planet.propagate`` returns at render time.
    """
    obj_data = hdul[fits_ext].data
    obj_header = hdul[fits_ext].header

    times_jd = _simulation_times_jd(obj_data[:, 0])
    t0 = float(times_jd[0])

    contrast_data = jnp.asarray(obj_data[:, 16:].T.astype(np.float32))

    # Both layouts store barycentric body states in columns 9:15. Convert
    # big-endian FITS data to native float64 before subtracting stellar reflex.
    star_data = hdul[_star_extension(hdul)].data
    relative_state = obj_data[0, 9:15].astype(np.float64) - star_data[0, 9:15].astype(
        np.float64
    )
    r_sky_au = _exovista_to_orbix(jnp.asarray(relative_state[:3]))
    v_sky_au_yr = _exovista_to_orbix(jnp.asarray(relative_state[3:]))

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

    if planet_spectra == "time":
        physical_model = TimeSeriesPhysicalModel(
            wavelengths_nm=wavelengths_nm,
            times_jd=times_jd,
            contrast_grid=contrast_data[None, ...],
        )
        return Planet(
            Rp_Rearth=jnp.array([float(obj_header["R"])]),
            Mp_Mearth=jnp.array([float(obj_header["M"])]),
            orbit=orbit,
            physical_model=physical_model,
        ), t0

    # beta(t_i) per FITS input time, using the same solver the runtime uses.
    _r_AU, beta_rad, _dist_AU = orbit.propagate(trig_solver, times_jd, Ms_kg=star.Ms_kg)
    # orbix already returns the standard phase angle, as does Planet.propagate.
    phase_angle_rad = beta_rad
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


def _midplane_to_east_north(inc_deg: float, pa_deg: float) -> tuple[float, float]:
    """Convert ExoVista header (I, PA) to the east/north midplane chart.

    ExoVista's ``src/exovista/generate_scene.py`` ``generate_scene``
    docstring defines +x west, +y north, +z toward the observer. Mirroring
    x maps the node vector to minus the node vector at -PA, and the near
    in-plane vector to the near vector at -PA, with I unchanged. This
    preserves both the plane and its near side below and above I = 90;
    180 - PA would preserve only the projected major-axis line.
    """
    return inc_deg, (-pa_deg) % 360.0


def _load_disk(hdul: fits.HDUList, fits_ext: int) -> ExovistaDisk:
    """Load the ExoVista disk extension into an ExovistaDisk.

    The contrast cube is already rendered in the on-sky frame by
    ExoVista, with +column west and +row north (see ExoVista's
    ``src/exovista/generate_scene.py``). Mirror columns at load so
    +column is the east offset (Delta alpha*) and +row is north,
    matching planets and parametric disks. The image center remains
    ``((nx - 1) / 2, (ny - 1) / 2)`` for even and odd dimensions.

    ``from_exovista`` converts the FITS star header's (I, PA) to
    (I, -PA modulo 360) on ``System.midplane_inc_deg`` /
    ``System.midplane_pa_deg``. These east/north angles orient parametric
    disks and outlines; this loaded cube already has its geometry.
    """
    obj_data = hdul[fits_ext].data
    header = hdul[fits_ext].header
    wavelengths_um = hdul[fits_ext - 1].data

    wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)
    contrast_cube = jnp.asarray(
        np.ascontiguousarray(obj_data[:-1, :, ::-1], dtype=np.float32)
    )
    pixel_scale_arcsec = header["PXSCLMAS"] * mas2arcsec

    return ExovistaDisk(
        pixel_scale_arcsec=pixel_scale_arcsec,
        wavelengths_nm=wavelengths_nm,
        contrast_cube=contrast_cube,
    )


def _earth_like_planet_indices_from_hdul(hdul: fits.HDUList) -> list[int]:
    """Earth-filter logic operating on an already-open HDUList."""
    star_ext = _star_extension(hdul)
    planet_extensions = _body_extensions(hdul, star_ext)
    star_luminosity_lsun = hdul[star_ext].header.get("LSTAR", 1.0)

    earth_indices: list[int] = []
    for i, ext in enumerate(planet_extensions):
        planet_header = hdul[ext].header
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
    load_disk: bool = True,
    *,
    planet_spectra: Literal["auto", "time", "phase"] = "auto",
) -> System:
    """Load an ExoVista FITS file into a ``scene.System``.

    State vectors are already on-sky; their (west, north, toward) axes
    are converted to orbix's (north, east, away) axes. The FITS star
    header's midplane (I, PA) becomes (I, -PA modulo 360) in the east/north
    chart on ``System.midplane_*``, used by parametric disks and outlines.
    These angles are not applied again to the states or loaded image.
    Elapsed simulation years become Julian Dates relative
    to J2000, with exactly 365.25 days per year. Enable ``jax_enable_x64``
    before loading to preserve this time coordinate.

    Args:
        fits_file: Path to ExoVista FITS file.
        planet_indices: Planet indices to load (0-based). ``None`` = all.
        only_earths: If True and *planet_indices* is None, auto-filter Earths.
        load_disk: If False, skip the disk extension entirely and set
            ``System.disk`` to ``None``. Useful when the downstream
            pipeline cannot render a disk (e.g. a coronagraph without a
            PSF datacube).
        planet_spectra: ``auto`` preserves time-dependent contrasts in
            v2.5+ scenes and the existing phase model for older scenes.
            ``time`` preserves the recorded spectra at their epochs, with
            no extrapolation beyond a multi-epoch baseline. A single epoch
            is a constant snapshot. ``phase`` explicitly assumes contrast
            is a function of phase alone, discarding activity history.

    Returns:
        ``scene.System`` with star, planets (tuple), disk, and midplane metadata.

    Raises:
        ValueError: JAX float64 is disabled, the FITS layout or spot records
            are invalid, the stellar header lacks a finite ``RSTAR >= 0`` or
            a finite positive ``DIST``, or a planet selection or spectra mode
            is invalid.
    """
    if not jax.config.x64_enabled:
        raise ValueError(
            "ExoVista import requires jax_enable_x64 for Julian Date precision. "
            'Set jax.config.update("jax_enable_x64", True) before loading.'
        )
    disk_ext = 2

    with fits.open(fits_file, memmap=False) as hdul:
        star_ext = _star_extension(hdul)
        planet_extensions = _body_extensions(hdul, star_ext)
        n_planets_total = len(planet_extensions)
        if planet_spectra not in ("auto", "time", "phase"):
            raise ValueError("planet_spectra must be auto, time, or phase.")
        if planet_spectra == "auto":
            planet_spectra = "time" if star_ext == 5 else "phase"

        if planet_indices is None:
            if only_earths:
                planet_indices = _earth_like_planet_indices_from_hdul(hdul)
            else:
                planet_indices = list(range(n_planets_total))
        if any(
            not isinstance(i, Integral) or not 0 <= i < n_planets_total
            for i in planet_indices
        ):
            raise ValueError("planet_indices must address existing planets (0-based).")

        wavelengths_um = hdul[0].data
        wavelengths_nm = jnp.asarray(wavelengths_um * um2nm)
        star_header = hdul[star_ext].header
        midplane_inc_deg, midplane_pa_deg = _midplane_to_east_north(
            float(star_header.get("I", 0.0)), float(star_header.get("PA", 0.0))
        )

        star = _load_star(hdul, fits_ext=star_ext)
        solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
        planets = tuple(
            _load_single_planet(
                hdul,
                planet_extensions[i],
                star,
                wavelengths_nm,
                solver,
                planet_spectra=planet_spectra,
            )[0]
            for i in planet_indices
        )
        disk = _load_disk(hdul, disk_ext) if load_disk else None

    return System(
        star=star,
        planets=planets,
        disk=disk,
        trig_solver=solver,
        midplane_inc_deg=midplane_inc_deg,
        midplane_pa_deg=midplane_pa_deg,
    )
