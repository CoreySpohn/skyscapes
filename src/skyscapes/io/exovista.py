"""ExoVista FITS loader -- builds a ``skyscapes.scene.Scene`` directly.

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
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import R_sun
from astropy.io import fits
from hwoutils.constants import (
    J2000_JD,
    Mearth2kg,
    Msun2kg,
    cm22m2,
    d2s,
    mas2arcsec,
    um2nm,
    yr2s,
)
from hwoutils.conversions import (
    angular_diameter_arcsec,
    mag_to_flux_jy,
)
from hwoutils.epoch import Epoch
from optixstuff.grids import PixelGrid, SkyOrientation

from ..disk import ExovistaDisk
from ..physical_model import GridPhysicalModel, TimeSeriesPhysicalModel
from ..population import earth_like
from ..scene import AbstractLocalZodi, AYOZodi, Planet, Scene, Star, StarSpots

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
    return angular_diameter_arcsec(rstar * R_sun.to_value(u.m), dist)


def _simulation_times_d(elapsed_years, epoch: Epoch) -> np.ndarray:
    """Convert simulation years to days from the scene epoch on the host."""
    elapsed_d = np.asarray(elapsed_years, dtype=np.float64) * (yr2s / d2s)
    return elapsed_d + Epoch(J2000_JD, scale=epoch.scale).days_between(epoch)


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


def _load_spots(hdul: fits.HDUList, star_ext: int, times_d) -> StarSpots | None:
    """Preserve the optional v2.5+ spot extension without inventing evolution.

    ``times_d`` is days from the scene epoch.
    """
    if star_ext == 4:
        return None
    data = hdul[4].data
    if data is None:
        data = np.empty((len(times_d), 0, 4))
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or data.shape[0] != len(times_d) or data.shape[2] != 4:
        raise ValueError("ExoVista spots must have shape (n_times, n_spots, 4).")
    if not np.all(np.isfinite(data)) or np.any(data[..., 2] < 0):
        raise ValueError("ExoVista spot records must be finite with nonnegative areas.")
    if np.any(data[..., 3] < 1):
        raise ValueError("ExoVista total-spot/umbra area ratios must be at least one.")
    header = {key.upper(): value for key, value in hdul[star_ext].header.items()}
    return StarSpots(
        times_d=times_d,
        latitude_deg=jnp.asarray(data[..., 0]),
        longitude_deg=jnp.asarray(data[..., 1]),
        area_m2=jnp.asarray(data[..., 2] * cm22m2),
        area_ratio=jnp.asarray(data[..., 3]),
        umbra_temperature_K=header.get("TEFFUMBRA"),
        penumbra_temperature_K=header.get("TEFFPENUMBRA"),
    )


def _load_star(hdul: fits.HDUList, epoch: Epoch, fits_ext: int = 4) -> Star:
    """Load the FITS star extension into a Star."""
    obj_data = hdul[fits_ext].data
    obj_header = hdul[fits_ext].header
    wavelengths_um = hdul[0].data

    spectral_unit = obj_header.get("SPECUNIT")
    if spectral_unit is not None and spectral_unit.strip().lower() != "jy":
        raise ValueError("ExoVista stellar SPECUNIT must be Jy.")

    # File knots stay host float64: widened from the stored microns first.
    wavelengths_nm = np.asarray(wavelengths_um, dtype=np.float64) * um2nm
    times_d = jnp.asarray(_simulation_times_d(obj_data[:, 0], epoch))
    spectra = obj_data[:, 16:]
    if fits_ext == 5 and spectral_unit is None:
        spectra = _phoenix_surface_to_observer_jy(spectra, wavelengths_um, obj_header)
    flux_density_jy = jnp.asarray(np.asarray(spectra).T.astype(np.float32))

    fixed_diameter_arcsec = _angular_diameter_arcsec(obj_header)
    luminosity_Lsun = obj_header.get("LSTAR", 1.0)

    return Star(
        fixed_diameter_arcsec=fixed_diameter_arcsec,
        luminosity_Lsun=luminosity_Lsun,
        wavelengths_nm=wavelengths_nm,
        times_d=times_d,
        flux_density_jy=flux_density_jy,
        interpolation_method="linear",
        spots=_load_spots(hdul, fits_ext, times_d),
    )


def exovista_header(fits_file: str) -> dict:
    """Read the star extension header of an ExoVista file as a plain dictionary.

    The header holds host data a scene does not carry (``TEFF``, ``RA``,
    ``DEC``, target names and so on). It is returned beside the scene, never
    stored on a Module, so scenes with different headers still batch.

    Args:
        fits_file: Path to the ExoVista FITS file.

    Returns:
        Dictionary of the star extension cards under uppercase keys, excluding
        blank, ``COMMENT`` and ``HISTORY`` cards.
    """
    with fits.open(fits_file, memmap=False) as hdul:
        header = hdul[_star_extension(hdul)].header
        return {
            key.upper(): value
            for key, value in header.items()
            if key and key not in ("COMMENT", "HISTORY")
        }


def _exovista_to_orbix(vector: np.ndarray) -> np.ndarray:
    """Rotate a FITS (west, north, toward) vector into (east, north, away)."""
    return np.asarray(vector, dtype=np.float64) * (-1.0, 1.0, -1.0)


def _first_state(row):
    """Return the barycentric position [AU] and velocity [AU/day] in sky axes."""
    return _exovista_to_orbix(row[9:12]), _exovista_to_orbix(row[12:15]) / (yr2s / d2s)


def _load_single_planet(
    hdul: fits.HDUList,
    fits_ext: int,
    wavelengths_nm: np.ndarray,
    *,
    first_year: float,
    epoch: Epoch,
) -> tuple[Planet, np.ndarray, np.ndarray, float]:
    """Read a selected planet's appearance, barycentric state and solar mass.

    The Jacobi chain contains the star and selected planets only. Each
    planet's position and velocity relative to the star reproduce the file
    values at the first epoch to rounding.
    """
    data = hdul[fits_ext].data
    header = hdul[fits_ext].header
    if data[0, 0] != first_year:
        raise ValueError(
            f"Planet extension {fits_ext} and star have different first epochs"
        )
    position, velocity = _first_state(data[0])
    model = TimeSeriesPhysicalModel(
        wavelengths_nm=wavelengths_nm,
        times_d=_simulation_times_d(data[:, 0], epoch),
        contrast_grid=data[:, 16:].T.astype(np.float32)[None, ...],
    )
    planet = Planet(Rp_Rearth=jnp.array([float(header["R"])]), physical_model=model)
    return planet, position, velocity, float(header["M"]) * Mearth2kg / Msun2kg


def _phase_model(model: TimeSeriesPhysicalModel, beta_rad) -> GridPhysicalModel:
    """Regrid stored contrasts at their ephemeris phase angles [rad]."""
    wavelengths_nm = model.wavelengths_nm.values
    beta_deg = np.rad2deg(np.asarray(beta_rad))
    order = np.argsort(beta_deg)
    beta = beta_deg[order]
    spectra = np.asarray(model.contrast_grid[0])[:, order]
    regular_grid = np.linspace(0.0, 180.0, 100)

    # Merge phases to a microdegree, far below the regrid resolution.
    starts = [0]
    for i in range(1, beta.size):
        if beta[i] - beta[starts[-1]] > 1e-6:
            starts.append(i)
    counts = np.diff(np.r_[starts, beta.size])
    phases = np.add.reduceat(beta, starts) / counts
    if np.ptp(phases) < regular_grid[1] - regular_grid[0]:
        # Each stored epoch contributes equally to an unresolved spectrum.
        contrast_grid = np.broadcast_to(
            spectra.mean(axis=1, dtype=np.float64, keepdims=True).astype(spectra.dtype),
            (wavelengths_nm.size, regular_grid.size),
        )
        bounds = regular_grid[[0, -1]]
    else:
        means = np.add.reduceat(spectra, starts, axis=1, dtype=np.float64) / counts
        contrast_grid = np.stack(
            [np.interp(regular_grid, phases, row) for row in means]
        )
        bounds = phases[[0, -1]]

    return GridPhysicalModel(
        knots=wavelengths_nm,
        phase_angle_deg=regular_grid,
        contrast_grid=np.maximum(contrast_grid[None, ...], 0.0),
        phase_bounds_deg=bounds[None, :],
    )


def _midplane_to_east_north(inc_deg: float, pa_deg: float) -> tuple[float, float]:
    """Convert ExoVista header (I, PA) to the east/north midplane chart.

    ExoVista's ``src/exovista/generate_scene.py`` ``generate_scene``
    docstring defines +x west, +y north, +z toward the observer. Negating
    x and z gives nodes (-cos(PA), sin(PA), 0) and the near vector
    (cos(I) sin(PA), cos(I) cos(PA), -sin(I)). Astronomical angle 90+PA
    has nodes opposite the converted nodes (the same line), and precisely
    this near vector, preserving the physical near side even above I=90.
    """
    return inc_deg, (90.0 + pa_deg) % 360.0


def _load_disk(hdul: fits.HDUList, fits_ext: int) -> ExovistaDisk:
    """Load the ExoVista disk extension into an ExovistaDisk.

    The contrast cube is already rendered in the on-sky frame by
    ExoVista, with +column west and +row north (FITS display convention;
    see ExoVista's ``src/exovista/generate_scene.py``). It is used as written.
    With ``imshow(origin="lower")`` and no axis inversion, north is up and
    east is left; astronomical PA (north through east) is counterclockwise.
    Column offset = -east / pixel scale.

    ``from_exovista`` converts the FITS star header's (I, PA) to
    (I, 90 + PA modulo 360) on ``Scene.midplane_inc_deg`` /
    ``Scene.midplane_position_angle_deg``. These east/north angles orient parametric
    disks and outlines; this loaded cube already has its geometry.
    """
    obj_data = hdul[fits_ext].data
    header = hdul[fits_ext].header
    wavelengths_um = hdul[fits_ext - 1].data

    # The cube keeps its float32-rounded stored values but is carried in the
    # active float dtype, so wavelength interpolation never casts float64
    # values into a float32 buffer when x64 is enabled.
    active = jnp.result_type(float)
    wavelengths_nm = np.asarray(wavelengths_um, dtype=np.float64) * um2nm
    contrast_cube = jnp.asarray(obj_data[:-1].astype(np.float32), dtype=active)
    pixel_scale_arcsec = header["PXSCLMAS"] * mas2arcsec

    return ExovistaDisk(
        native_grid=PixelGrid.regular(
            contrast_cube.shape[-2:],
            pixel_scale_arcsec,
            "arcsec",
            orientation=SkyOrientation(0.0),
        ),
        wavelengths_nm=wavelengths_nm,
        contrast_cube=contrast_cube,
    )


def _earth_like_planet_indices_from_hdul(hdul: fits.HDUList) -> list[int]:
    """Earth-filter logic operating on an already-open HDUList."""
    star_ext = _star_extension(hdul)
    planet_extensions = _body_extensions(hdul, star_ext)
    star_luminosity_Lsun = hdul[star_ext].header.get("LSTAR", 1.0)

    headers = [hdul[ext].header for ext in planet_extensions]
    a_AU = np.array([h.get("A", 1.0) for h in headers], dtype=float)
    radius_Rearth = np.array([h.get("R", 1.0) for h in headers], dtype=float)
    mask = np.asarray(earth_like(a_AU, radius_Rearth, star_luminosity_Lsun))
    return [int(i) for i in np.flatnonzero(mask)]


def get_earth_like_planet_indices(fits_file: str) -> list[int]:
    """Identify Earth-like planets in an ExoVista FITS file.

    Classification criteria:
      - Scaled semi-major axis: 0.95 <= a / sqrt(L_star) < 1.67 AU
      - Planet radius: 0.8 / sqrt(a_scaled) <= R < 1.4 R_earth

    Args:
        fits_file: Path of the ExoVista FITS file.

    Returns:
        Zero-based indices of the planets that meet the Earth-like criteria.
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
    epoch: Epoch | None = None,
    local_zodi: AbstractLocalZodi | None = AYOZodi(),  # noqa: B008
) -> Scene:
    """Load an ExoVista FITS file into a ``Scene``.

    The chain contains the star and selected planets only. At the first epoch,
    each selected planet reproduces its file position and velocity relative to
    the star to rounding.

    State vectors are already on-sky. Their (west, north, toward) axes
    are converted to orbix's (east, north, away) axes. The FITS star
    header's midplane (I, PA) becomes (I, 90 + PA modulo 360) in the east/north
    chart on ``Scene.midplane_*``, used by parametric disks and outlines.
    These angles are not applied again to the states or loaded image.
    Elapsed simulation years use exactly 365.25 days per year and an
    assigned J2000 origin. By default the first stellar timestamp sets the
    scene epoch, split into whole and fractional days. The file states no
    time scale; the epoch's label is used without a scale conversion.
    Every stored time and orbit reference is converted in NumPy float64
    to days from the scene epoch before transfer to JAX. Enable
    ``jax_enable_x64`` before loading to preserve orbital-state precision.

    Args:
        fits_file: Path to ExoVista FITS file.
        planet_indices: Planet indices to load (0-based). ``None`` = all.
        only_earths: If True and *planet_indices* is None, auto-filter Earths.
        load_disk: If False, skip the disk extension entirely and set
            ``Scene.disk`` to ``None``. Useful when the downstream
            pipeline cannot render a disk (e.g. a coronagraph without a
            PSF datacube).
        planet_spectra: ``auto`` preserves time-dependent contrasts in
            v2.5+ scenes and the existing phase model for older scenes.
            ``time`` preserves the recorded spectra at their epochs, with
            no extrapolation beyond a multi-epoch baseline. A single epoch
            is a constant snapshot. ``phase`` explicitly assumes contrast
            is a function of phase alone, discarding activity history. The phase
            angles come from the assembled scene at the stored epochs. A span
            narrower than one phase-grid interval gives the epoch-mean spectrum.
            Resolved curves hold their endpoint spectra outside the sampled span.
        epoch: Scene epoch that stored and observation times are measured
            from. ``None`` uses the first stellar timestamp. Pass a shared
            epoch to place several scenes on one clock.
        local_zodi: Local zodi model stored on the scene. The default is
            :class:`~skyscapes.scene.AYOZodi` (22 mag per square arcsecond); ``None``
            stores no zodi.

    Returns:
        ``Scene`` with star, planets (tuple), disk, and midplane metadata.
        The scene is placed on the sky from the stellar header ``RA`` and
        ``DEC`` (degrees); a header without them gives an unplaced scene.

    Raises:
        ValueError: JAX float64 is disabled, the FITS layout or spot records
            are invalid, the stellar header lacks a finite ``RSTAR >= 0`` or
            a finite positive ``DIST``, a planet selection or spectra mode is
            invalid, or a selected planet and the star have different first epochs.
    """
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "ExoVista import requires jax_enable_x64 for orbital-state precision. "
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
        wavelengths_nm = np.asarray(wavelengths_um, dtype=np.float64) * um2nm
        star_header = hdul[star_ext].header
        midplane_inc_deg, midplane_position_angle_deg = _midplane_to_east_north(
            float(star_header.get("I", 0.0)), float(star_header.get("PA", 0.0))
        )

        first_row = hdul[star_ext].data[0]
        star_position, star_velocity = _first_state(first_row)
        first_year = float(first_row[0])
        first_d = first_year * (yr2s / d2s)
        whole_d = np.floor(first_d)
        if epoch is None:
            epoch = Epoch(J2000_JD + whole_d, first_d - whole_d)
        star = _load_star(hdul, epoch, fits_ext=star_ext)
        star_mass_Msun = float(star_header["MASS"])
        dist_pc = float(star_header["DIST"])
        loaded = tuple(
            _load_single_planet(
                hdul,
                planet_extensions[i],
                wavelengths_nm,
                first_year=first_year,
                epoch=epoch,
            )
            for i in planet_indices
        )
        disk = _load_disk(hdul, disk_ext) if load_disk else None

    scene = Scene.from_cartesian(
        star=star,
        planets=tuple(item[0] for item in loaded),
        star_mass_Msun=star_mass_Msun,
        planet_mass_Msun=np.asarray([item[3] for item in loaded]),
        dist_pc=dist_pc,
        epoch=epoch,
        epoch_d=_simulation_times_d(first_year, epoch),
        position_AU=np.stack([star_position, *(item[1] for item in loaded)]),
        velocity_AU_per_d=np.stack([star_velocity, *(item[2] for item in loaded)]),
        disk=disk,
        midplane_inc_deg=midplane_inc_deg,
        midplane_position_angle_deg=midplane_position_angle_deg,
        local_zodi=local_zodi,
        ra_deg=star_header.get("RA"),
        dec_deg=star_header.get("DEC"),
    )
    if planet_spectra == "phase" and scene.planets:
        # Solve once even when selected planets have different sampling times.
        tables = [planet.physical_model for planet in scene.planets]
        times = np.unique(np.concatenate([table.times_d for table in tables]))
        eph = scene.ephemeris(times)
        beta = eph.phase_angle_rad(eph.scene_planet_indices, 0)
        models = tuple(
            _phase_model(table, angles[np.searchsorted(times, table.times_d)])
            for table, angles in zip(tables, beta, strict=True)
        )
        scene = eqx.tree_at(
            lambda s: tuple(p.physical_model for p in s.planets), scene, models
        )
    return scene
