"""PSG cross-section opacity adapter.

The PSG (NASA's Planetary Spectrum Generator) hosts pre-computed
absorption cross-sections in a uniform ASCII format for many
atmospheric gases that ExoJAX's line-by-line machinery
(``MdbHitran``/``OpaPremodit``) cannot consume directly -- either
because the band is an electronic continuum (e.g. O3 Chappuis,
SO2 in the UV) or because HITRAN has no entries in the desired
wavelength range. This module ships an ``Opa``-shaped adapter that
loads any such PSG cross-section file and exposes the same
``xsmatrix(Tarr, pressure) -> (n_layers, n_nu)`` contract as
ExoJAX's line-list opa engines.

Format details (PSG xuv linelists):
    Header lines start with ``#``. The ``#TEMP:`` line lists one or
    more reference temperatures [K] (e.g. 1 column for many molecules,
    11 columns for O3's Serdyuchenko set). Data rows are
    ``wavelength_um  sigma_T1  sigma_T2 ...`` with cross-sections in
    cm^2/molecule. See
    https://psg.gsfc.nasa.gov/helpatm.php for the full catalog of
    available molecules.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def fetch_psg_xs(
    url: str,
    cache_filename: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a PSG cross-section file once and cache it locally.

    Idempotent: if the cached file already exists it is returned
    without re-downloading.

    Args:
        url: PSG download URL.
        cache_filename: Name to give the cached file under
            ``cache_dir``.
        cache_dir: Directory to cache into. Defaults to
            ``~/.cache/skyscapes/`` if None.

    Returns:
        Absolute path to the cached file.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "skyscapes"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / cache_filename
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def _load_psg_xs_table(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a PSG cross-section ASCII file.

    The PSG format begins with ``#``-prefixed header lines including a
    ``#TEMP:`` row listing the temperature axis [K] (one or more
    values), followed by data rows of
    ``wavelength_um  sigma_T1  sigma_T2 ... sigma_TN`` with
    cross-sections in cm^2/molecule.

    Args:
        path: Path to the PSG ASCII file.

    Returns:
        Tuple of ``(wavelength_um, T_grid_K, sigma)`` where ``sigma``
        has shape ``(n_wavelengths, n_temperatures)`` in cm^2/molecule.
    """
    T_grid = None
    with open(path) as f:
        for line in f:
            if line.startswith("#TEMP:"):
                tail = line.split(":", 1)[1].split("!", 1)[0]
                T_grid = np.array([float(x) for x in tail.split()])
                break
    if T_grid is None:
        raise ValueError(f"PSG xs file at {path} has no #TEMP: header line.")
    data = np.loadtxt(path, comments="#")
    wavelength_um = data[:, 0]
    sigma = data[:, 1:]
    if sigma.shape[1] != T_grid.size:
        raise ValueError(
            f"PSG xs file at {path}: #TEMP header lists {T_grid.size} "
            f"temperatures but data has {sigma.shape[1]} sigma columns."
        )
    return wavelength_um, T_grid, sigma


class PsgCrossSectionOpacity:
    """Cross-section-backed opacity from any PSG xs file.

    Quacks like ExoJAX's ``OpaPremodit``: exposes
    ``xsmatrix(Tarr, pressure) -> (n_layers, n_nu)`` and a ``molmass``
    attribute, so it slots into the same ``opa_engines`` tuple as the
    line-list-backed opacity engines.

    Pressure broadening is ignored. The PSG cross-section files
    describe electronic-continuum / pre-broadened absorption; the
    pressure-dependence at typical atmospheric pressures is below
    the photon-noise floor for HWO contrast levels. Temperature
    interpolation is linear against the file's temperature axis, with
    out-of-range temperatures clamped to the nearest endpoint.

    For files with a single temperature column the cross-section is
    returned independent of layer temperature.

    Attributes:
        name: Molecule name (e.g. ``"O3"``, ``"SO2"``).
        molmass: Molar mass [g/mol]. Caller supplies; used by
            ``opacity_profile_xs`` to convert mmr to column density.
        nu_grid: Wavenumber grid [cm^-1] this adapter was built for.
    """

    def __init__(
        self,
        name: str,
        url: str,
        molmass: float,
        nu_grid: Array | np.ndarray,
        cache_dir: str | Path | None = None,
        xs_table_path: str | Path | None = None,
        cache_filename: str | None = None,
    ):
        """Build the adapter by pre-interpolating to a wavenumber grid.

        Args:
            name: Molecule name (used in error messages and as
                ``self.name``).
            url: PSG download URL for this molecule.
            molmass: Molar mass [g/mol].
            nu_grid: Wavenumber grid [cm^-1], shape ``(n_nu,)``. Should
                match the rt_engine's ``nu_grid``.
            cache_dir: Directory for the cached PSG file. Defaults to
                ``~/.cache/skyscapes/`` if None.
            xs_table_path: Optional explicit path to a PSG-format
                cross-section file. Useful for tests so they can avoid
                a network fetch.
            cache_filename: Override the cached filename. Defaults to
                ``psg_<name_lower>_xs.txt``.
        """
        self.name = name
        self.molmass = molmass

        if xs_table_path is None:
            if cache_filename is None:
                cache_filename = f"psg_{name.lower()}_xs.txt"
            xs_table_path = fetch_psg_xs(url, cache_filename, cache_dir)
        wavelength_um, T_grid, sigma_2d = _load_psg_xs_table(xs_table_path)

        # PSG wavelengths are ascending in um; ascending um is descending
        # in wavenumber. Sort by wavenumber so np.interp gets a monotone
        # increasing xp array.
        nu_data = 1.0e4 / wavelength_um  # um -> cm^-1
        sort_idx = np.argsort(nu_data)
        nu_data = nu_data[sort_idx]
        sigma_2d = sigma_2d[sort_idx, :]

        # Pre-interpolate each temperature column to the rt_engine's
        # wavenumber grid. Out-of-range wavenumbers get 0 (no
        # absorption outside the file's coverage).
        nu_grid_np = np.asarray(nu_grid)
        sigma_at_nu = np.zeros((T_grid.size, nu_grid_np.size), dtype=np.float64)
        for j in range(T_grid.size):
            sigma_at_nu[j, :] = np.interp(
                nu_grid_np, nu_data, sigma_2d[:, j], left=0.0, right=0.0
            )

        self.nu_grid = jnp.asarray(nu_grid)
        self._T_grid = jnp.asarray(T_grid)
        self._sigma_at_nu = jnp.asarray(sigma_at_nu)

    def xsmatrix(self, Tarr: Array, pressure: Array) -> Array:
        """Cross-section matrix at the requested layer temperatures.

        ``pressure`` is accepted for interface parity with
        ``OpaPremodit`` but is unused.

        Args:
            Tarr: Layer temperatures [K], shape ``(n_layers,)``.
            pressure: Layer pressures [bar], shape ``(n_layers,)``;
                ignored.

        Returns:
            Cross-section matrix [cm^2/molecule], shape
            ``(n_layers, n_nu)``. Temperatures outside the PSG file's
            range clamp to the nearest endpoint cross-section. For
            single-temperature files the returned matrix is constant
            in T.
        """
        if self._T_grid.shape[0] == 1:
            # Single-temperature file: broadcast the only column.
            return jnp.broadcast_to(
                self._sigma_at_nu[0],
                (Tarr.shape[0], self._sigma_at_nu.shape[1]),
            )
        Tarr_safe = jnp.clip(Tarr, self._T_grid[0], self._T_grid[-1])

        def _interp_one_nu(sigma_T_col):
            return jnp.interp(Tarr_safe, self._T_grid, sigma_T_col)

        return jax.vmap(_interp_one_nu, in_axes=1, out_axes=1)(self._sigma_at_nu)
