"""Local HITRAN CIA tables (Karman et al. 2019, Icarus 328, 160).

ExoJAX 2.5's reader assumes a single rectangular spectral grid. Four of
these files instead contain multiple bands, changing grids, and sometimes
an uncertainty column. Its OpaCIA lookup also selects the next spectral bin
and does not zero spectral extrapolation. We preinterpolate each block in
wavenumber and interpolate coefficients linearly in temperature with JAX.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from exojax.database.cia.io import read_cia
from jaxtyping import Array

CIA_FILES = {
    "N2-N2": "N2-N2_2021.cia",
    "O2-O2": "O2-O2_2024.cia",
    "O2-N2": "O2-N2_2024.cia",
    "CO2-CO2": "CO2-CO2_2024.cia",
    "N2-H2O": "N2-H2O_2018.cia",
}


class TabulatedCIA(eqx.Module):
    """CIA coefficients [cm^5 molecule^-2] on the emission spectral grid.

    Bands carry independent temperature grids. Outside each spectral block
    the coefficient is zero; temperatures clamp to each band's endpoints.
    """

    partners: tuple[str, str] = eqx.field(static=True)
    temperatures: tuple[Array, ...]
    coefficients: tuple[Array, ...]
    n_nu: int = eqx.field(static=True)

    def coefficient_matrix(self, temperature_K: Array) -> Array:
        """Interpolate layer temperatures, returning shape (n_layers, n_nu)."""
        result = jnp.zeros((temperature_K.shape[0], self.n_nu))
        for temperatures, coefficients in zip(
            self.temperatures, self.coefficients, strict=True
        ):
            result = result + jax.vmap(
                lambda column, ts=temperatures: jnp.interp(temperature_K, ts, column),
                in_axes=1,
                out_axes=1,
            )(coefficients)
        return result


def cia_path(pair: str, directory: str | Path | None) -> Path:
    """Resolve a supported cached table without fetching or creating files."""
    if pair not in CIA_FILES:
        raise ValueError(
            f"Unsupported CIA pair {pair!r}; choose from {tuple(CIA_FILES)}"
        )
    directory = (
        Path.home() / ".cache" / "skyscapes" / "cia"
        if directory is None
        else Path(directory).expanduser()
    )
    path = directory / CIA_FILES[pair]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CIA file {path}. Obtain it from https://hitran.org/cia/; "
            "CIA files are never downloaded automatically."
        )
    return path


def _read_blocks(path: Path, pair: str):
    """Read HITRAN header/count blocks, accepting an extra uncertainty column."""
    if pair == "N2-H2O":
        # This file is rectangular and works with ExoJAX's native reader.
        nu, temperatures, coefficients = read_cia(str(path), -np.inf, np.inf)
        return [(t, nu, row) for t, row in zip(temperatures, coefficients, strict=True)]
    blocks = []
    with path.open() as stream:
        for header in stream:
            if not header.strip():
                continue
            fields = header.split()
            if fields[0] != pair:
                raise ValueError(f"Unexpected CIA header in {path}: {header.strip()}")
            count, temperature = int(fields[3]), float(fields[4])
            rows = np.array(
                [list(map(float, next(stream).split()[:2])) for _ in range(count)]
            )
            if (
                rows.shape != (count, 2)
                or not np.all(np.isfinite(rows))
                or np.any(np.diff(rows[:, 0]) <= 0)
            ):
                raise ValueError(f"Invalid CIA data block in {path}")
            # Experimental baselines can be negative; absorption cannot.
            blocks.append((temperature, rows[:, 0], np.maximum(rows[:, 1], 0.0)))
    return blocks


def load_cia(pair: str, directory: str | Path | None, nu_grid: Array) -> TabulatedCIA:
    """Load cached bands and prepare their coefficients on ``nu_grid``.

    Substantially overlapping blocks are temperature samples of the same band,
    including unsorted temperature sets with different spectral coverage.
    At a repeated temperature, later blocks take precedence only inside their
    coverage. Adjacent bands retain separate temperature grids and add in any
    small edge overlap; spectral gaps remain zero.
    """
    path = cia_path(pair, directory)
    nu_grid = np.asarray(nu_grid)
    blocks = _read_blocks(path, pair)
    # Temperature-dependent grids overlap over most of their width. Requiring
    # half the narrower interval separates adjacent bands with small edge overlaps
    # while joining the unsorted N2 fundamental temperature sets.
    ordered_blocks = sorted(enumerate(blocks), key=lambda item: item[1][1][0])
    bands = []
    for index, block in ordered_blocks:
        lo, hi = block[1][[0, -1]]
        if bands:
            band_lo, band_hi, _ = bands[-1]
            overlap = min(hi, band_hi) - lo
            same_band = overlap >= 0.5 * min(hi - lo, band_hi - band_lo)
        else:
            same_band = False
        if same_band:
            bands[-1][1] = max(bands[-1][1], hi)
            bands[-1][2].append((index, block))
        else:
            bands.append([lo, hi, [(index, block)]])
    temperatures, coefficients = [], []
    for _, _, band in bands:
        rows = {}
        for _, (t, nu, coefficient) in sorted(band, key=lambda item: item[0]):
            covered = (nu_grid >= nu[0]) & (nu_grid <= nu[-1])
            row = rows.setdefault(t, np.zeros_like(nu_grid, dtype=float))
            row[covered] = np.interp(nu_grid[covered], nu, coefficient)
        if not any(np.any(row) for row in rows.values()):
            continue
        ordered = sorted(rows)
        temperatures.append(jnp.asarray(ordered))
        coefficients.append(jnp.asarray(np.stack([rows[t] for t in ordered])))
    return TabulatedCIA(
        partners=tuple(pair.split("-")),
        temperatures=tuple(temperatures),
        coefficients=tuple(coefficients),
        n_nu=len(nu_grid),
    )
