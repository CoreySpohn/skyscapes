"""Resample the planet-type geometric albedo spectra onto a common 1 nm grid.

Reads the albedo files named in ``albedo_list.csv`` from an ExoVista data
directory and writes ``population/data/albedo_spectra.npz``. Each file holds
wavelength in micron and geometric albedo, as two columns. The spectra are
evaluated with a cubic spline that extrapolates, which is how the ExoVista
scene writer reads them, so the table stores the same albedo at any wavelength
between 300 and 2500 nm.

Usage: python tools/build_albedo_table.py <exovista data dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

GRID_NM = np.arange(300.0, 2501.0, 1.0)
OUT = (
    Path(__file__).resolve().parents[1]
    / "src/skyscapes/population/data/albedo_spectra.npz"
)


def _read(path):
    text = path.read_text().replace("\r\n", "\n").replace("\r", "\n")
    rows = [line.split() for line in text.split("\n") if line.strip()]
    if any(len(row) != 2 for row in rows):
        raise ValueError(f"{path.name} is not a two-column albedo file")
    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1]


def main(data_dir):
    """Write the albedo table from the ExoVista data directory `data_dir`."""
    data_dir = Path(data_dir)
    table = pd.read_csv(data_dir / "albedo_list.csv", skipinitialspace=True)
    names = [name.strip() for name in table["Files"]]
    albedo = np.zeros((len(names), GRID_NM.size))
    for k, name in enumerate(names):
        wavelength_um, value = _read(
            data_dir / "geometric_albedo_files" / f"{name}.txt"
        )
        spline = interp1d(
            wavelength_um,
            value,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        albedo[k] = spline(GRID_NM / 1000.0)
    np.savez_compressed(
        OUT, wavelength_nm=GRID_NM, albedo=albedo, names=np.array(names)
    )
    print(OUT, albedo.shape, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main(sys.argv[1])
